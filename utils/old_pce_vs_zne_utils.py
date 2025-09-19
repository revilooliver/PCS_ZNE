"""
Utils for running experiments comparing pce and zne.
"""

import numpy as np
# from cirq import Sampler
from qiskit.quantum_info import random_clifford, Pauli, Statevector
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Optional, Tuple, Dict

from qiskit import *
from qiskit.circuit import Qubit

from utils.pauli_checks import convert_to_PCS_circ # new util

from functools import partial

from mitiq import zne

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Reset
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate, CXGate, CZGate, SwapGate
from qiskit_experiments.library import StandardRB

import os
import csv
import re


def apply_measurement_basis(circuit, pauli_string):
    """Modify the circuit to measure in the basis specified by the Pauli string.
    """
    for i, pauli in enumerate(pauli_string):
        if pauli == 'X':
            circuit.h(i)
        elif pauli == 'Y':
            circuit.sdg(i)
            circuit.h(i)


def compute_expectation_value(counts, pauli_string):
    """Compute expectation value of a Pauli string observable from measurement counts.
    """
    total_shots = sum(counts.values())
    expectation = 0

    for bitstring, count in counts.items():
        # Reverse bitstring if needed to match Qiskit's little-endian convention
        # bitstring = bitstring[::-1]
        value = 1

        for i, pauli in enumerate(pauli_string):
            if pauli == 'I':
                continue
            elif pauli == 'Z' or pauli == 'X' or pauli == 'Y':
                # X and Y are already rotated to Z basis, so interpret as Z
                value *= 1 if bitstring[i] == '0' else -1
            else:
                raise ValueError(f"Invalid Pauli operator: {pauli}")

        expectation += value * count

    return expectation / total_shots

from qiskit.transpiler import Layout

def ibmq_executor(circuit: QuantumCircuit, backend, pauli_string: str, shots: int, layout):
    """Executor for ZNE.
       Computes the expectation value using the Estimator primitive.
    """


    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=layout)
    isa_circuit = pm.run(circuit)
    # print(f">>> Circuit ops (ISA): {isa_circuit.count_ops()}")

    observable = SparsePauliOp.from_list([(pauli_string, 1.0)])
    isa_observable = observable.apply_layout(isa_circuit.layout)
    print("isa observable: ", isa_observable)

    # print("backend = ", backend)
    estimator = Estimator(backend)
    estimator.options.default_shots = shots

    add_em_estimator(estimator)

    remove = RemoveBarriers()
    remove(isa_circuit)
    # --- NEW: draw the transpiled circuit inline ---
    # 1) ask for a matplotlib figure
    fig = isa_circuit.draw(output='mpl', fold=-1)
    # 2) display it in the notebook
    display(fig)
    plt.close(fig)   # (optional) prevent duplicate figures

    # print(isa_circuit)
    job = estimator.run([(isa_circuit, isa_observable)])

    # Get results for the first (and only) PUB
    pub_result = job.result()[0]
    # print("result: ", pub_result)

    print(f">>> Expectation value: {pub_result.data.evs}")
    expectation_value = pub_result.data.evs

    return expectation_value


def mitigate_zne(circ, backend, pauli_string, shots, method="richardson", scale_factors=[1,2,3], fold_method=zne.scaling.fold_global, layout=None):
    """
    Runs ibmq_executor and mitigates the expectation for 'pauli_string' observable using zne. Method set to default for now.
    """
    zne_executor = partial(ibmq_executor, backend=backend, pauli_string=pauli_string, shots=shots, layout=layout)

    if method == "richardson":
        factory = zne.inference.RichardsonFactory(scale_factors=scale_factors)
    elif method == "linear":
        factory = zne.inference.LinearFactory(scale_factors=scale_factors)
    elif method == "exp":
        factory = zne.inference.ExpFactory(scale_factors=scale_factors, asymptote=0.5)
    elif method == "poly":
        factory = zne.inference.PolyFactory(scale_factors=scale_factors, order=2)
    else:
        raise ValueError(f"Unsupported ZNE method: {method!r}")


    mitigated = zne.execute_with_zne(circ, zne_executor, factory=factory, scale_noise=fold_method)

    print("expectation values: ", factory.get_expectation_values())
    fig = factory.plot_fit()   
    plt.show()
    return mitigated


def filter_counts(no_checks, sign_list_in, in_counts):
    """
    Adjusts for minus signs.
    """
    sign_list = deepcopy(sign_list_in)
    sign_list.reverse()
    err_free_checks = ""
    for i in sign_list:
        if i == "+1":
            err_free_checks += "0"
        else:
            err_free_checks += "1"
            
    out_counts = {}
    for key in in_counts.keys():
        if err_free_checks == key[:no_checks]:
            new_key = key[no_checks:]
            out_counts[new_key] = in_counts[key]
    return out_counts

from IPython.display import display
from qiskit.transpiler.passes import RemoveBarriers

def add_em_sampler(samp: Sampler):
    samp.options.twirling.enable_measure=True # This is the trex twirling.
    samp.options.twirling.shots_per_randomization= "auto"
    samp.options.twirling.strategy= "active-circuit" #I usually use this setting, but there are other ones.
    samp.options.twirling.enable_gates=True

    print(f">>> gate twirling is turned on: {samp.options.twirling.enable_gates}")
    print(f">>> trex twriing is turned on: {samp.options.twirling.enable_measure}")



def add_em_estimator(est: Estimator):
    # Turn on gate twirling.
    est.options.twirling.enable_gates = True
    # Turn on measurement error mitigation.
    est.options.resilience.measure_mitigation = True

    print(f">>> gate twirling is turned on: {est.options.twirling.enable_gates}")
    print(f">>> measurement error mitigation is turned on: {est.options.resilience.measure_mitigation}")


def ibmq_executor_pcs(circuit: QuantumCircuit, backend, pauli_string: str, num_qubits, shots: int, signs = None, initial_layout = None):
    """Executor for PCS.
    """
    # Modify the circuit to measure the required Pauli observables
    measurement_circuit = circuit.copy()
    apply_measurement_basis(measurement_circuit, pauli_string)
    measurement_circuit.measure_all()
    print("circuit with appropriate measurements:")
    display(measurement_circuit.draw("mpl", fold=-1))

    pm = generate_preset_pass_manager(backend=backend, initial_layout=initial_layout, optimization_level=0) 
    isa_circuit = pm.run(measurement_circuit)
    print(f">>> Circuit ops (ISA): {isa_circuit.count_ops()}")
    print("transpiled circuit")
    display(isa_circuit.draw("mpl", fold=-1))
    # if initial_layout != None:
    #     print("virtual to physical mapping = ", isa_circuit.layout.final_virtual_layout(filter_ancillas=True))

    # Run the circuit
    sampler = Sampler(mode=backend)

    # add_em_sampler(sampler)

    remove = RemoveBarriers()
    isa_circuit = remove(isa_circuit)

    # # --- NEW: draw the transpiled circuit inline ---
    # # 1) ask for a matplotlib figure
    # fig = isa_circuit.draw(output='mpl', fold=-1)
    # # 2) display it in the notebook
    # display(fig)
    # plt.close(fig)   # (optional) prevent duplicate figures

    job = sampler.run([isa_circuit], shots=shots)
    results = job.result()
    counts = results[0].data.meas.get_counts()
    print("unmitigated counts: ", counts)

    # Filter counts based on check data
    total_qubits = circuit.num_qubits
    num_checks = total_qubits - num_qubits
    print("num checks = ", num_checks)
    filtered_counts = filter_counts(num_checks, signs, counts)
    print("filtered counts: ", filtered_counts)

    print("post selection rate: ", sum(filtered_counts.values()) / sum(counts.values()))
    print("number of filtered counts: ", sum(filtered_counts.values()))

    # Compute the expectation value based on filtered counts
    expectation_value = compute_expectation_value(filtered_counts, pauli_string)
    return expectation_value


# Generates circuits with specified depth
def random_clifford_circuit(num_qubits, depth, max_operands=2, measure=False,
                            conditional=False, reset=False, seed=None):
    """
    Generate a random circuit composed exclusively of Clifford gates.

    Only one-qubit and two-qubit Clifford gates are used. For one-qubit gates,
    the following are selected:
      IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate.
    For two-qubit Clifford gates, the following are selected:
      CXGate, CZGate, SwapGate.

    """
    if max_operands < 1 or max_operands > 2:
        raise ValueError("max_operands for Clifford circuits must be 1 or 2.")

    # Define allowed Clifford gates.
    # one_q_ops = [IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate]
    one_q_ops = [XGate, YGate, ZGate, HGate, SGate, SdgGate]
    # two_q_ops = [CXGate, CZGate, SwapGate]
    two_q_ops = [CXGate]

    # Add Reset gate if desired.
    if reset:
        one_q_ops.append(Reset)

    # Initialize registers.
    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qr)

    # Add a classical register if measurements or conditionals are desired.
    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    # Set the random seed.
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # Build the circuit layer by layer.
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        # Randomize the qubit order for each layer.
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            # For Clifford circuits, only 1- and 2-qubit gates are supported.
            # If a two-qubit gate is possible pick from [1, 2] randomly.
            if len(remaining_qubits) >= 2:
                num_operands = rng.choice([1, 2])
            else:
                num_operands = 1

            # For 2-qubit selection, choose the first two qubits.
            operands = [remaining_qubits.pop(0) for _ in range(num_operands)]

            # Pick appropriate random gate.
            if num_operands == 1:
                op_class = rng.choice(one_q_ops)
            elif num_operands == 2:
                op_class = rng.choice(two_q_ops)

            op = op_class()

            # Optionally add a conditional.
            if conditional and rng.choice(range(10)) == 0:
                # Pick a random bit value.
                value = rng.integers(0, 2 ** num_qubits)
                op.condition = (cr, value)
            qc.append(op, [qr[q] for q in operands])

    # Add measurements if requested.
    if measure:
        qc.measure(qr, cr)

    return qc

# Standard RB circuit generator
# def random_clifford_circuit(
#     num_qubits: int,
#     lengths: List[int] = [1],
#     num_samples: int = 1,
#     seed: Optional[int] = None,
# ) -> List[QuantumCircuit]:
#     """
#     Generate transpiled randomized-benchmarking circuits on `n_qubits`.

#     Args:
#         n_qubits: Total number of physical qubits to use (will be qubits 0..n_qubits-1).
#         lengths: RB sequence lengths (number of random Cliffords before the inverse).
#         num_samples: How many random instances per length.
#         seed: RNG seed for reproducibility.
#         backend: A Qiskit BackendV2 (if None, defaults to FakePerth()).
#         optimization_level: Qiskit transpiler optimization level (0–3).

#     Returns:
#         A list of `QuantumCircuit`s, each of depth `length+1` (includes global inverse),
#         transpiled to `backend`’s basis gates & coupling map.
#     """
#     physical_qubits = tuple(range(num_qubits))

#     exp = StandardRB(
#         physical_qubits,
#         lengths=lengths,
#         num_samples=num_samples,
#         seed=seed,
#     )

#     # Get RB sequences
#     sequences = exp._sample_sequences()

#     # Get the synthesis options and the “to_instruction” helper
#     synth_opts = exp._get_synthesis_options()

#     # Rebuild each circuit *without* appending the inverse
#     no_inv_circuits = []
#     for seq in sequences:
#         circ = QuantumCircuit(num_qubits)
#         for elem in seq:
#             inst = exp._to_instruction(elem, synth_opts)
#             circ.append(inst, circ.qubits)
#         no_inv_circuits.append(circ)

#     return no_inv_circuits

def random_cliff_circs(
    num_qubits: int,
    depth: int,
    num_circs: int,
    pauli_string: str,
    tol: float = 1e-8
) -> Tuple[List[QuantumCircuit], float]:
    """
    Returns:
      circs: List of Clifford circuits (each without inverse),
    """
    circs     = []
    while len(circs) < num_circs:
        # circ = random_clifford_circuit(
        #     num_qubits=num_qubits,
        #     lengths=[depth],
        #     num_samples=1
        # )[0].decompose()

        circ = random_clifford_circuit(num_qubits=num_qubits, depth=depth)

        expect = get_ideal_expectation(circ, pauli_string)
        if abs(abs(expect) - 1) < tol:
            circs.append(circ)

    return circs

from qiskit import qpy

def load_or_generate_random_cliffs(
    folder: str,
    num_qubits: int,
    circuit_depth: int,
    requested_num_circs: int,
    pauli_string: str
) -> List[QuantumCircuit]:
    """
    Load up to `requested_num_circs` from the smallest cached file 
    with N>=requested_num_circs, or else generate exactly
    requested_num_circs via random_cliff_circs.
    """
    # Pattern for filenames in the folder
    pattern = re.compile(
        rf"^rand_cliffs_n{num_qubits}_d{circuit_depth}_nc(\d+)\.qpy$"
    )
    candidates = []
    
    # 1) scan the folder for matching files
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            m = pattern.match(fname)
            if m:
                N = int(m.group(1))
                if N >= requested_num_circs:
                    candidates.append((N, fname))
    
    # 2) if we found any file with N >= requested, pick the smallest N
    if candidates:
        N_sel, fname_sel = min(candidates, key=lambda x: x[0])
        path = os.path.join(folder, fname_sel)
        with open(path, "rb") as f:
            all_circs = qpy.load(f)
        print(f"Loaded {N_sel} circuits from {path}; returning first {requested_num_circs}.")
        return all_circs[:requested_num_circs]
    
    # 3) otherwise generate fresh
    print(f"No cached file with ≥{requested_num_circs} circuits found.")
    print(f"Generating {requested_num_circs} new circuits (n={num_qubits}, d={circuit_depth})…")
    circs = random_cliff_circs(
        num_qubits=num_qubits,
        depth=circuit_depth,
        num_circs=requested_num_circs,
        pauli_string=pauli_string
    )
    
    # (Optionally: save them under the exact requested name for next time)
    out_fname = os.path.join(
        folder,
        f"rand_cliffs_n{num_qubits}_d{circuit_depth}_nc{requested_num_circs}.qpy"
    )
    os.makedirs(folder, exist_ok=True)
    with open(out_fname, "wb") as f:
        qpy.dump(circs, f)
    print(f"Saved the {requested_num_circs} new circuits to {out_fname}")
    
    return circs

# import itertools
# import numpy as np
# from typing import List, Tuple
# from qiskit import QuantumCircuit
# import time

# def random_cliff_circs(
#     num_qubits: int,
#     depth: int,
#     num_circs: int,
#     tol: float = 1e-8
# ) -> Tuple[List[QuantumCircuit], List[str], List[int]]:
#     """
#     Generate `num_circs` random Clifford circuits of given `depth`
#     and for each one search for a Pauli string P with |⟨P⟩| ≈ 1.
#     Pauli strings are tried in increasing weight until one is found.

#     Returns
#     -------
#     circs       : List of the Clifford circuits
#     pauli_strs  : List of the Pauli string found for each circuit
#     signs       : List of +1 or -1 for ⟨P⟩ on that circuit
#     """
#     # Pre-generate all non-trivial Pauli strings sorted by Hamming weight
#     chars = ['X','Y','Z']
#     pauli_list = [
#         ''.join(p)
#         for p in itertools.product(chars, repeat=num_qubits)
#         if any(c!='I' for c in p)
#     ]
#     total_paulis = len(pauli_list)
#     print("total paulis: ", total_paulis)

#     circs       = []
#     pauli_strs  = []
#     signs       = []

#     print(f"→ Generating {num_circs} circuits (n={num_qubits}, depth={depth})")
#     while len(circs) < num_circs:
#         circ = random_clifford_circuit(
#             num_qubits=num_qubits,
#             lengths=[depth],
#             num_samples=1
#         )[0].decompose()

#         start_time = time.time()

#         for p_idx, p in enumerate(pauli_list, start=1):
#             if p_idx % 1000 == 0 or p_idx == 1:
#                 elapsed = time.time() - start_time
#                 print(f"  Checked {p_idx}/{total_paulis} Pauli strings  (elapsed {elapsed:.1f}s)", end='\r')

#             expect = get_ideal_expectation(circ, p)
#             if abs(abs(expect) - 1) < tol:
#                 sign = int(np.sign(expect))
#                 circs.append(circ)
#                 pauli_strs.append(p)
#                 signs.append(sign)
#                 break

#         idx = len(circs)
#         print(f"  → Generated circuit {idx}/{num_circs}")
#     print(f"→ Finished generating {num_circs} circuits.")
#     return circs, pauli_strs, signs

# def random_cliff_circs(num_qubits, depth, num_circs, pauli_string, tol=1e-8):
#     circs = []
#     while len(circs) < num_circs:
#         # circ = random_clifford_circuit(num_qubits=num_qubits, depth=depth)

#         # Compute the ideal expectation for this circuit using the given observable.
#         expect = get_ideal_expectation(circ, pauli_string)

#         # Check if the ideal expectation is close to +1 or -1.
#         if abs(abs(expect) - 1) < tol:
#             circs.append(circ)
#             print(len(circs), " generated")

#     return circs

def get_pcs_circs(circ, num_checks, only_Z_checks=False):
    """
    Returns a list of circs with 1 check, 2 checks, ... up to 'num_checks' checks
    """
    num_qubits = circ.num_qubits
    print("num qubits = ", num_qubits)
    circs_list = []
    signs_list = []
    for check_id in range(1, num_checks + 1):
        print("generating check circ #", check_id)
        sign, circ = convert_to_PCS_circ(circ, num_qubits, check_id, only_Z_checks=only_Z_checks)
        circs_list.append(circ)
        signs_list.append(sign)

    return circs_list, signs_list


import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import lagrange


def extrapolate_checks(num_checks_to_fit: int, extrap_checks, expectation_values, method: str = 'linear'):
    """
    Fit an extrapolation model to the first `num_checks_to_fit` expectation values,
    and extrapolate to the desired check numbers.

    Parameters:
    - num_checks_to_fit: int, number of initial data points to use for fitting.
    - extrap_checks: iterable of int, check numbers to extrapolate to.
    - expectation_values: list or array-like, observed values.
    - method: str, either "linear" or "exponential" (default is "linear").

    Returns:
    - extrap_values: list of extrapolated values, one for each value in `extrap_checks`.
    - fit_func: the fitting function (polynomial for linear, callable for exponential).
    """
    # Use check numbers starting at 1 up to num_checks_to_fit.
    check_numbers = np.array(range(1, num_checks_to_fit + 1))
    y_data = np.array(expectation_values[:num_checks_to_fit])

    if method == 'linear':
        # Fit a degree-1 polynomial (linear regression)
        coeffs = np.polyfit(check_numbers, y_data, 1)
        # Define the polynomial function.
        fit_func = np.poly1d(coeffs)
        extrap_values = [fit_func(c) for c in extrap_checks]

        # Plot the data and the fitted line
        plt.figure()
        plt.scatter(check_numbers, y_data, label='Data Points')
        m_fit = np.linspace(check_numbers.min(), max(extrap_checks), 100)
        plt.plot(m_fit, fit_func(m_fit),
                 label=f'Linear fit: slope={coeffs[0]:.3f}, intercept={coeffs[1]:.3f}')
        plt.xlabel('Check Number')
        plt.ylabel('Expectation Value')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.show()
    
    elif method == 'exponential':
        # Define the exponential model: E(m) = a * b^m + c
        def exp_model(m, a, b, c):
            # print(f"DEBUG exp_model called with m={m}, a={a}, b={b}, c={c}")
            return a * (b ** m) + c

        # Provide an initial guess for [a, b, c]
        initial_guess = [1.0, 0.9, 0.0]

        lower_bounds = [-np.inf, 0.6, -np.inf]
        upper_bounds = [np.inf, 1.2, np.inf]

        popt, _ = curve_fit(exp_model, check_numbers, y_data, p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds), maxfev=10000)
        # print(popt)
        # Create a lambda function for easy extrapolation.
        fit_func = lambda m: exp_model(m, *popt)
        extrap_values = [fit_func(c) for c in extrap_checks]

        plt.figure()
        plt.scatter(check_numbers, y_data, label='Data Points', color='blue')
        m_fit = np.linspace(min(check_numbers), max(extrap_checks), 100)
        plt.plot(m_fit, fit_func(m_fit), label=f'Fitted: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}',
                 color='red')
        plt.xlabel('Check Number')
        plt.ylabel('Expectation Value')
        plt.title('Exponential Curve Fit')
        plt.legend()
        plt.show()

    else:
        raise ValueError("Unsupported method. Please use 'linear' or 'exponential'.")

    return extrap_values, fit_func

def get_ideal_expectation(circ, pauli_string):
    """
    Calculates the ideal expectation of the state prepared by 'circ', wrt the observable 'pauli_string'.
    """
    operator = Pauli(pauli_string)
    psi = Statevector(circ)
    expect = np.array(psi).T.conj() @ operator.to_matrix() @ np.array(psi)
    return expect

#----------------------------------------------------
# Functions for saving/loading data and plotting
#----------------------------------------------------

def save_avg_errors(circ_folder, filename, avg_errors, avg_cx=None, overwrite=False):
    """
    Save average error results for multiple methods, embedding the average CX count
    in the filename if provided.

    Parameters:
      circ_folder (str): Subfolder under "data_PCE_vs_ZNE".
      filename (str): Base filename for the CSV (without .csv extension).
      avg_errors (dict): Dictionary with keys as method names and values as their average error.
      avg_cx (float or None): If provided, inserts the average CX count into the filename.
      overwrite (bool): Whether to overwrite the file if it already exists. Defaults to False.
    """
    os.makedirs(circ_folder, exist_ok=True)

    # Build filename with CX count if provided
    base, ext = os.path.splitext(filename)
    if avg_cx is not None:
        # round or format avg_cx as desired
        cx_str = f"cx={avg_cx:.1f}"
        base = f"{base}_{cx_str}"
    final_filename = f"{base}{ext or '.csv'}"
    filepath = os.path.join(circ_folder, final_filename)

    if os.path.exists(filepath) and not overwrite:
        print(f"File {filepath} already exists. To overwrite, set overwrite=True.")
        return

    # Write only the method-error table
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Average Absolute Error"])
        for method, error in avg_errors.items():
            writer.writerow([method.upper(), error])

    print(f"Results successfully saved to {filepath}")

# def save_avg_errors(circ_folder, filename, avg_errors, overwrite=False):
#     """
#     Save average error results for multiple methods without overwriting an existing file.

#     Parameters:
#       circ_folder (str): Subfolder under "data_PCE_vs_ZNE".
#       filename (str): Filename for the CSV.
#       avg_errors (dict): Dictionary with keys as method names (e.g., "ZNE_linear", "ZNE_default", "PCE")
#                          and values as their corresponding average error.
#       overwrite (bool): Whether to overwrite the file if it already exists. Defaults to False.
#     """
#     dir_path = circ_folder
#     os.makedirs(dir_path, exist_ok=True)  # Ensure the subfolder exists

#     filepath = os.path.join(dir_path, filename)
#     if os.path.exists(filepath) and not overwrite:
#         print(f"File {filepath} already exists. To overwrite, set overwrite=True.")
#         return

#     with open(filepath, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Method", "Average Absolute Error"])
#         for method, error in avg_errors.items():
#             writer.writerow([method.upper(), error])
#     print(f"Results successfully saved to {filepath}")



def load_avg_errors(circ_folder, num_circs, num_samples, depth):
    """
    Load average error results from CSV files for multiple methods.

    The function searches for files named with the pattern:
       avg_errors_n={num_qubits}_...depth={depth}..._num_circs={num_circs}_num_samp={num_samples}.csv

    Parameters:
      circ_folder (str): The folder containing the saved CSV files.
      num_circs (int): The number of circuits used (embedded in the filename).
      num_samples (int): The number of samples used (embedded in the filename).
      depth (int): The circuit depth to filter the files.

    Returns:
      dict: A dictionary mapping num_qubits to another dictionary with keys as method names
            (e.g., 'zne_linear', 'zne_default', 'pce') and values as the corresponding average error.
            For example:
                { 5: {'zne_linear': 0.0123, 'zne_default': 0.0456, 'pce': 0.0789}, ... }
    """
    data = {}
    # List all CSV files in the provided folder.
    for file in os.listdir(circ_folder):
        if file.endswith(".csv"):
            # Only consider files that include num_circs, num_samples, and depth in their names.
            if (f"num_circs={num_circs}" in file and
                    f"num_samp={num_samples}" in file and
                    f"d={depth}" in file):
                # Extract number of qubits from the file name using regex.
                # Expected file pattern: "avg_errors_n={num_qubits}_..."
                match = re.search(r"avg_errors_n=(\d+)_", file)
                if match:
                    num_qubits = int(match.group(1))
                    with open(os.path.join(circ_folder, file), mode='r', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader)  # Skip header
                        errors = {}
                        for row in reader:
                            # Expecting each row: [Method, Average Absolute Error]
                            method, error = row
                            errors[method.lower()] = float(error)
                    data[num_qubits] = errors
    return data

def plot_avg_errors_by_qubit(data, num_samples=None, num_circs=None, depth=None, save_path=None):
    """
    Plot average errors for multiple methods grouped by number of qubits.

    Parameters:
      data (dict): {num_qubits: {method_name: error, ...}, ...}
      num_samples (int, optional): Number of samples, used in the plot title.
      num_circs (int, optional): Number of circuits, used in the plot title.
      depth (int, optional): Circuit depth, used in the plot title.
      save_path (str, optional): Path to save the resulting plot.
    """
    # Determine all methods present across the data
    methods = set()
    for errors in data.values():
        methods.update(errors.keys())
    methods = sorted(methods)
    num_methods = len(methods)

    # Get a sorted list of qubit counts
    qubit_list = sorted(data.keys())
    x = np.arange(len(qubit_list))
    width = 0.8 / num_methods  # Adjust width for grouped bars

    fig, ax = plt.subplots()
    for i, method in enumerate(methods):
        # Extract error values for each qubit count for the given method.
        method_errors = [data[q].get(method, None) for q in qubit_list]
        # Bar positions for this method
        bar_positions = x - 0.4 + i * width + width / 2
        bars = ax.bar(bar_positions, method_errors, width, label=method)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Average Absolute Error")
    title_parts = []
    if num_samples is not None:
        title_parts.append(f"total # of samples = {num_samples}")
    if num_circs is not None:
        title_parts.append(f"# of circuits = {num_circs}")
    if depth is not None:
        title_parts.append(f"depth = {depth}")
    title_str = " | ".join(title_parts) if title_parts else "Comparison to ZNE"
    ax.set_title(title_str)
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_list)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_avg_errors_by_depth(data, num_samples=None, num_circs=None, qubits=None, save_path=None):
    """
    Plot average errors for multiple methods grouped by circuit depth.

    Parameters:
      data (dict): {depth: {method_name: error, ...}, ...}
      num_samples (int, optional): Number of samples, used in the plot title.
      num_circs (int, optional): Number of circuits, used in the plot title.
      qubits (int, optional): Number of qubits (fixed), used in the plot title.
      save_path (str, optional): Path to save the resulting plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Determine all methods present across the data
    methods = set()
    for errors in data.values():
        methods.update(errors.keys())
    methods = sorted(methods)
    num_methods = len(methods)

    # Get a sorted list of circuit depths
    depths = sorted(data.keys())
    x = np.arange(len(depths))
    width = 0.8 / num_methods  # Adjust width for grouped bars

    fig, ax = plt.subplots()
    for i, method in enumerate(methods):
        # Extract error values for each depth for the given method.
        method_errors = [data[d].get(method, None) for d in depths]
        # Bar positions for this method
        bar_positions = x - 0.4 + i * width + width / 2
        bars = ax.bar(bar_positions, method_errors, width, label=method)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Average Absolute Error")
    title_parts = []
    if num_samples is not None:
        title_parts.append(f"# of samples = {num_samples}")
    if num_circs is not None:
        title_parts.append(f"# of circuits = {num_circs}")
    if qubits is not None:
        title_parts.append(f"# of qubits = {qubits}")
    title_str = " | ".join(title_parts) if title_parts else "Comparison of Error vs Depth"
    ax.set_title(title_str)
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()




    ##########################
    # Hardware experiments
    ##########################

    from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import networkx as nx
from utils.pauli_checks import convert_to_PCS_circ

def build_qaoa_ansatz(
    graph: nx.Graph,
    p: int,
    barriers: bool = True,
    gamma_vals: list[float] | None = None,
    beta_vals:  list[float] | None = None
) -> QuantumCircuit:
    """
    Build a depth-p QAOA ansatz on `graph` without any PCS checks.
    
    Returns:
        QuantumCircuit: a parameter-bound QAOA circuit.
    """
    n = graph.number_of_nodes()
    # define symbolic parameters
    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    # total qubits = data qubits only
    qc = QuantumCircuit(n)
    # initial layer: Hadamards on all data qubits
    # qc.h(range(n))

    # alternate phase-separators and mixers
    for layer in range(p):
        # phase separator for this layer
        phase = QuantumCircuit(n)
        for i, j in graph.edges():
            phase.cx(i, j)
            phase.rz(2 * gamma[layer], j)
            phase.cx(i, j)

        # append phase separator
        qc.compose(phase, qubits=range(n), inplace=True)
        if barriers:
            qc.barrier(range(n))

        # mixer layer
        for q in range(n):
            qc.rx(2 * beta[layer], q)

        if barriers and layer < p - 1:
            qc.barrier(range(n))

    # bind values (default zeros if none provided)
    if gamma_vals is None:
        gamma_vals = [0.0] * p
    if beta_vals is None:
        beta_vals = [0.0] * p
    if len(gamma_vals) != p or len(beta_vals) != p:
        raise ValueError("gamma_vals and beta_vals must each have length p")

    bind_dict = {gamma[i]: gamma_vals[i] for i in range(p)}
    bind_dict.update({beta[i]:  beta_vals[i]  for i in range(p)})
    qc = qc.assign_parameters(bind_dict)

    return qc


def build_pcs_qaoa_ansatz(
    graph: nx.Graph,
    p: int,
    num_checks: int,
    barriers: bool = True,
    only_Z_checks: bool = True,
    gamma_vals: list[float] | None = None,
    beta_vals:  list[float] | None = None
) -> tuple[list[list[str]], QuantumCircuit]:
    """
    Build a depth-p QAOA ansatz on `graph` with each phase-separator
    (the CX–RZ–CX per edge) wrapped by `num_checks` PCS checks.
    The PCS block is fully inlined, so drawing shows all individual gates.
    """
    n = graph.number_of_nodes()
    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    total_qubits = n + num_checks
    qc = QuantumCircuit(total_qubits)
    # qc.h(range(n))  # initial |+>^⊗n

    for layer in range(p):
        # 1) build just the plain phase-separator on n qubits
        phase = QuantumCircuit(n)
        for i, j in graph.edges():
            phase.cx(i, j)
            phase.rz(2 * gamma[layer], j)
            phase.cx(i, j)

        # 2) wrap it with PCS checks (returns a full (n+num_checks)-qubit circuit)
        signs, pcs_phase = convert_to_PCS_circ(
            phase,
            num_qubits=n,
            num_checks=num_checks,
            barriers=barriers,
            only_Z_checks=only_Z_checks
        )

        # 3) inline all of pcs_phase’s gates into qc
        #    qubit ordering: first n are data, next num_checks are ancillas
        qc.compose(pcs_phase, qubits=range(total_qubits), inplace=True)

        if barriers:
            qc.barrier(range(total_qubits))

        # 4) mixer on data qubits
        for q in range(n):
            qc.rx(2 * beta[layer], q)

        if barriers and layer < p - 1:
            qc.barrier(range(total_qubits))

    # bind γ,β (default to 0’s if none supplied)
    if gamma_vals is None: gamma_vals = [0.0]*p
    if beta_vals  is None: beta_vals  = [0.0]*p
    if len(gamma_vals)!=p or len(beta_vals)!=p:
        raise ValueError("gamma_vals and beta_vals must each have length p")

    bind_dict = {gamma[i]: gamma_vals[i] for i in range(p)}
    bind_dict.update({beta[i]:  beta_vals[i]  for i in range(p)})
    qc = qc.assign_parameters(bind_dict)

    return signs, qc

def get_line_initial_layout(
    qc: QuantumCircuit,
    num_system: int,
    num_checks: int
) -> Dict[Qubit, int]:
    """
    Build an initial_layout so that on a line of N physical qubits:
      - system qubit j  ↦ physical 2*j
      - ancilla qubit j ↦ physical 2*j + 1
    The circuit qc must have qc.num_qubits == num_system + num_checks,
    with system qubits in qc.qubits[0:num_system] and ancillas in qc.qubits[num_system:].
    """
    N = qc.num_qubits
    M = num_system
    K = num_checks
    if M + K != N:
        raise ValueError(f"Need num_system+num_checks == {N}, got {M}+{K}")
    if K > M:
        raise ValueError(f"num_checks ({K}) can’t exceed num_system ({M})")

    layout: Dict[Qubit, int] = {}

    # Pair the first K system qubits with the K ancillas:
    for j in range(K):
        sys_q = qc.qubits[j]
        anc_q = qc.qubits[M + j]
        layout[sys_q] = 2*j
        layout[anc_q] = 2*j + 1

    # Pack any leftover system qubits into the remaining slots 2K … N-1
    free_positions = list(range(2*K, N))
    leftover_sys = qc.qubits[K:M]
    for phys, q in zip(free_positions, leftover_sys):
        layout[q] = phys

    return layout

