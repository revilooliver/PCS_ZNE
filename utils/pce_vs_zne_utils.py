"""
Utils for running experiments comparing pce and zne.
"""

import numpy as np
from cirq import Sampler
from qiskit.quantum_info import random_clifford, Pauli, Statevector
import matplotlib.pyplot as plt
from copy import deepcopy

from qiskit import *

from utils.pauli_checks import convert_to_PCS_circ # new util

from functools import partial

from mitiq import zne

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


from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# def run_zne_estimator(circuits: list, backend, pauli_string: str, shots: int = 10_000, zne=True):
#     """
#     Run Qiskit's Estimator primitive with ZNE.
#     """

#     # Prepare a common base observable.
#     base_observable = SparsePauliOp.from_list([(pauli_string, 1.0)])
#     job_data = []

#     # Prepare the pass manager once.
#     pm = generate_preset_pass_manager(backend=backend, optimization_level=0)

#     # Process each circuit.
#     for circ in circuits:
#         isa_circ = pm.run(circ)
#         # Adjust the observable to match the transpiled circuit layout.
#         isa_observable = base_observable.apply_layout(isa_circ.layout)
#         job_data.append((isa_circ, isa_observable))

#     estimator = Estimator(backend)
#     estimator.options.default_shots = shots

#     if zne:
#         estimator.options.resilience.zne_mitigation = True
#         estimator.options.resilience.zne.noise_factors = (1, 3, 5)
#         estimator.options.resilience.zne.extrapolator = ("exponential", "linear")

#     # Submit all circuits at once.
#     job = estimator.run(job_data)
#     print(f">>> Job ID: {job.job_id()}")
#     print(f">>> Job Status: {job.status()}")
#     results = job.result()
#     # print("results: ", results)

#     # Extract expectation values from results.
#     expectation_values = [res.data.evs for res in results]
#     for ev in expectation_values:
#         print(f">>> Expectation value: {ev}")

#     return expectation_values


    # Create the Estimator. Depending on your version of Estimator, you might need
    # to initialize without passing in backend if it doesn't accept it.
    # estimator = Estimator(backend)
    # estimator.options.default_shots = shots
    #
    # # Submit all circuits at once.
    # job = estimator.run(job_data)
    # results = job.result()
    # print("results: ", results)
    #
    # # Extract expectation values from results.
    # expectation_values = [res.data.evs for res in results]
    # for ev in expectation_values:
    #     print(f">>> Expectation value: {ev}")
    #
    # return expectation_values


def ibmq_executor(circuit: QuantumCircuit, backend, pauli_string: str, shots: int = 10_000):
    """Executor for ZNE.
       Computes the expectation value using the Estimator primitive.
    """

    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    isa_circuit = pm.run(circuit)
    print(f">>> Circuit ops (ISA): {isa_circuit.count_ops()}")

    observable = SparsePauliOp.from_list([(pauli_string, 1.0)])
    isa_observable = observable.apply_layout(isa_circuit.layout)

    estimator = Estimator(backend)
    estimator.options.default_shots = shots
    # print(isa_circuit)
    job = estimator.run([(isa_circuit, isa_observable)])

    # Get results for the first (and only) PUB
    pub_result = job.result()[0]
    # print("result: ", pub_result)

    # print(f">>> Expectation value: {pub_result.data.evs}")
    expectation_value = pub_result.data.evs

    return expectation_value


### Old function that uses counts to compute expectation value.
# def ibmq_executor(circuit: QuantumCircuit, shots: int = 10_000):
# def ibmq_executor(circuit: QuantumCircuit, backend, pauli_string: str, shots: int = 10_000):
#     """Executor for ZNE.
#     """
#     # Modify the circuit to measure the required Pauli observables
#     measurement_circuit = circuit.copy()
#     measurement_circuit.barrier()
#     apply_measurement_basis(measurement_circuit, pauli_string)
#     measurement_circuit.measure_all()
#     # print(measurement_circuit)
#
#     # Transpile for the backend
#     exec_circuit = transpile(
#         measurement_circuit,
#         backend=backend,
#         optimization_level=0 # Preserve gate structure for simulation accuracy.
#     )
#
#     # print("transpiled circuit")
#     # print(exec_circuit)
#
#     # Run the circuit
#     job = backend.run(exec_circuit, shots=shots)
#     counts = job.result().get_counts()
#
#     # Compute the expectation value based on counts
#     # expectation_value = sum((-1 if (bin(int(state, 16)).count('1') % 2) else 1) * count for state, count in counts.items()) / shots
#     expectation_value = compute_expectation_value(counts, pauli_string)
#     return expectation_value

def mitigate_zne(circ, backend, pauli_string, shots=10_000, method="richardson", scale_factors=[1,2,3], fold_method=zne.scaling.fold_global):
    """
    Runs ibmq_executor and mitigates the expectation for 'pauli_string' observable using zne. Method set to default for now.
    """
    zne_executor = partial(ibmq_executor, backend=backend, pauli_string=pauli_string, shots=shots)

    if method == "richardson":
        factory = zne.inference.RichardsonFactory(scale_factors=scale_factors)
        mitigated = zne.execute_with_zne(circ, zne_executor, factory=factory, scale_noise=fold_method)

    elif method == "linear":
        factory = zne.inference.LinearFactory(scale_factors=scale_factors)
        mitigated = zne.execute_with_zne(circ, zne_executor, factory=factory, scale_noise=fold_method)

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

# def run_circs_pcs
#     sampler = Sampler(backend)
#
#             # Submit all circuits at once.
#             job = sampler.run(job_data) # might not want to pass observable here.
#             print(f">>> Job ID: {job.job_id()}")
#             print(f">>> Job Status: {job.status()}")
#             results = job.result()
#             print("results: ", results)
#
#             # Extract expectation values from results.
#             expectation_values = [res.data.evs for res in results]
#             for ev in expectation_values:
#                 print(f">>> Expectation value: {ev}")

from qiskit_ibm_runtime import SamplerV2 as Sampler

def ibmq_executor_pcs(circuit: QuantumCircuit, backend, pauli_string: str, num_qubits, shots: int = 10_000, signs = None):
    """Executor for PCS.
    """
    # Modify the circuit to measure the required Pauli observables
    measurement_circuit = circuit.copy()
    apply_measurement_basis(measurement_circuit, pauli_string)
    measurement_circuit.measure_all()
    # print(measurement_circuit)

    # Transpile for the backend
    # exec_circuit = transpile(
    #     measurement_circuit,
    #     backend=backend,
    #     # basis_gates=['z', 'y', 'x', 's', 'sdg', 't', 'tdg', 'h', 'cx', 'cy', 'cz'], # test set
    #     optimization_level=0 # keep at level 0 for mirror circuits, or it will cancel out all the gates.
    # )

    # print("transpiled circuit")
    # print(exec_circuit)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=0) # Changed optimization level to 3 temporarily 
    isa_circuit = pm.run(measurement_circuit)
    print(f">>> Circuit ops (ISA): {isa_circuit.count_ops()}")

    # Run the circuit
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)
    results = job.result()
    # print("results: ", results)
    counts = results[0].data.meas.get_counts()
    # counts = results[0]
    # print("counts: ", counts)

    # Retrieve the quasi-distribution for our single circuit and convert it to counts.
    # quasi_dists = results[0].data.quasi_dists
    # counts = {bitstr: int(round(prob * shots)) for bitstr, prob in quasi_dists.items()}

    # job = backend.run(exec_circuit, shots=shots)
    # print(job.result().quasi_dists)

    
    # counts = job.result().get_counts()
    # print("counts: ", counts)
    # print()

    # Filter counts based on check data
    total_qubits = circuit.num_qubits
    num_checks = total_qubits - num_qubits
    # filtered_counts = filter_counts(counts, num_checks)
    filtered_counts = filter_counts(num_checks, signs, counts)
    # print("filtered_counts: ", filtered_counts)

    # Compute the expectation value based on filtered counts
    expectation_value = compute_expectation_value(filtered_counts, pauli_string)
    return expectation_value


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Reset
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate, CXGate, CZGate, SwapGate


def random_clifford_circuit(num_qubits, depth, max_operands=2, measure=False,
                            conditional=False, reset=False, seed=None):
    """
    Generate a random circuit composed exclusively of Clifford gates.

    Only one-qubit and two-qubit Clifford gates are used. For one-qubit gates,
    the following are selected:
      IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate.
    For two-qubit Clifford gates, the following are selected:
      CXGate, CZGate, SwapGate.

    Args:
        num_qubits (int): Number of qubits.
        depth (int): Number of layers of operations (critical path length).
        max_operands (int): Maximum number of qubits acted on by an operation.
                            Currently supported values are 1 or 2.
        measure (bool): If True measure all qubits at the end.
        conditional (bool): If True, randomly add conditional operations.
        reset (bool): If True, allow Reset operations in the one-qubit gate set.
        seed (int): Seed for random number generator (optional).

    Returns:
        QuantumCircuit: A randomly generated Clifford circuit.
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


def random_cliff_circs(num_qubits, depth, num_circs, pauli_string, tol=1e-8):
    circs = []
    while len(circs) < num_circs:
        # clifford_obj = random_clifford(num_qubits)
        # circ = clifford_obj.to_circuit()

        circ = random_clifford_circuit(num_qubits=num_qubits, depth=depth)

        # Compute the ideal expectation for this circuit using the given observable.
        expect = get_ideal_expectation(circ, pauli_string)

        # Check if the ideal expectation is close to +1 or -1.
        if abs(abs(expect) - 1) < tol:
            circs.append(circ)

        # circs.append(circ)

    return circs

def random_cliff_mirror_circs(num_qubits, num_circs):
    circs = []
    for _ in range(num_circs):
        clifford_obj = random_clifford(num_qubits)
        circ = clifford_obj.to_circuit()

        inverse_circ = circ.inverse()

        circuit = QuantumCircuit(num_qubits)

        circuit.compose(circ, range(num_qubits), inplace=True)
        circuit.compose(inverse_circ, range(num_qubits), inplace=True)

        circs.append(circuit)

    return circs


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
    elif method == 'exponential':
        # Define the exponential model: E(m) = a * b^m + c
        def exp_model(m, a, b, c):
            return a * (b ** m) + c

        # Provide an initial guess for [a, b, c]
        initial_guess = [1.0, 0.9, 0.0]
        # popt, _ = curve_fit(exp_model, check_numbers, y_data, p0=initial_guess, maxfev=10000)

        lower_bounds = [-np.inf, 0.5, -np.inf]
        upper_bounds = [np.inf, 2.0, np.inf]
        popt, _ = curve_fit(exp_model, check_numbers, y_data, p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds), maxfev=10000)
        # Create a lambda function for easy extrapolation.
        fit_func = lambda m: exp_model(m, *popt)
        extrap_values = [fit_func(c) for c in extrap_checks]

        # plt.figure()
        # plt.scatter(check_numbers, y_data, label='Data Points', color='blue')
        # m_fit = np.linspace(min(check_numbers), max(extrap_checks), 100)
        # plt.plot(m_fit, fit_func(m_fit), label=f'Fitted: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}',
        #          color='red')
        # plt.xlabel('Check Number')
        # plt.ylabel('Expectation Value')
        # plt.title('Exponential Curve Fit')
        # plt.legend()
        # plt.show()

    else:
        raise ValueError("Unsupported method. Please use 'linear' or 'exponential'.")

    return extrap_values, fit_func

# def extrapolate_checks(num_checks_to_fit: int, extrap_checks, expectation_values):
#     """
#     Fit a linear model to the first `num_checks_to_fit` expectation values,
#     and extrapolate to multiple check numbers.
#
#     Parameters:
#     - num_checks_to_fit: int, number of initial data points to use for fitting
#     - extrap_checks: iterable of int, check numbers to extrapolate to
#     - expectation_values: list or array-like, observed values
#
#     Returns:
#     - list of extrapolated values, one for each value in `extrap_checks`
#     """
#     check_numbers = range(1, num_checks_to_fit + 1)
#
#     # Fit a degree-1 polynomial (linear regression)
#     polynomial_coefficients = np.polyfit(check_numbers, expectation_values[:num_checks_to_fit], 1)
#     polynomial = np.poly1d(polynomial_coefficients)
#
#     # Extrapolate to all specified check numbers
#     extrap_values = [polynomial(c) for c in extrap_checks]
#
#     return extrap_values, polynomial

def get_ideal_expectation(circ, pauli_string):
    """
    Calculates the ideal expectation of the state prepared by 'circ', wrt the observable 'pauli_string'.
    """
    operator = Pauli(pauli_string)
    psi = Statevector(circ)
    expect = np.array(psi).T.conj() @ operator.to_matrix() @ np.array(psi)
    return expect

# def get_pcs_circs(num_circs

def save_avg_errors(circ_folder, filename, avg_errors, overwrite=False):
    """
    Save average error results for multiple methods without overwriting an existing file.

    Parameters:
      circ_folder (str): Subfolder under "data_PCE_vs_ZNE".
      filename (str): Filename for the CSV.
      avg_errors (dict): Dictionary with keys as method names (e.g., "ZNE_linear", "ZNE_default", "PCE")
                         and values as their corresponding average error.
      overwrite (bool): Whether to overwrite the file if it already exists. Defaults to False.
    """
    dir_path = os.path.join("data_PCE_vs_ZNE", circ_folder)
    os.makedirs(dir_path, exist_ok=True)  # Ensure the subfolder exists

    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath) and not overwrite:
        print(f"File {filepath} already exists. To overwrite, set overwrite=True.")
        return

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Average Absolute Error"])
        for method, error in avg_errors.items():
            writer.writerow([method.upper(), error])
    print(f"Results successfully saved to {filepath}")


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
        title_parts.append(f"# of samples = {num_samples}")
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