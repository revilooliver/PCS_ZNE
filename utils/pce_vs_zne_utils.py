"""
Utils for running experiments comparing pce and zne.
"""

import numpy as np
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


# def ibmq_executor(circuit: QuantumCircuit, shots: int = 10_000):
def ibmq_executor(circuit: QuantumCircuit, backend, pauli_string: str, shots: int = 10_000):
    """Executor for ZNE. 
    """
    # Modify the circuit to measure the required Pauli observables
    measurement_circuit = circuit.copy()
    measurement_circuit.barrier()
    apply_measurement_basis(measurement_circuit, pauli_string)
    measurement_circuit.measure_all()
    # print(measurement_circuit)

    # Transpile for the backend
    exec_circuit = transpile(
        measurement_circuit,
        backend=backend,
        optimization_level=0 # Preserve gate structure for simulation accuracy.
    )

    # print("transpiled circuit")
    # print(exec_circuit)

    # Run the circuit
    job = backend.run(exec_circuit, shots=shots)
    counts = job.result().get_counts()

    # Compute the expectation value based on counts
    # expectation_value = sum((-1 if (bin(int(state, 16)).count('1') % 2) else 1) * count for state, count in counts.items()) / shots
    expectation_value = compute_expectation_value(counts, pauli_string)
    return expectation_value

def mitigate_zne(circ, backend, pauli_string, method="richardson", scale_factors=[1,2,3], fold_method=zne.scaling.fold_global):
    """
    Runs ibmq_executor and mitigates the expectation for 'pauli_string' observable using zne. Method set to default for now.
    """
    zne_executor = partial(ibmq_executor, backend=backend, pauli_string=pauli_string)

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

def ibmq_executor_pcs(circuit: QuantumCircuit, backend, pauli_string: str, num_qubits, shots: int = 10_000, signs = None):
    """Executor for PCS.
    """
    # Modify the circuit to measure the required Pauli observables
    measurement_circuit = circuit.copy()
    apply_measurement_basis(measurement_circuit, pauli_string)
    measurement_circuit.measure_all()
    # print(measurement_circuit)

    # Transpile for the backend
    exec_circuit = transpile(
        measurement_circuit,
        backend=backend,
        optimization_level=0 # keep at level 0 for mirror circuits, or it will cancel out all the gates.
    )

    # print("transpiled circuit")
    # print(exec_circuit)

    # Run the circuit
    job = backend.run(exec_circuit, shots=shots)
    # print(job.result().quasi_dists)

    
    counts = job.result().get_counts()
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


def random_cliff_circs(num_qubits, num_circs):
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

def extrapolate_checks(num_checks_to_fit: int, extrap_checks, expectation_values):
    """
    Fit a linear model to the first `num_checks_to_fit` expectation values, 
    and extrapolate to multiple check numbers.

    Parameters:
    - num_checks_to_fit: int, number of initial data points to use for fitting
    - extrap_checks: iterable of int, check numbers to extrapolate to
    - expectation_values: list or array-like, observed values

    Returns:
    - list of extrapolated values, one for each value in `extrap_checks`
    """
    check_numbers = range(1, num_checks_to_fit + 1)

    # Fit a degree-1 polynomial (linear regression)
    polynomial_coefficients = np.polyfit(check_numbers, expectation_values[:num_checks_to_fit], 1)
    polynomial = np.poly1d(polynomial_coefficients)

    # Extrapolate to all specified check numbers
    extrap_values = [polynomial(c) for c in extrap_checks]

    return extrap_values, polynomial

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

def load_avg_errors(circ_folder, num_circs, num_samples):
    """
    Load average error results from CSV files for multiple methods.

    The function searches for files named with the pattern:
       avg_errors_n={num_qubits}_num_circs={num_circs}_num_samp={num_samples}.csv

    Parameters:
      circ_folder (str): The folder containing the saved CSV files.
      num_circs (int): The number of circuits used (embedded in the filename).
      num_samples (int): The number of samples used (embedded in the filename).

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
            # Only consider files that include the num_circs and num_samples information in their names.
            if f"num_circs={num_circs}" in file and f"num_samp={num_samples}" in file:
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


def plot_avg_errors_by_qubit(data, num_samples=None, num_circs=None, save_path=None):
    """
    Plot average errors for multiple methods grouped by number of qubits.

    Parameters:
      data (dict): {num_qubits: {method_name: error, ...}, ...}
      num_samples (int, optional): Number of samples, used in the plot title.
      num_circs (int, optional): Number of circuits, used in the plot title.
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