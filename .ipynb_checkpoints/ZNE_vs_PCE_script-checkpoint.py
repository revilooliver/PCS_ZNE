from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator, QasmSimulator
import qiskit_aer.noise as noise
from functools import partial
from utils.pce_vs_zne_utils import *
from qiskit_ibm_runtime.fake_provider import *
from itertools import combinations
import os
import csv
import matplotlib.pyplot as plt
from typing import List

def setup_noise_model():
    """Set up the custom noise model for simulation."""
    prob_1 = 0.002  # 1-qubit gate
    prob_2 = 0.02  # 2-qubit gate
    
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)
    
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'], ['cz'])
    return noise_model


def setup_backend(use_real_hardware=False, use_fake_backend=False):
    """Set up the quantum backend.

    Args:
        use_real_hardware (bool): Whether to use real IBM quantum hardware
        use_fake_backend (bool): Whether to use a fake backend instead of custom noise model
    """
    if QiskitRuntimeService.saved_accounts() and use_real_hardware:
        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)
        noise_model = None
    elif use_fake_backend:
        fake_backend = FakeCairoV2()
        noise_model = noise.NoiseModel.from_backend(fake_backend)
        backend = AerSimulator(noise_model=noise_model)
    else:
        noise_model = setup_noise_model()
        backend = AerSimulator(noise_model=noise_model)
    return backend

def save_avg_errors(circ_folder, filename, zne_avg_error, pce_avg_error):
    """Save the average errors to a CSV file."""
    dir_path = os.path.join("data_PCE_vs_ZNE", circ_folder)
    os.makedirs(dir_path, exist_ok=True)
    
    filepath = os.path.join(dir_path, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Average Absolute Error"])
        writer.writerow(["ZNE", zne_avg_error])
        writer.writerow(["PCE", pce_avg_error])

def run_zne_experiment(cliff_circs, backend, pauli_string, ideal_expectations):
    """Run the ZNE experiment for all circuits."""
    zne_abs_errors = []
    for i, circ in enumerate(cliff_circs):
        zne_executor = partial(ibmq_executor, backend=backend, pauli_string=pauli_string)
        zne_exp = zne.execute_with_zne(circ, zne_executor)

        print(f"ZNE mitigated exp for {pauli_string}: {zne_exp}")
        print(f"Ideal expectation = {ideal_expectations[i]}")

        abs_error = np.abs(ideal_expectations[i] - zne_exp)
        zne_abs_errors.append(abs_error)

    return np.mean(zne_abs_errors)

# def run_zne_experiment(cliff_circs, backend, pauli_string, ideal_expectations,
#                        scale_factors=None, factory_type='linear'):
#     """Run the ZNE experiment for all circuits with customizable ZNE settings.
#
#     Args:
#         cliff_circs: List of clifford circuits
#         backend: Quantum backend to use
#         pauli_string: Pauli string to measure
#         ideal_expectations: List of ideal expectation values
#         scale_factors: List of scale factors for noise extrapolation (default: [1.0, 2.0, 3.0])
#         factory_type: Type of extrapolation factory ('linear' or 'richardson')
#     """
#     zne_abs_errors = []
#
#     # Set up default scale factors if none provided
#     if scale_factors is None:
#         scale_factors = [1.0, 2.0, 3.0]
#
#     # Create the appropriate factory
#     if factory_type.lower() == 'linear':
#         factory = zne.inference.LinearFactory(scale_factors=scale_factors)
#     elif factory_type.lower() == 'richardson':
#         factory = zne.inference.RichardsonFactory(scale_factors=scale_factors)
#     else:
#         raise ValueError(f"Unsupported factory type: {factory_type}")
#
#     for i, circ in enumerate(cliff_circs):
#         zne_executor = partial(ibmq_executor, backend=backend, pauli_string=pauli_string)
#         zne_exp = zne.execute_with_zne(circ, zne_executor, factory=factory)
#
#         print(f"ZNE mitigated exp for {pauli_string}: {zne_exp}")
#         print(f"Ideal expectation = {ideal_expectations[i]}")
#
#         abs_error = np.abs(ideal_expectations[i] - zne_exp)
#         zne_abs_errors.append(abs_error)
#
#     return np.mean(zne_abs_errors)

def plot_extrapolation(check_points: List[int], 
                      expectation_values: List[float],
                      extrapolated_values: List[float],
                      polynomial: np.poly1d,
                      ideal_value: float,
                      circuit_idx: int) -> None:
    """
    Plot the extrapolation process for PCE in real-time.
    
    Args:
        check_points: Points where checks were performed
        expectation_values: Measured expectation values
        extrapolated_values: Extrapolated values
        polynomial: Fitted polynomial
        ideal_value: The ideal expectation value
        circuit_idx: Index of the current circuit
    """
    plt.clf()  # Clear the current figure
    
    # Plot measured points
    plt.scatter(check_points[:len(expectation_values)], expectation_values, 
               color='blue', label='Measured values', zorder=3)
    
    # Plot extrapolated points
    plt.scatter(check_points[len(expectation_values):], extrapolated_values,
               color='red', label='Extrapolated values', zorder=3)
    
    # Plot ideal value
    plt.axhline(y=ideal_value, color='g', linestyle='--', label='Ideal value')
    
    # Plot polynomial fit
    x_continuous = np.linspace(min(check_points), max(check_points), 100)
    plt.plot(x_continuous, polynomial(x_continuous), 'k-', alpha=0.5, label='Polynomial fit')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of checks')
    plt.ylabel('Expectation value')
    plt.title(f'PCE Extrapolation for Circuit {circuit_idx + 1}')
    plt.legend()
    
    plt.draw()
    plt.pause(0.1)  # Add small pause to allow plot to update

def run_pce_experiment(cliff_circs, pcs_circs, signs_list, backend, pauli_string, num_qubits, 
                      num_checks, num_checks_to_fit, extrap_checks, ideal_expectations):
    """Run the PCE experiment for all circuits."""
    pce_abs_errors = []
    check_points = list(range(1, num_checks + 1)) + list(extrap_checks)
    
    # Create a persistent figure window
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    
    for i, cliff_circ in enumerate(cliff_circs):
        print(f"i = {i+1} out of {len(cliff_circs)}")
        
        expectation_values = []
        for j in range(num_checks):
            pcs_circ = pcs_circs[i][j]
            signs = signs_list[i][j]
            expectation_value = ibmq_executor_pcs(pcs_circ, backend=backend, 
                                                pauli_string=pauli_string, 
                                                num_qubits=num_qubits, 
                                                signs=signs)
            expectation_values.append(expectation_value)
        
        print("Expected values from implemented checks:", expectation_values)
        
        extrapolated_values, polynomial = extrapolate_checks(num_checks_to_fit, 
                                                           extrap_checks, 
                                                           expectation_values)
        
        # Plot the extrapolation
        plot_extrapolation(check_points, 
                         expectation_values,
                         extrapolated_values,
                         polynomial,
                         ideal_expectations[i],
                         i)
        
        pce_exp = extrapolated_values[-1]
        
        print(f"PCE exp for {pauli_string}: {pce_exp}")
        print(f"Ideal expectation = {ideal_expectations[i]}")
        abs_error = np.abs(ideal_expectations[i] - pce_exp)
        print(f"Abs error = {abs_error}")
        pce_abs_errors.append(abs_error)
    
    plt.ioff()  # Turn off interactive mode
    plt.close()  # Close the figure window when done
    
    return np.mean(pce_abs_errors)

def main():
    # Parameters
    USE_REAL_HARDWARE = False
    USE_FAKE_BACKEND = True
    num_qubits = 6
    num_circs = 20
    pauli_string = 'Z' * num_qubits
    num_checks = num_qubits // 2
    num_checks_to_fit = num_checks
    extrap_checks = range(num_checks_to_fit + 1, num_qubits + 1)
    only_Z_checks = True
    
    # Setup
    np.set_printoptions(precision=6, edgeitems=10, linewidth=150, suppress=True)
    backend = setup_backend(USE_REAL_HARDWARE, USE_FAKE_BACKEND)

    # Generate circuits
    cliff_circs = random_cliff_circs(num_qubits, num_circs)
    
    # Calculate ideal expectations
    operator = Pauli(pauli_string)
    ideal_expectations = []
    for circ in cliff_circs:
        psi = Statevector(circ)
        expect = np.array(psi).T.conj() @ operator.to_matrix() @ np.array(psi)
        ideal_expectations.append(expect)
    
    # Run ZNE experiment with custom settings
    zne_avg_error = run_zne_experiment(
        cliff_circs, 
        backend, 
        pauli_string, 
        ideal_expectations
    )
    print(f"Average absolute error for ZNE: {zne_avg_error:.5f}")
    
    # Prepare PCE circuits
    pcs_circs = [[] for _ in range(num_circs)]
    signs_list = [[] for _ in range(num_circs)]
    
    for i, circ in enumerate(cliff_circs):
        print(f"i = {i+1} out of {num_circs}")
        for check_id in range(1, num_checks + 1):
            print(f"check_id = {check_id}")
            sign, pcs_circ = convert_to_PCS_circ(circ, num_qubits, check_id, 
                                                only_Z_checks=only_Z_checks)
            pcs_circs[i].append(pcs_circ)
            signs_list[i].append(sign)
    
    # Run PCE experiment
    pce_avg_error = run_pce_experiment(cliff_circs, pcs_circs, signs_list, backend, 
                                     pauli_string, num_qubits, num_checks, 
                                     num_checks_to_fit, extrap_checks, ideal_expectations)
    print(f"Average absolute error for PCE: {pce_avg_error:.5f}")
    print(f"Average absolute error for ZNE: {zne_avg_error:.5f}")

    
    # Save results
    circ_folder = "rand_cliffs"
    file_name = f"avg_errors_n={num_qubits}_num_circs={num_circs}_num_samp={10_000}.csv"
    save_avg_errors(circ_folder, file_name, zne_avg_error, pce_avg_error)

if __name__ == "__main__":
    main()