"""
High-level experiment runners for PCE vs ZNE experiments.
Each function runs a specific type of experiment or analysis.
"""

from typing import Dict, List, Tuple, Any
from .quantum_execution import execute_circuit, apply_pauli_basis, compute_z_expectation, calibrate_mthree
from .pcs_filtering import filter_counts, compute_post_selection_rate
from .error_mitigation import extrapolate_checks
from qiskit_aer import AerSimulator

from IPython.display import display

def run_ideal_experiment(circuit, pauli_string: str, shots: int):
    ideal_circuit = apply_pauli_basis(circuit, pauli_string)
    ideal_sim = AerSimulator()
    ideal_circuit.measure_all()
    ideal_job = ideal_sim.run(ideal_circuit, shots=shots)
    ideal_counts = ideal_job.result().get_counts()
    ideal_expectation = compute_z_expectation(ideal_counts)

    return ideal_expectation

def run_baseline_experiment(
    circuit,
    backend,
    pauli_string: str,
    shots: int,
    layout=None,
    twirling: bool = False,
    mthree=None
) -> Tuple[float, Dict[str, Any]]:
    """
    Run baseline (no error mitigation) experiment.
    
    Returns:
        (expectation_value, metadata)
    """
    # Apply measurement basis and add measurements
    rotated_circuit = apply_pauli_basis(circuit, pauli_string)
    rotated_circuit.measure_all()
    
    # Execute circuit
    counts = execute_circuit(
        rotated_circuit, backend, shots,
        layout=layout, twirling=twirling, mthree=mthree
    )
    
    # Compute expectation value
    expectation = compute_z_expectation(counts)
    
    metadata = {
        'method': 'baseline',
        'total_shots': sum(counts.values()),
        'counts': counts
    }
    
    return expectation, metadata


def run_pcs_experiment(
    pcs_circuit,
    backend,
    pauli_string: str,
    shots: int,
    num_checks: int,
    signs: List[str],
    layout=None,
    twirling: bool = False,
    mthree=None
) -> Tuple[float, Dict[str, Any]]:
    """
    Run PCS experiment with post-selection.
    
    Returns:
        (expectation_value, metadata)
    """
    # Apply measurement basis and add measurements
    rotated_circuit = apply_pauli_basis(pcs_circuit, pauli_string)
    rotated_circuit.measure_all()
    
    # Execute circuit
    raw_counts = execute_circuit(
        rotated_circuit, backend, shots,
        layout=layout, twirling=twirling, mthree=mthree
    )
    
    # Apply PCS filtering
    filtered_counts = filter_counts(num_checks, signs, raw_counts)
    post_selection_rate = compute_post_selection_rate(raw_counts, filtered_counts)
    
    # Compute expectation value
    expectation = compute_z_expectation(filtered_counts)
    
    metadata = {
        'method': 'pcs',
        'num_checks': num_checks,
        'signs': signs,
        'total_raw_shots': sum(raw_counts.values()),
        'total_filtered_shots': sum(filtered_counts.values()),
        'post_selection_rate': post_selection_rate,
        'raw_counts': raw_counts,
        'filtered_counts': filtered_counts
    }
    
    return expectation, metadata



def collect_zne_data(
    circuit,
    backend,
    pauli_string: str,
    shots: int,
    scale_factors: List[float],
    fold_method,
    layout=None,
    twirling: bool = False,
    mthree=None
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Collect ZNE expectation values for different scale factors without extrapolation.
    
    Returns:
        (list_of_expectation_values, metadata)
    """
    from mitiq import zne
    
    expectation_values = []
    
    print(f"Collecting ZNE data for scale factors: {scale_factors}")
    
    for scale_factor in scale_factors:
        print(f"  Scale factor {scale_factor}...")
        
        # Scale the circuit using mitiq
        if scale_factor == 1.0:
            scaled_circuit = circuit.copy()
        else:
            scaled_circuit = fold_method(circuit, scale_factor)
        
        # Apply measurement basis and execute
        rotated_circuit = apply_pauli_basis(scaled_circuit, pauli_string)
        rotated_circuit.measure_all()
        
        counts = execute_circuit(
            rotated_circuit, backend, shots,
            layout=layout, twirling=twirling, mthree=mthree
        )
        
        expectation = compute_z_expectation(counts)
        expectation_values.append(expectation)
        
        print(f"    Expectation: {expectation:.6f}")
    
    metadata = {
        'method': 'zne_data_collection',
        'scale_factors': scale_factors,
        'shots_per_scale': shots,
        'expectation_values': expectation_values
    }
    
    return expectation_values, metadata


def extrapolate_zne_data(
    expectation_values: List[float],
    scale_factors: List[float],
    method: str = 'linear',
    show_plot: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Apply ZNE extrapolation to collected expectation value data.
    High-level wrapper that adds metadata around core extrapolation.
    
    Returns:
        (extrapolated_expectation, metadata)
    """
    from .error_mitigation import extrapolate_zne
    
    # Use core extrapolation function with plotting
    extrapolated_value = extrapolate_zne(expectation_values, scale_factors, method, show_plot)
    
    metadata = {
        'method': 'zne',
        'zne_method': method,
        'scale_factors': scale_factors,
        'input_values': expectation_values,
        'extrapolated_value': extrapolated_value
    }
    
    return extrapolated_value, metadata


def extrapolate_pcs_data(
    pcs_expectation_values: List[float],
    num_checks_to_fit: int,
    extrap_target: int,
    method: str = 'linear',
    show_plot: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Apply PCE extrapolation to existing PCS data.
    
    Args:
        pcs_expectation_values: List of PCS expectation values
        num_checks_to_fit: Number of data points to use for fitting
        extrap_target: Target number of checks to extrapolate to
        method: Extrapolation method ('linear' or 'exponential')
        show_plot: Whether to show the fit plot
    
    Returns:
        (extrapolated_value, metadata)
    """
    extrap_values, fit_func = extrapolate_checks(
        num_checks_to_fit=num_checks_to_fit,
        extrap_checks=[extrap_target],
        expectation_values=pcs_expectation_values,
        method=method,
        show_plot=show_plot
    )
    
    expectation = extrap_values[0] # Sends back first element since we are only passing one 'extrap_target'
    
    metadata = {
        'method': 'pce',
        'pce_method': method,
        'num_fit_points': num_checks_to_fit,
        'input_values': pcs_expectation_values[:num_checks_to_fit],
        'extrap_target': extrap_target
    }
    
    return expectation, metadata