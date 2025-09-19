"""
Simple QAOA parameter optimization for MaxCut problems.
"""

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from typing import Dict, List
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def get_maxcut_pauli_string(qubits, num_qubits):
    # Create Pauli string for this edge
    pauli_string = ['I'] * num_qubits
    pauli_string[qubits[0]] = 'Z'
    pauli_string[qubits[1]] = 'Z'
    pauli_str = ''.join(pauli_string)
    return pauli_str


def compute_maxcut_expectation(graph: nx.Graph, gamma: List[float], beta: List[float]) -> float:
    """Compute expected MaxCut value for given QAOA parameters."""
    from utils.pce_vs_zne_utils_v2 import build_pcs_qaoa_ansatz, build_max_cut_paulis, get_ideal_expectation
    
    n = graph.number_of_nodes()
    p = len(gamma)
    
    # Create QAOA circuit (build_pcs_qaoa_ansatz already includes initial state preparation)
    qc = build_pcs_qaoa_ansatz(
        graph=graph,
        p=p,
        gamma_vals=gamma,
        beta_vals=beta,
        barriers=False
    )
    
    # Get MaxCut Pauli strings
    pauli_list = build_max_cut_paulis(graph)
    
    # Compute total expectation value for MaxCut Hamiltonian
    total_expectation = 0.0
    for pauli_op, qubits, weight in pauli_list:

        pauli_str = get_maxcut_pauli_string(qubits, num_qubits=n)
        
        # Get expectation value for ZZ term
        zz_expectation = get_ideal_expectation(qc, pauli_str).real
        
        # MaxCut Hamiltonian: sum over edges of (1 - ZiZj)/2
        total_expectation += weight * 0.5 * (1 - zz_expectation)
    
    return total_expectation


def optimize_qaoa_maxcut(graph: nx.Graph, p: int, num_attempts: int = 10, early_stop_tolerance: float = 1e-6) -> Dict:
    """
    Optimize QAOA parameters for MaxCut problem with multiple random starts.
    
    Args:
        graph: NetworkX graph for MaxCut problem
        p: QAOA depth (number of layers)
        num_attempts: Number of optimization attempts with different initial parameters
        early_stop_tolerance: Stop optimization when within this tolerance of true MaxCut
        
    Returns:
        Dictionary with 'gamma_optimal', 'beta_optimal', 'optimal_value', and optimization details
    """
    # Calculate true MaxCut value for early stopping
    true_maxcut = get_true_maxcut_by_diagonalization(graph)
    print(f"True MaxCut value: {true_maxcut:.6f}")
    print(f"Will stop early if QAOA reaches within {early_stop_tolerance:.6f} of true MaxCut")
    
    # Objective function (minimize negative for maximization)
    def objective(params):
        gamma = params[:p]
        beta = params[p:]
        return -compute_maxcut_expectation(graph, gamma.tolist(), beta.tolist())
    
    # Parameter bounds
    bounds = [(0, np.pi)] * p + [(0, np.pi/2)] * p
    
    best_result = None
    best_value = -np.inf
    all_results = []
    
    print(f"Running up to {num_attempts} optimization attempts...")
    
    for attempt in range(num_attempts):
        # Random initial parameters for this attempt
        gamma_init = np.random.uniform(0, np.pi, p)
        beta_init = np.random.uniform(0, np.pi/2, p)
        x0 = np.concatenate([gamma_init, beta_init])
        
        # Optimize
        try:
            result = minimize(objective, x0, method='COBYLA', bounds=bounds)
            
            gamma_opt = result.x[:p].tolist()
            beta_opt = result.x[p:].tolist()
            value = -result.fun
            
            all_results.append({
                'attempt': attempt + 1,
                'gamma': gamma_opt,
                'beta': beta_opt,
                'value': value,
                'success': result.success
            })
            
            print(f"  Attempt {attempt + 1}: Value = {value:.6f}, Success = {result.success}")
            
            # Update best result if this is better
            if value > best_value:
                best_value = value
                best_result = {
                    'gamma_optimal': gamma_opt,
                    'beta_optimal': beta_opt,
                    'optimal_value': value,
                    'best_attempt': attempt + 1
                }
                
                # Check for early stopping
                if abs(value - true_maxcut) <= early_stop_tolerance:
                    print(f"  🎯 EARLY STOP: Reached true MaxCut! ({value:.6f} ≈ {true_maxcut:.6f})")
                    print(f"  Stopping after {attempt + 1} attempts")
                    break
                
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Failed with error {e}")
            all_results.append({
                'attempt': attempt + 1,
                'gamma': None,
                'beta': None,
                'value': -np.inf,
                'success': False,
                'error': str(e)
            })
    
    if best_result is None:
        raise RuntimeError("All optimization attempts failed!")
    
    # Add summary statistics
    successful_values = [r['value'] for r in all_results if r['success']]
    early_stopped = abs(best_value - true_maxcut) <= early_stop_tolerance if best_result else False
    
    best_result['optimization_summary'] = {
        'num_attempts': len(all_results),
        'num_requested': num_attempts,
        'num_successful': len(successful_values),
        'best_value': best_value,
        'true_maxcut': true_maxcut,
        'gap_to_optimal': abs(best_value - true_maxcut) if best_result else None,
        'early_stopped': early_stopped,
        'mean_value': np.mean(successful_values) if successful_values else None,
        'std_value': np.std(successful_values) if len(successful_values) > 1 else None,
        'all_results': all_results
    }
    
    print(f"\nOptimization complete!")
    print(f"  Best value: {best_value:.6f} (attempt {best_result['best_attempt']})")
    print(f"  True MaxCut: {true_maxcut:.6f}")
    print(f"  Gap to optimal: {abs(best_value - true_maxcut):.6f}")
    print(f"  Successful attempts: {len(successful_values)}/{len(all_results)}")
    if early_stopped:
        print(f"  ✅ Early stopped: Found optimal solution!")
    if len(successful_values) > 1:
        print(f"  Mean ± std: {np.mean(successful_values):.6f} ± {np.std(successful_values):.6f}")
    
    return best_result


# Calculate true MaxCut value by diagonalizing the MaxCut Hamiltonian
def get_true_maxcut_by_diagonalization(G):
    """Calculate the true MaxCut value by diagonalizing the MaxCut Hamiltonian."""
    from qiskit.quantum_info import SparsePauliOp
    
    # Build MaxCut Hamiltonian: sum over edges of (1 - ZiZj)/2
    pauli_strings = []
    coefficients = []
    
    # Add identity term: num_edges/2
    num_edges = G.number_of_edges()
    num_qubits = G.number_of_nodes()
    pauli_strings.append('I' * num_qubits)
    coefficients.append(num_edges / 2.0)
    
    # Add ZZ terms: -1/2 * ZiZj for each edge
    for edge in G.edges():
        i, j = edge
        pauli_string = ['I'] * num_qubits
        pauli_string[i] = 'Z'
        pauli_string[j] = 'Z'
        zz_string = ''.join(pauli_string)
        
        pauli_strings.append(zz_string)
        coefficients.append(-0.5)
    
    # Create Hamiltonian operator
    hamiltonian = SparsePauliOp(pauli_strings, coefficients)
    
    # Get matrix representation and diagonalize
    hamiltonian_matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvals(hamiltonian_matrix)
    
    # Maximum eigenvalue gives the true maximum cut
    true_maxcut_value = np.max(eigenvalues.real)
    
    print(f"MaxCut Hamiltonian diagonalization:")
    print(f"  Number of edges: {num_edges}")
    print(f"  True MaxCut value: {true_maxcut_value:.6f}")
    print(f"  Eigenvalue range: [{np.min(eigenvalues.real):.3f}, {np.max(eigenvalues.real):.3f}]")
    
    return true_maxcut_value


def find_digital_parameters(graph: nx.Graph, p_max: int = 3, gamma_steps: int = 8, beta_steps: int = 8) -> Dict:
    """
    Find QAOA parameters that produce computational basis states (digital outputs).
    
    Args:
        graph: NetworkX graph for MaxCut problem
        p_max: Maximum QAOA depth to try
        gamma_steps: Number of gamma values to test
        beta_steps: Number of beta values to test
        
    Returns:
        Dictionary with optimal parameters and digitality score
    """
    from utils.pce_vs_zne_utils_v2 import build_pcs_qaoa_ansatz
    
    print(f"Searching for digital QAOA parameters...")
    print(f"Parameter space: p ∈ [1,{p_max}], γ ∈ [0,π], β ∈ [0,π/2]")
    
    best_params = None
    best_digitality = 0
    best_maxcut_value = 0
    all_results = []
    
    # Test different QAOA depths
    for p in range(1, p_max + 1):
        print(f"\n--- Testing QAOA depth p = {p} ---")
        
        # Parameter ranges
        gamma_range = np.linspace(0, np.pi, gamma_steps)
        beta_range = np.linspace(0, np.pi/2, beta_steps)
        
        for i, gamma in enumerate(gamma_range):
            for j, beta in enumerate(beta_range):
                try:
                    gamma_vals = [gamma] * p
                    beta_vals = [beta] * p
                    
                    # Build QAOA circuit
                    qc = build_pcs_qaoa_ansatz(
                        graph, p, 
                        gamma_vals=gamma_vals, 
                        beta_vals=beta_vals, 
                        barriers=False
                    )
                    
                    # Get statevector
                    psi = Statevector(qc)
                    amplitudes = np.abs(psi.data) ** 2
                    
                    # Calculate digitality metrics
                    max_prob = np.max(amplitudes)  # Highest probability
                    entropy = -np.sum(amplitudes * np.log2(amplitudes + 1e-12))  # Shannon entropy
                    participation_ratio = 1 / np.sum(amplitudes ** 2)  # Effective number of states
                    
                    # Digitality score: prefer high max probability, low entropy
                    digitality = max_prob * (1 / (1 + entropy))
                    
                    # Also compute MaxCut value for this state
                    from utils.pce_vs_zne_utils_v2 import build_max_cut_paulis, get_ideal_expectation
                    maxcut_observable = build_max_cut_paulis(graph)
                    maxcut_value = get_ideal_expectation(qc, maxcut_observable).real
                    
                    # Store result
                    result = {
                        'p': p,
                        'gamma': gamma_vals,
                        'beta': beta_vals,
                        'digitality': digitality,
                        'max_prob': max_prob,
                        'entropy': entropy,
                        'participation_ratio': participation_ratio,
                        'maxcut_value': maxcut_value
                    }
                    all_results.append(result)
                    
                    # Update best if this is more digital
                    if digitality > best_digitality:
                        best_digitality = digitality
                        best_maxcut_value = maxcut_value
                        best_params = result.copy()
                        print(f"  New best: γ={gamma:.3f}, β={beta:.3f} → digitality={digitality:.4f}, "
                              f"max_prob={max_prob:.3f}, MaxCut={maxcut_value:.3f}")
                
                except Exception as e:
                    print(f"  Error at γ={gamma:.3f}, β={beta:.3f}: {e}")
                    continue
    
    if best_params is None:
        raise RuntimeError("No valid parameters found!")
    
    # Format result similar to optimize_qaoa_maxcut
    final_result = {
        'gamma_optimal': best_params['gamma'],
        'beta_optimal': best_params['beta'],
        'optimal_value': best_params['maxcut_value'],
        'digitality_score': best_params['digitality'],
        'max_probability': best_params['max_prob'],
        'entropy': best_params['entropy'],
        'participation_ratio': best_params['participation_ratio'],
        'search_summary': {
            'total_evaluations': len(all_results),
            'best_p': best_params['p'],
            'parameter_ranges': {
                'gamma': (0, np.pi, gamma_steps),
                'beta': (0, np.pi/2, beta_steps)
            }
        }
    }
    
    print(f"\n=== Digital Parameter Search Complete! ===")
    print(f"Best digitality score: {best_digitality:.4f}")
    print(f"Max probability: {best_params['max_prob']:.3f}")
    print(f"Shannon entropy: {best_params['entropy']:.3f}")
    print(f"MaxCut value: {best_params['maxcut_value']:.3f}")
    print(f"Optimal p: {best_params['p']}")
    print(f"Optimal γ: {best_params['gamma']}")
    print(f"Optimal β: {best_params['beta']}")
    
    return final_result