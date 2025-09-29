"""
Hardware experiment utilities for PCS and PCE experiments.
Extended with comprehensive data saving capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit import transpile
import mapomatic as mm

def generate_rand_mirror_cliff(num_qubits: int, depth: int) -> QuantumCircuit:
    """Generate a random mirrored Clifford circuit."""
    from qiskit.circuit.library import HGate, SGate, CXGate
    import random
    
    qc = QuantumCircuit(num_qubits)
    
    # Random forward circuit
    for _ in range(depth):
        for q in range(num_qubits):
            gate_choice = random.choice(['h', 's', 'id'])
            if gate_choice == 'h':
                qc.h(q)
            elif gate_choice == 's':
                qc.s(q)
        
        # Add some CNOTs
        if num_qubits > 1:
            for _ in range(depth // 2):
                q1, q2 = random.sample(range(num_qubits), 2)
                qc.cx(q1, q2)
    
    # Mirror
    qc.barrier()
    for instr in reversed(qc.data[:-1]):  # Skip the barrier
        if instr.operation.name == 'h':
            qc.h(instr.qubits[0])
        elif instr.operation.name == 's':
            qc.sdg(instr.qubits[0])
        elif instr.operation.name == 'cx':
            qc.cx(instr.qubits[0], instr.qubits[1])
    
    return qc

def ghz_mirror_circ(num_qubits: int) -> QuantumCircuit:
    """Generate a GHZ state preparation followed by its mirror."""
    qc = QuantumCircuit(num_qubits)
    
    # GHZ preparation
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    
    qc.barrier()
    
    # Mirror
    for i in range(num_qubits-1, 0, -1):
        qc.cx(0, i)
    qc.h(0)
    
    return qc

def pauli_z_expectation(counts: Dict[str, int]) -> float:
    """Calculate expectation value of all-Z Pauli string."""
    total_shots = sum(counts.values())
    expectation = 0
    
    for bitstring, count in counts.items():
        value = 1
        for bit in bitstring:
            value *= 1 if bit == '0' else -1
        expectation += value * count
    
    return expectation / total_shots

def get_mapomatic_payload_scores(circuit: QuantumCircuit, backend, 
                                 num_layouts: int = 10, verbose: bool = False) -> List[Tuple[List[int], float]]:
    """Get best qubit layouts using Mapomatic."""
    trans_qc = transpile(circuit, backend, optimization_level=0)
    small_qc = mm.deflate_circuit(trans_qc)
    layouts = mm.matching_layouts(small_qc, backend)
    scores = mm.evaluate_layouts(small_qc, layouts, backend)

    
    if verbose:
        print(f"Mapomatic found {len(scores)} layouts")
        for i, (layout, score) in enumerate(scores[:5]):
            print(f"  Layout {i+1}: {layout} (score: {score:.4f})")
    
    return scores

def convert_score_to_layout(qc: QuantumCircuit, score: Tuple[List[int], float]) -> Dict[Qubit, int]:
    final_layout: Dict[Qubit, int] = {}
    layout, _ = score
    for i, phys_qubit in enumerate(layout):
        final_layout[qc.qubits[i]] = phys_qubit

    return final_layout


def find_pcs_ancillas(payload_layout: List[int], backend, num_checks: int, 
                      verbose: bool = False, require_connectivity: bool = True) -> Tuple[List[int], List[int], bool]:
    """
    Find ancilla qubits for PCS checks with connectivity verification.
    
    Returns:
        Tuple of (ancilla_positions, check_indices, all_connected)
        - ancilla_positions: Physical qubit indices for ancillas
        - check_indices: Logical indices of qubits to check
        - all_connected: Boolean indicating if all ancillas have required connectivity
    """
    coupling_map = backend.configuration().coupling_map
    adjacency = {i: set() for i in range(backend.configuration().n_qubits)}
    for edge in coupling_map:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    
    used_qubits = set(payload_layout)
    available_qubits = set(range(backend.configuration().n_qubits)) - used_qubits
    
    # Find ancillas with connections to payload qubits
    ancilla_candidates = []
    for q in available_qubits:
        connections = []
        for i, payload_q in enumerate(payload_layout):
            if payload_q in adjacency[q]:
                connections.append(i)
        if connections:
            ancilla_candidates.append((q, connections))
    
    # Sort by number of connections (more is better)
    ancilla_candidates.sort(key=lambda x: len(x[1]), reverse=True)
    
    ancilla_positions = []
    check_indices = []
    all_connected = True
    
    for i in range(min(num_checks, len(ancilla_candidates))):
        ancilla_q, connections = ancilla_candidates[i]
        ancilla_positions.append(ancilla_q)
        # Choose the first available connection
        check_idx = connections[0] if connections else None
        check_indices.append(check_idx)
        
        if check_idx is None:
            all_connected = False
    
    # If we don't have enough connected ancillas
    if len(ancilla_positions) < num_checks:
        all_connected = False
        if not require_connectivity:
            # Fill with any available qubits
            remaining = num_checks - len(ancilla_positions)
            for q in available_qubits:
                if q not in ancilla_positions:
                    ancilla_positions.append(q)
                    check_indices.append(0)  # Default check index
                    remaining -= 1
                    if remaining == 0:
                        break
    
    if verbose:
        print(f"Found {len(ancilla_positions)} ancillas for {num_checks} checks")
        print(f"  Ancilla positions: {ancilla_positions}")
        print(f"  Check indices: {check_indices}")
        print(f"  All connected: {all_connected}")
    
    return ancilla_positions, check_indices, all_connected

def create_pcs_circuit(base_circuit: QuantumCircuit, check_indices: List[int], 
                       check_type: str = 'Z', barriers: bool = True) -> QuantumCircuit:
    """Create PCS circuit with specified checks."""
    num_checks = len(check_indices)
    num_qubits = base_circuit.num_qubits
    
    pcs_circuit = QuantumCircuit(num_qubits + num_checks)
    
    # Initialize ancillas
    for i in range(num_checks):
        pcs_circuit.h(num_qubits + i)
    
    if barriers:
        pcs_circuit.barrier()
    
    # First check layer
    for i, check_idx in enumerate(check_indices):
        if check_type == 'Z':
            pcs_circuit.cz(num_qubits + i, check_idx)
        elif check_type == 'X':
            pcs_circuit.cx(num_qubits + i, check_idx)
    
    if barriers:
        pcs_circuit.barrier()
    
    # Main computation
    for instr in base_circuit.data:
        if instr.operation.name != 'barrier':
            qubits = [pcs_circuit.qubits[base_circuit.qubits.index(q)] for q in instr.qubits]
            pcs_circuit.append(instr.operation, qubits)
    
    if barriers:
        pcs_circuit.barrier()
    
    # Second check layer
    for i, check_idx in enumerate(check_indices):
        if check_type == 'Z':
            pcs_circuit.cz(num_qubits + i, check_idx)
        elif check_type == 'X':
            pcs_circuit.cx(num_qubits + i, check_idx)
    
    if barriers:
        pcs_circuit.barrier()
    
    # Measure ancillas
    for i in range(num_checks):
        pcs_circuit.h(num_qubits + i)
    
    return pcs_circuit

def convert_list_layout_to_dict(circuit: QuantumCircuit, layout_list: List[int]) -> Dict[Qubit, int]:
    """Convert a list layout to dict format for transpiler."""
    if len(layout_list) != circuit.num_qubits:
        raise ValueError(f"Layout list length {len(layout_list)} doesn't match circuit qubits {circuit.num_qubits}")
    
    layout_dict = {}
    for logical_idx, physical_idx in enumerate(layout_list):
        layout_dict[circuit.qubits[logical_idx]] = physical_idx
    
    return layout_dict

def aggregate_results(all_results: List[Dict], max_num_checks: int) -> Dict:
    """Aggregate results across multiple circuits."""
    aggregated = {}
    
    # Aggregate ideal scores
    ideal_values = [r['ideal'] for r in all_results]
    aggregated['ideal'] = {
        'mean': np.mean(ideal_values),
        'std': np.std(ideal_values),
        'values': ideal_values
    }
    
    # Aggregate baseline scores
    baseline_values = [r['baseline'] for r in all_results]
    aggregated['baseline'] = {
        'mean': np.mean(baseline_values),
        'std': np.std(baseline_values),
        'values': baseline_values
    }
    
    # Aggregate PCS scores
    for check_num in range(1, max_num_checks + 1):
        pcs_values = [r['pcs'][check_num - 1] for r in all_results if len(r['pcs']) >= check_num]
        aggregated[f'pcs_{check_num}'] = {
            'mean': np.mean(pcs_values),
            'std': np.std(pcs_values),
            'values': pcs_values
        }
    
    # Aggregate PCE scores
    for result in all_results:
        for pce_key, pce_value in result['pce'].items():
            if pce_key not in aggregated:
                aggregated[pce_key] = {'values': []}
            if not np.isnan(pce_value):
                aggregated[pce_key]['values'].append(pce_value)
    
    # Calculate mean and std for PCE
    for pce_key in list(aggregated.keys()):
        if pce_key.startswith('pce_'):
            values = aggregated[pce_key]['values']
            if values:
                aggregated[pce_key]['mean'] = np.mean(values)
                aggregated[pce_key]['std'] = np.std(values)
            else:
                aggregated[pce_key]['mean'] = np.nan
                aggregated[pce_key]['std'] = np.nan
    
    # Aggregate ZNE scores
    if all_results[0].get('zne'):
        for i in range(len(all_results[0]['zne'])):
            zne_config = all_results[0]['zne'][i]
            config_key = f"zne_{zne_config['method']}_{str(zne_config['scales']).replace(' ', '')}_{zne_config['fold']}"
            zne_values = [r['zne'][i]['score'] for r in all_results]
            aggregated[config_key] = {
                'mean': np.mean(zne_values),
                'std': np.std(zne_values),
                'values': zne_values
            }
    
    return aggregated

def save_aggregated_results(aggregated_results, num_qubits, circ_depth, num_circuits, backend_name, use_real_hardware=False):
    """Save aggregated results to CSV."""
    backend_folder = f"fake_{backend_name}" if not use_real_hardware else backend_name
    save_dir = f"results/{backend_folder}"
    os.makedirs(save_dir, exist_ok=True)
    
    data = []
    ideal_mean = aggregated_results['ideal']['mean']
    baseline_mean = aggregated_results['baseline']['mean']
    
    for method, stats in aggregated_results.items():
        data.append({
            'method': method,
            'mean': stats['mean'],
            'std': stats['std'],
            'improvement': stats['mean'] - baseline_mean if method != 'ideal' else 0,
            'distance_from_ideal': abs(stats['mean'] - ideal_mean),
            'num_samples': len(stats['values'])
        })
    
    df = pd.DataFrame(data)
    filename = f"{save_dir}/rand_cliff_{num_qubits}q_{circ_depth}d_{num_circuits}circs.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df

def append_circuit_result(circuit_result: Dict, num_qubits: int, circ_depth: int,
                         backend_name: str, use_real_hardware: bool = False,
                         max_num_checks: int = None) -> str:
    """
    Append a single circuit result to the results file.
    Creates file if it doesn't exist, appends if it does.
    """
    backend_folder = f"fake_{backend_name}" if not use_real_hardware else backend_name
    save_dir = f"results/{backend_folder}/incremental"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with timestamp for this run session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/rand_cliff_{num_qubits}q_{circ_depth}d_incremental.jsonl"
    
    # Add metadata to the circuit result
    circuit_result_with_meta = {
        'timestamp': datetime.now().isoformat(),
        'num_qubits': num_qubits,
        'circ_depth': circ_depth,
        'backend_name': backend_name,
        'use_real_hardware': use_real_hardware,
        'max_num_checks': max_num_checks,
        **circuit_result
    }
    
    # Append to JSONL file (one JSON object per line)
    with open(filename, 'a') as f:
        json.dump(circuit_result_with_meta, f)
        f.write('\n')
    
    print(f"✅ Circuit {circuit_result['circuit_idx'] + 1} results saved to {filename}")
    return filename

def load_incremental_results(filename: str) -> List[Dict]:
    """Load incrementally saved results from JSONL file."""
    results = []
    with open(filename, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def save_detailed_results(all_results: List[Dict], num_qubits: int, circ_depth: int, 
                         backend_name: str, use_real_hardware: bool = True,
                         max_num_checks: int = None) -> str:
    """
    Save detailed results including all individual data points and extrapolation values.
    
    Args:
        all_results: List of dictionaries containing results for each circuit
        num_qubits: Number of qubits in the circuits
        circ_depth: Depth of the circuits
        backend_name: Name of the backend used
        use_real_hardware: Whether real hardware was used
        max_num_checks: Maximum number of PCS checks used
        
    Returns:
        str: Path to the saved file
    """
    backend_folder = backend_name if use_real_hardware else f"fake_{backend_name}"
    save_dir = f"results/{backend_folder}/detailed"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare comprehensive data structure
    detailed_data = {
        'metadata': {
            'backend_name': backend_name,
            'num_qubits': num_qubits,
            'circ_depth': circ_depth,
            'num_circuits': len(all_results),
            'use_real_hardware': use_real_hardware,
            'max_num_checks': max_num_checks,
            'timestamp': timestamp
        },
        'circuits': []
    }
    
    # Store data for each circuit
    for circ_idx, result in enumerate(all_results):
        circuit_data = {
            'circuit_idx': circ_idx,
            'ideal_score': result['ideal'],
            'baseline_score': result['baseline'],
            'pcs_scores': result['pcs'],  # List of PCS scores for each check number
            'pcs_measured_values': result.get('pcs_measured', []),  # Individual measured values if available
            'pce_extrapolations': {},
            'zne_results': []
        }
        
        # Store PCE extrapolation details
        for pce_key, pce_value in result['pce'].items():
            # Parse the PCE method details
            parts = pce_key.split('_')
            method = parts[1]  # 'linear' or 'exponential'
            fit_points = int(parts[2].replace('pts', ''))
            
            circuit_data['pce_extrapolations'][pce_key] = {
                'method': method,
                'fit_points': fit_points,
                'data_points_used': result['pcs'][:fit_points] if fit_points <= len(result['pcs']) else result['pcs'],
                'extrapolated_value': pce_value,
                'extrapolation_target': num_qubits  # Extrapolating to max checks
            }
        
        # Store ZNE results if available
        if 'zne' in result:
            for zne_result in result['zne']:
                circuit_data['zne_results'].append({
                    'method': zne_result['method'],
                    'scale_factors': zne_result['scales'],
                    'fold_method': zne_result['fold'],
                    'extrapolated_value': zne_result['score']
                })
        
        detailed_data['circuits'].append(circuit_data)
    
    # Save as JSON for human readability
    json_filename = f"{save_dir}/detailed_results_n{num_qubits}_d{circ_depth}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(detailed_data, f, indent=2, default=str)
    print(f"Detailed results saved to: {json_filename}")
    
    # Also save as pickle for easy loading in Python
    pickle_filename = f"{save_dir}/detailed_results_n{num_qubits}_d{circ_depth}_{timestamp}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(detailed_data, f)
    print(f"Pickle file saved to: {pickle_filename}")
    
    # Create a CSV with individual circuit results for easy analysis
    csv_data = []
    for circ_idx, circuit_data in enumerate(detailed_data['circuits']):
        row = {
            'circuit_idx': circ_idx,
            'ideal': circuit_data['ideal_score'],
            'baseline': circuit_data['baseline_score']
        }
        
        # Add PCS scores
        for i, pcs_score in enumerate(circuit_data['pcs_scores']):
            row[f'pcs_{i+1}'] = pcs_score
        
        # Add PCE extrapolations
        for pce_key, pce_data in circuit_data['pce_extrapolations'].items():
            row[pce_key] = pce_data['extrapolated_value']
        
        # Add ZNE results
        for zne_result in circuit_data['zne_results']:
            zne_key = f"zne_{zne_result['method']}_{str(zne_result['scale_factors']).replace(' ', '')}"
            row[zne_key] = zne_result['extrapolated_value']
        
        csv_data.append(row)
    
    csv_df = pd.DataFrame(csv_data)
    csv_filename = f"{save_dir}/individual_circuit_results_n{num_qubits}_d{circ_depth}_{timestamp}.csv"
    csv_df.to_csv(csv_filename, index=False)
    print(f"Individual circuit CSV saved to: {csv_filename}")
    
    return json_filename

def load_detailed_results(filepath: str) -> Dict:
    """
    Load detailed results from saved file.
    
    Args:
        filepath: Path to the saved results file (JSON or pickle)
        
    Returns:
        Dictionary containing detailed results
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def plot_comparison_with_errors(df, num_qubits, circ_depth, num_circuits, backend_name=None, save_plots=True):
    """Plot comparison with error bars, with automatic saving."""
    ideal = df[df['method'] == 'ideal']['mean'].values[0]
    baseline = df[df['method'] == 'baseline']['mean'].values[0]
    
    zne_df = df[df['method'].str.startswith('zne_')]
    pce_df = df[df['method'].str.startswith('pce_')]
    
    methods = []
    means = []
    stds = []
    colors = []
    
    # Baseline
    baseline_row = df[df['method'] == 'baseline']
    methods.append('Baseline')
    means.append(baseline_row['mean'].values[0])
    stds.append(baseline_row['std'].values[0])
    colors.append('red')
    
    # ZNE methods
    for _, row in zne_df.iterrows():
        method_parts = row['method'].split('_')
        zne_type = method_parts[1]
        methods.append(f'ZNE {zne_type}')
        means.append(row['mean'])
        stds.append(row['std'])
        colors.append('orange' if zne_type == 'linear' else 'coral')
    
    # PCE methods
    for _, row in pce_df.iterrows():
        method_parts = row['method'].split('_')
        methods.append(f"PCE {method_parts[1]}\n({method_parts[2]})")
        means.append(row['mean'])
        stds.append(row['std'])
        colors.append('purple' if 'linear' in method_parts else 'brown')
    
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(methods))
    bars = plt.errorbar(x_pos, means, yerr=stds, fmt='o', markersize=8, capsize=5, capthick=2)
    
    for i, (mean, std, color) in enumerate(zip(means, stds, colors)):
        plt.bar(i, mean, yerr=std, alpha=0.5, color=color, capsize=5)
    
    plt.axhline(y=ideal, color='green', linestyle='--', label='Ideal', linewidth=2)
    plt.axhline(y=baseline, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.ylabel('Expectation Value', fontsize=12)
    plt.title(f'PCE vs ZNE: Random Clifford ({num_qubits}q, depth={circ_depth}, n={num_circuits})', fontsize=14)
    plt.xticks(x_pos, methods, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add text showing improvement percentages
    for i, (mean, method) in enumerate(zip(means, methods)):
        if 'Baseline' not in method:
            improvement = ((mean - baseline) / abs(baseline - ideal)) * 100
            plt.text(i, mean + stds[i] + 0.02, f'{improvement:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Auto-save if backend_name provided and save_plots is True
    if save_plots and backend_name:
        save_dir = f"results/{backend_name}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/comparison_n{num_qubits}_d{circ_depth}_nc{num_circuits}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {filename}")
    
    plt.show()

def plot_pcs_extrapolation(df, num_qubits, circ_depth, backend_name=None, save_plots=True):
    """Plot PCS data with extrapolations, with automatic saving."""
    ideal = df[df['method'] == 'ideal']['mean'].values[0]
    baseline = df[df['method'] == 'baseline']['mean'].values[0]
    
    pcs_df = df[df['method'].str.startswith('pcs_')]
    pce_df = df[df['method'].str.startswith('pce_')]
    
    plt.figure(figsize=(10, 8))
    
    plt.axhline(y=ideal, color='green', linestyle='--', label='Ideal', linewidth=2)
    plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline', linewidth=2)
    
    # Plot PCS data points
    if not pcs_df.empty:
        pcs_nums = [int(m.split('_')[1]) for m in pcs_df['method']]
        pcs_means = pcs_df['mean'].values
        pcs_stds = pcs_df['std'].values
        
        plt.errorbar(pcs_nums, pcs_means, yerr=pcs_stds, 
                    fmt='bo', markersize=8, capsize=5, 
                    label='PCS data', zorder=3)
        
        # Add shaded confidence region
        plt.fill_between(pcs_nums, 
                        pcs_means - pcs_stds, 
                        pcs_means + pcs_stds, 
                        alpha=0.2, color='blue')
    
    # Plot PCE extrapolations
    colors = {'linear': 'purple', 'exponential': 'brown'}
    markers = {'2pts': 's', '3pts': '^'}
    
    for _, row in pce_df.iterrows():
        method_parts = row['method'].split('_')
        if len(method_parts) >= 3:
            pce_type = method_parts[1]
            pts = method_parts[2]
            
            plt.errorbar(num_qubits, row['mean'], yerr=row['std'],
                        fmt=markers.get(pts, 'o'), 
                        color=colors.get(pce_type, 'gray'),
                        markersize=10, capsize=5,
                        label=f'PCE {pce_type} ({pts})', zorder=4)
            
            # Draw extrapolation line
            if not pcs_df.empty and pts == '2pts' and len(pcs_nums) >= 2:
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(pcs_nums[:2], pcs_means[:2])
                x_line = np.array([1, num_qubits])
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, '--', 
                        color=colors.get(pce_type, 'gray'), 
                        alpha=0.5, linewidth=1)
    
    plt.xlabel('Number of Checks', fontsize=12)
    plt.ylabel('Expectation Value', fontsize=12)
    plt.title(f'PCS Data with PCE Extrapolations ({num_qubits}q, depth={circ_depth})', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, num_qubits + 0.5)
    
    # Adjust y-limits to show all data with margin
    y_data = list(pcs_means) + [baseline, ideal] + list(pce_df['mean'].values)
    y_min, y_max = min(y_data), max(y_data)
    y_margin = (y_max - y_min) * 0.15
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # Auto-save if backend_name provided and save_plots is True
    if save_plots and backend_name:
        save_dir = f"results/{backend_name}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/pcs_extrapolation_n{num_qubits}_d{circ_depth}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"PCS extrapolation plot saved to: {filename}")
    
    plt.show()