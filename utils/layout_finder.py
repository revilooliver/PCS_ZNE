"""
Clean layout finding utilities using mapomatic.
Optimized for clarity and efficiency with single-responsibility functions.
"""

from typing import List, Dict, Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_error_map
import mapomatic as mm
from .circuit_generation import build_pcs_circuit


def find_best_layouts(circuit: QuantumCircuit, backend, max_candidates: int = 100) -> List[List[int]]:
    """Find ranked list of best payload layouts using mapomatic."""
    trans_qc = transpile(circuit, backend, optimization_level=0)
    small_qc = mm.deflate_circuit(trans_qc)
    
    layouts = mm.matching_layouts(small_qc, backend)
    scores = mm.evaluate_layouts(small_qc, layouts, backend)
    
    return [layout for layout, score in scores[:max_candidates]]


def build_connectivity_graph(backend) -> Dict[int, set]:
    """Build adjacency graph from backend coupling map."""
    adjacency = {i: set() for i in range(backend.configuration().n_qubits)}
    for edge in backend.configuration().coupling_map:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return adjacency


def find_available_ancillas(payload_layout: List[int], connectivity: Dict[int, set], num_checks: int, verbose: bool = False) -> Optional[List[Tuple[int, int]]]:
    """
    Find unique ancilla assignments for PCS checks.
    
    Returns:
        List of (payload_idx, ancilla_physical) tuples, or None if impossible
    """
    used_qubits = set(payload_layout)
    assignments = []
    
    # Try to find assignments for each payload qubit until we have enough
    for payload_idx, payload_qubit in enumerate(payload_layout):
        if verbose:
            print(f"\nTrying payload qubit {payload_idx} (physical {payload_qubit})")
            print(f"  Connected to: {connectivity[payload_qubit]}")
            print(f"  Used qubits: {used_qubits}")
        
        if len(assignments) >= num_checks:
            if verbose:
                print(f"  Already have {num_checks} assignments, stopping")
            break
            
        # Find unused neighbor
        available_neighbors = connectivity[payload_qubit] - used_qubits
        if verbose:
            print(f"  Available neighbors: {available_neighbors}")
        
        if available_neighbors:
            ancilla = min(available_neighbors)  # Pick lowest numbered available
            assignments.append((payload_idx, ancilla))
            used_qubits.add(ancilla)
            if verbose:
                print(f"  ✅ Assigned ancilla {ancilla} to payload {payload_idx}")
                print(f"  Current assignments: {assignments}")
        else:
            if verbose:
                print(f"  ❌ No available neighbors for payload {payload_idx}")
            # Don't return None immediately - try next payload qubit
    
    if verbose:
        print(f"\nFinal assignments: {assignments}")
        print(f"Required: {num_checks}, Found: {len(assignments)}")
    
    return assignments if len(assignments) >= num_checks else None


def validate_layout_for_checks(payload_layout: List[int], backend, max_checks: int, verbose: bool = False) -> bool:
    """Check if payload layout can support max_checks unique PCS checks."""
    connectivity = build_connectivity_graph(backend)
    if verbose:
        print("connectivity graph = ", connectivity)
    assignments = find_available_ancillas(payload_layout, connectivity, max_checks, verbose=verbose)
    if verbose:
        print("assignments = ", assignments)
    return assignments is not None


def find_best_payload_layout(circuit: QuantumCircuit, backend, max_checks: int, verbose: bool = True) -> List[int]:
    """
    Find the best payload layout that supports max_checks unique PCS checks.
    
    Returns:
        Payload layout (list of physical qubit indices)
    """
    candidate_layouts = find_best_layouts(circuit, backend)
    
    for layout in candidate_layouts:
        if verbose:
            print("testing layout: ", layout)
        if validate_layout_for_checks(layout, backend, max_checks, verbose=verbose):
            return layout
    
    raise RuntimeError(f"No payload layout found supporting {max_checks} unique PCS checks")


def get_check_assignments(payload_layout: List[int], backend, num_checks: int, verbose: bool = False) -> List[Tuple[int, int]]:
    """
    Get PCS check assignments for a validated payload layout.
    
    Returns:
        List of (payload_idx, ancilla_physical) tuples
    """
    connectivity = build_connectivity_graph(backend)
    assignments = find_available_ancillas(payload_layout, connectivity, num_checks, verbose=verbose)
    
    if assignments is None:
        raise ValueError(f"Cannot assign {num_checks} unique checks to payload layout {payload_layout}")
    
    return assignments


def build_pcs_config(circuit: QuantumCircuit, payload_layout: List[int], num_checks: int, backend, verbose: bool = False) -> Dict:
    """
    Build PCS configuration for specific number of checks.
    
    Returns:
        Dictionary with 'circuit', 'layout', 'check_indices', 'ancilla_positions'
    """
    assignments = get_check_assignments(payload_layout, backend, num_checks, verbose=verbose)
    
    check_indices = [pair[0] for pair in assignments]
    ancilla_positions = [pair[1] for pair in assignments]
    full_layout = payload_layout + ancilla_positions
    
    pcs_circuit = build_pcs_circuit(circuit, check_indices, check_type='Z', barriers=True)
    
    return {
        'circuit': pcs_circuit,
        'layout': full_layout,
        'check_indices': check_indices,
        'ancilla_positions': ancilla_positions,
        'num_checks': num_checks
    }