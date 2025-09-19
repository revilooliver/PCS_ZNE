"""
Simple graph generator that finds graphs with high ZZ expectations for QAOA.
"""

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def get_mean_zz_magnitude(G, gamma, beta):
    """
    Get mean |ZZ| magnitude for QAOA state.
    
    Args:
        G: Graph
        gamma: QAOA gamma parameter
        beta: QAOA beta parameter  
        
    Returns:
        mean |ZZ| magnitude
    """
    n = G.number_of_nodes()
    
    # Build simple QAOA circuit
    qc = QuantumCircuit(n)
    
    # Initial state |+⟩^n
    for i in range(n):
        qc.h(i)
    
    # Cost layer: ZZ rotations
    for edge in G.edges():
        i, j = edge
        qc.rzz(2 * gamma, i, j)
    
    # Mixer layer: X rotations
    for i in range(n):
        qc.rx(2 * beta, i)
    
    # Get final state
    state = Statevector.from_instruction(qc)
    
    # Calculate ZZ expectations for all pairs
    zz_magnitudes = []
    for i in range(n):
        for j in range(i+1, n):
            pauli_string = ['I'] * n
            pauli_string[i] = 'Z'
            pauli_string[j] = 'Z'
            pauli_op = SparsePauliOp.from_list([(''.join(pauli_string), 1.0)])
            zz_exp = abs(state.expectation_value(pauli_op).real)
            zz_magnitudes.append(zz_exp)
    
    return np.mean(zz_magnitudes) if zz_magnitudes else 0


def generate_random_graph_with_high_zz(num_qubits, max_attempts=50, zz_threshold=0.8):
    """
    Generate random graphs until finding one with high ZZ expectations.
    
    Args:
        num_qubits: Number of qubits/nodes
        max_attempts: Maximum graphs to try
        zz_threshold: Minimum mean |ZZ| value required
        
    Returns:
        (graph, gamma, beta, maxcut_value)
    """
    
    # Try a few standard parameter sets
    param_sets = [
        (np.pi/4, np.pi/4),    
        (np.pi/3, np.pi/6),    
        (np.pi/2, np.pi/4),    
        (0.8, 0.6),            
    ]
    
    for attempt in range(max_attempts):
        # Generate random connected graph
        if num_qubits <= 3:
            G = nx.complete_graph(num_qubits)
        else:
            graph_type = attempt % 4
            
            if graph_type == 0:
                degree = min(3, num_qubits - 1)
                G = nx.random_regular_graph(degree, num_qubits)
            elif graph_type == 1:
                p = 0.5
                G = nx.erdos_renyi_graph(num_qubits, p)
            elif graph_type == 2:
                G = nx.path_graph(num_qubits)
            else:
                G = nx.cycle_graph(num_qubits)
        
        # Ensure connected
        if not nx.is_connected(G):
            continue
            
        # Try each parameter set
        for gamma, beta in param_sets:
            try:
                mean_zz = get_mean_zz_magnitude(G, gamma, beta)
                
                if mean_zz >= zz_threshold:
                    # Found a good graph!
                    maxcut = G.number_of_edges()  # Approximate
                    
                    print(f"✅ Found graph with high ZZ (attempt {attempt + 1}):")
                    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                    print(f"   Parameters: γ={gamma:.3f}, β={beta:.3f}")
                    print(f"   Mean |ZZ|: {mean_zz:.3f} (>{zz_threshold})")
                    
                    return G, [gamma], [beta], maxcut
                    
            except Exception:
                continue
    
    # Fallback: just return a complete graph
    print(f"❌ Using fallback complete graph")
    G = nx.complete_graph(num_qubits)
    gamma, beta = np.pi/4, np.pi/4
    maxcut = G.number_of_edges()
    
    return G, [gamma], [beta], maxcut


if __name__ == "__main__":
    # Test
    G, gamma, beta, maxcut = generate_random_graph_with_high_zz(6)
    print(f"Result: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Params: γ={gamma[0]:.3f}, β={beta[0]:.3f}")
    print(f"MaxCut: {maxcut}")