"""
Simple utility to split Hamiltonian simulation circuits after the H+RZ layers.
"""

from qiskit import QuantumCircuit
from qiskit_addon_utils.slicing import slice_by_depth, combine_slices
from typing import Tuple
from IPython.display import display


def split_hamiltonian_circuit(circuit: QuantumCircuit, time_step: int = 0, total_time: int = 0) -> Tuple[QuantumCircuit, QuantumCircuit]:
    """
    Split a Hamiltonian circuit after the first two layers (H + RZ).
    
    Args:
        circuit: Original Hamiltonian simulation circuit
        time_step: Time step parameter (unused, for compatibility)
        total_time: Total time parameter (unused, for compatibility)
        
    Returns:
        (first_part, remaining_part) where:
        - first_part: First two layers (H + RZ)
        - remaining_part: Everything else
    """
    # Split into slices of depth 2 (assumes H layer + RZ layer = depth 2)
    slices = slice_by_depth(circuit, 2)

    combined = combine_slices(slices, include_barriers=True)
    display(combined.draw("mpl", fold=-1))
    
    # First part is the first slice (first two layers)
    first_part = slices[0]
    
    # Rest is everything after the first slice
    rest_of_circ = combine_slices(slices[1:]) if len(slices) > 1 else QuantumCircuit(circuit.num_qubits)
    
    print(f"Split circuit into {len(slices)} depth-2 slices:")
    print(f"  First part: {first_part.depth()} depth, {len(first_part.data)} instructions")
    print(f"  Remaining part: {rest_of_circ.depth()} depth, {len(rest_of_circ.data)} instructions")
    
    return first_part, rest_of_circ