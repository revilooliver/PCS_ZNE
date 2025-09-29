"""
Simple utility to apply PCS checks at custom positions in a circuit.
"""

from qiskit import QuantumCircuit
from utils.pauli_checks import convert_to_PCS_circ
from typing import Tuple, List


def protect_circuit_portion(circuit: QuantumCircuit, 
                           start_instruction: int,
                           end_instruction: int,
                           num_checks: int,
                           only_Z_checks: bool = True,
                           only_even_q: bool = False,
                           barriers: bool = False) -> Tuple[List[str], QuantumCircuit]:
    """
    Apply PCS checks around a specific portion of a circuit.
    
    Args:
        circuit: Original circuit
        start_instruction: Index where protection starts (inclusive)
        end_instruction: Index where protection ends (exclusive)
        num_checks: Number of check qubits
        only_Z_checks: Whether to use only Z checks
        only_even_q: Whether to use only even check qubits  
        barriers: Whether to add barriers
        
    Returns:
        (signs, protected_circuit) where:
        - signs: List of sign factors for checks
        - protected_circuit: Circuit with PCS checks around the specified portion
    """
    num_qubits = circuit.num_qubits
    instructions = list(circuit.data)
    
    # Split circuit into three parts: before, protected, after
    before_part = QuantumCircuit(num_qubits)
    protected_part = QuantumCircuit(num_qubits)
    after_part = QuantumCircuit(num_qubits)
    
    # Before part (0 to start_instruction)
    for i in range(start_instruction):
        if i < len(instructions):
            instruction, qargs, cargs = instructions[i]
            before_part.append(instruction, qargs, cargs)
    
    # Protected part (start_instruction to end_instruction)
    for i in range(start_instruction, end_instruction):
        if i < len(instructions):
            instruction, qargs, cargs = instructions[i]
            protected_part.append(instruction, qargs, cargs)
    
    # After part (end_instruction to end)
    for i in range(end_instruction, len(instructions)):
        instruction, qargs, cargs = instructions[i]
        after_part.append(instruction, qargs, cargs)
    
    print(f"Circuit split:")
    print(f"  Before: {len(before_part.data)} instructions")
    print(f"  Protected: {len(protected_part.data)} instructions") 
    print(f"  After: {len(after_part.data)} instructions")
    
    # Apply PCS to protected part
    if protected_part.depth() == 0:
        print("Warning: No instructions to protect")
        return [], circuit
    
    signs, pcs_protected = convert_to_PCS_circ(
        protected_part,
        num_qubits=num_qubits,
        num_checks=num_checks,
        barriers=barriers,
        only_Z_checks=only_Z_checks,
        only_even_q=only_even_q
    )
    
    # Build final circuit: before + PCS(protected) + after
    total_qubits = num_qubits + num_checks
    final_circuit = QuantumCircuit(total_qubits)
    
    # Add before part (only on payload qubits)
    for instruction, qargs, cargs in before_part.data:
        mapped_qargs = [final_circuit.qubits[q._index if hasattr(q, '_index') else circuit.qubits.index(q)] for q in qargs]
        final_circuit.append(instruction, mapped_qargs, cargs)
    
    # Add PCS protected part
    for instruction, qargs, cargs in pcs_protected.data:
        final_circuit.append(instruction, qargs, cargs)
    
    # Add after part (only on payload qubits)  
    for instruction, qargs, cargs in after_part.data:
        mapped_qargs = [final_circuit.qubits[q._index if hasattr(q, '_index') else circuit.qubits.index(q)] for q in qargs]
        final_circuit.append(instruction, mapped_qargs, cargs)
    
    print(f"Final circuit: {final_circuit.num_qubits} qubits, {final_circuit.depth()} depth")
    
    return signs, final_circuit