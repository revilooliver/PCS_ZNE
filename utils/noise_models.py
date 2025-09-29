"""
Custom noise models for PCS experiments.
"""

import numpy as np
from qiskit_aer import noise
from typing import List, Optional


def create_payload_only_noise_model(
    num_payload_qubits: int,
    num_check_qubits: int,
    prob_1_qubit: float = 5e-4,
    prob_2_qubit: float = 5e-3,
    basis_gates: Optional[List[str]] = None
) -> noise.NoiseModel:
    """
    Create a noise model that applies errors only to payload qubits.
    Check qubits remain noiseless to isolate PCS error correction effectiveness.
    
    Args:
        num_payload_qubits: Number of data/payload qubits
        num_check_qubits: Number of check qubits (will be noiseless)
        prob_1_qubit: Single-qubit depolarizing error probability
        prob_2_qubit: Two-qubit depolarizing error probability  
        basis_gates: List of basis gates to apply noise to
        
    Returns:
        NoiseModel with selective qubit noise
    """
    if basis_gates is None:
        basis_gates = ['u1', 'u2', 'u3', 'sx', 'x', 'rx', 'rz', 'ry', 'h'] 
    
    two_qubit_gates = ['cx', 'cz', 'swap']

    # Create error models
    error_1 = noise.depolarizing_error(prob_1_qubit, 1)
    error_2 = noise.depolarizing_error(prob_2_qubit, 2)
    
    noise_model = noise.NoiseModel()
    
    # Apply single-qubit errors only to payload qubits (0 to num_payload_qubits-1)
    payload_qubits = list(range(num_payload_qubits))
    
    for gate in basis_gates:
            for qubit in payload_qubits:
                if gate not in two_qubit_gates:
                    noise_model.add_quantum_error(error_1, gate, [qubit])
    
    # Apply two-qubit errors only to payload-payload pairs
    for gate in two_qubit_gates:
        # Add noise to all payload-payload combinations
        for i in payload_qubits:
            for j in payload_qubits:
                if i != j:
                    noise_model.add_quantum_error(error_2, gate, [i, j])
    
    print(f"Created payload-only noise model:")
    print(f"  - Payload qubits: {payload_qubits} (noisy)")
    print(f"  - Check qubits: {list(range(num_payload_qubits, num_payload_qubits + num_check_qubits))} (noiseless)")
    print(f"  - Single-qubit error rate: {prob_1_qubit}")
    print(f"  - Two-qubit error rate: {prob_2_qubit}")

    print(noise_model)
    
    return noise_model