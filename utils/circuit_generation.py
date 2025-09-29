"""
Circuit generation utilities.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Reset
from qiskit.circuit.library import XGate, YGate, ZGate, HGate, SGate, SdgGate, CXGate
import numpy as np
from typing import List


def random_clifford_circuit(num_qubits, depth, max_operands=2, measure=False,
                            conditional=False, reset=False, seed=None):
    """
    Generate a random circuit composed exclusively of Clifford gates.

    Only one-qubit and two-qubit Clifford gates are used. For one-qubit gates,
    the following are selected:
      IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate.
    For two-qubit Clifford gates, the following are selected:
      CXGate, CZGate, SwapGate.

    """
    if max_operands < 1 or max_operands > 2:
        raise ValueError("max_operands for Clifford circuits must be 1 or 2.")

    # Define allowed Clifford gates.
    # one_q_ops = [IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate]
    one_q_ops = [XGate, YGate, ZGate, HGate, SGate, SdgGate]
    # two_q_ops = [CXGate, CZGate, SwapGate]
    two_q_ops = [CXGate]

    # Add Reset gate if desired.
    if reset:
        one_q_ops.append(Reset)

    # Initialize registers.
    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qr)

    # Add a classical register if measurements or conditionals are desired.
    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    # Set the random seed.
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # Build the circuit layer by layer.
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        # Randomize the qubit order for each layer.
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            # For Clifford circuits, only 1- and 2-qubit gates are supported.
            # If a two-qubit gate is possible pick from [1, 2] randomly.
            if len(remaining_qubits) >= 2:
                num_operands = rng.choice([1, 2])
            else:
                num_operands = 1

            # For 2-qubit selection, choose the first two qubits.
            operands = [remaining_qubits.pop(0) for _ in range(num_operands)]

            # Pick appropriate random gate.
            if num_operands == 1:
                op_class = rng.choice(one_q_ops)
            elif num_operands == 2:
                op_class = rng.choice(two_q_ops)

            op = op_class()

            # Optionally add a conditional.
            if conditional and rng.choice(range(10)) == 0:
                # Pick a random bit value.
                value = rng.integers(0, 2 ** num_qubits)
                op.condition = (cr, value)
            qc.append(op, [qr[q] for q in operands])

    # Add measurements if requested.
    if measure:
        qc.measure(qr, cr)

    return qc


def generate_rand_mirror_cliff(num_qubits: int, depth: int) -> QuantumCircuit:
    """Generate a random mirrored Clifford circuit."""
    # Generate random Clifford circuit
    base_circuit = random_clifford_circuit(num_qubits, depth)
    
    # Add barrier and inverse
    # base_circuit.barrier()
    mirrored_circuit = base_circuit.compose(base_circuit.inverse())
    
    return mirrored_circuit


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


def build_pcs_circuit(base_circuit: QuantumCircuit, check_indices: List[int], 
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