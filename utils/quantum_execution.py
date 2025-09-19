"""
Simple quantum circuit execution utilities.
Each function does exactly one thing well.
"""

from typing import Dict, List, Optional, Union
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Qubit
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers
import mthree

from IPython.display import display


def execute_circuit(
    circuit: QuantumCircuit,
    backend,
    shots: int,
    layout: Optional[Union[Dict[Qubit, int], List[int]]] = None,
    twirling: bool = False,
    mthree=None
) -> Dict[str, int]:
    """
    Execute a quantum circuit and return measurement counts.
    
    Args:
        circuit: QuantumCircuit to execute (must have measurements)
        backend: Qiskit backend
        shots: Number of shots
        layout: Optional qubit layout
        twirling: Enable gate/measure twirling
        mthree: Optional M3Mitigation object
        
    Returns:
        Dict[bitstring, count]: Measurement counts
    """
    # Transpile for backend
    transpiled = transpile_circuit(circuit, backend, layout)
    
    # Execute on backend
    sampler = Sampler(mode=backend)
    
    if twirling:
        sampler.options.twirling.enable_measure=False # This is the trex twirling.
        sampler.options.twirling.shots_per_randomization= "auto"
        sampler.options.twirling.strategy= "active-circuit" # I usually use this setting, but there are other ones.
        sampler.options.twirling.enable_gates=True
    
    # print("executing the following transpiled circuit:")
    # display(transpiled.draw("mpl", fold=-1))
    job = sampler.run([transpiled], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    
    # Apply readout mitigation if provided
    if mthree and layout:
        physical_qubits = list(layout.values()) if isinstance(layout, dict) else layout
        counts = mthree.apply_correction(counts, physical_qubits)
    
    return counts


def transpile_circuit(
    circuit: QuantumCircuit, 
    backend, 
    layout: Optional[Union[Dict, List]] = None,
    optimization_level: int = 0
) -> QuantumCircuit:
    """
    Transpile circuit for backend.
    
    Args:
        circuit: Circuit to transpile
        backend: Target backend
        layout: Optional initial layout
        optimization_level: Optimization level (0-3)
        
    Returns:
        Transpiled circuit
    """
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        initial_layout=layout
    )
    
    transpiled = pm.run(circuit)
    
    # Remove barriers for execution
    remove_barriers = RemoveBarriers()
    return remove_barriers(transpiled)


def apply_pauli_basis(circuit: QuantumCircuit, pauli_string: str) -> QuantumCircuit:
    """
    Rotate circuit to measure Pauli string in computational basis.
    
    Args:
        circuit: Input circuit
        pauli_string: Pauli string (e.g., "XYZI")
        
    Returns:
        New circuit with basis rotations applied
    """
    rotated = circuit.copy()
    
    for i, pauli in enumerate(pauli_string):
        if pauli == 'X':
            rotated.h(i)
        elif pauli == 'Y':
            rotated.sdg(i)
            rotated.h(i)
        # Z and I require no rotation
    
    return rotated


def compute_z_expectation(counts: Dict[str, int]) -> float:
    """
    Compute expectation value assuming all qubits measured in Z basis.
    
    Args:
        counts: Measurement counts
        
    Returns:
        Expectation value in [-1, +1]
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    expectation = 0
    for bitstring, count in counts.items():
        # Compute (-1)^(number of 1s) = parity
        parity = (-1) ** bitstring.count('1')
        expectation += parity * count
    
    return expectation / total


def calibrate_mthree(backend, qubits: List[int], shots: int = 10000):
    """
    Calibrate M3 readout mitigation.
    
    Args:
        backend: Quantum backend
        qubits: Physical qubit indices to calibrate
        shots: Calibration shots
        
    Returns:
        Calibrated M3Mitigation object
    """
    
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(qubits, shots=shots)
    return mit