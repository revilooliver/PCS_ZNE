# author: Ji Liu (ji.liu@anl.gov)
import numpy as np
import itertools
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.opflow.converters import AbelianGrouper
from qiskit.dagcircuit.dagnode import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.converters import circuit_to_dag

def read_from_file(filename):
    #read from a file, return the hamiltonian list
    Hamiltonian_list = []
    f = open(filename,"r") # Here we read in the qubit Hamiltonian from the .txt file
    for line in f:
        a = line.strip().split(',')
        c = (str(a[0]), float(a[1]))
        Hamiltonian_list.append(c)
    return Hamiltonian_list

def find_commute_groups(Hamiltonian_input, is_list = True):
    if is_list is True:
        #find the commute groups in the list of paulis.
        SumOp = PauliSumOp.from_list(Hamiltonian_list)
    else:
        SumOp = Hamiltonian_input
    commute_groups = AbelianGrouper.group_subops(SumOp)
    pauli_commute = []
    for group in commute_groups:
        pauli_commute.append(group.primitive.to_list())
    return pauli_commute

def MeasureCircuit(Sum_Op, num_qubits, num_qargs):
    # Determine how many commute groups are in the SummedOp
    num_terms = len(Sum_Op)

    # Find the Paulis with least number of I in it(shoud be 0).
    # The problem is here. In this case, not not all the subgroups have a term with no I in it. So we have to loop over all
    # of the terms to construct a Pauli string.
    Pauli = ''

    for i in range(num_qubits):
        intermed = []
        for j in range(num_terms):
            intermed.append(Sum_Op[j][0][i])
        if 'X' in intermed:
            Pauli += 'X'
        elif 'Y' in intermed:
            Pauli += 'Y'
        else:
            Pauli += 'Z'

    if len(Pauli) != num_qubits:
        raise Exception('The length does not equal, traverse has problem.')

    Pauli_string = Pauli[::-1]  # This has reversed the order.
    # Now Pauli_string is the target that we should use to construct the measurement circuit.

    qc = QuantumCircuit(num_qargs)
#     qc.barrier()
    print(Pauli_string)
    
    for i in range(num_qubits):
        if Pauli_string[i] == 'X':
            qc.u(np.pi / 2, 0, np.pi, i)
        if Pauli_string[i] == 'Y':
            qc.u(np.pi / 2, 0, np.pi / 2, i)
        else:
            None

    return qc

def evaluation(d: dict, shots: int, Pauli: str):
    # This Pauli_string is in arbitrary form of I, X, Y and Z.
    # Determine the number of qubits, which is also related to the number of measurement outcomes.
    num_qubits = len(Pauli)
    Pauli_string = ''
    for i in Pauli:
        if i == 'I':
            Pauli_string += i
        else:
            Pauli_string += 'Z'

    def kbits(n):
        result = []
        for k in range(0, n + 1):
            for bits in itertools.combinations(range(n), k):
                s = ['0'] * n
                for bit in bits:
                    s[bit] = '1'
                result.append(''.join(s))
        return result

    # Generate all binary strings of N bits.
    outcomes = kbits(num_qubits)

    def get_from(d: dict, key: str):
        value = 0
        if key in d:
            value = d[key]
        return value

    # Here we compute the expectation value.
    expectation_value = 0
    for i in outcomes:
        intermediate = 0
        for j in range(num_qubits):
            if (Pauli_string[j] == 'Z') and (i[j] == '1'):
                intermediate += 1
            else:
                None

        if (intermediate % 2) == 0:
            expectation_value += get_from(d, i)
        else:
            expectation_value -= get_from(d, i)

    expectation_value = expectation_value / shots

    return expectation_value


def apply_checking_circuit(qc, ctrl_bits, ancilla_bits, side = None, phase_z = None, phase_y = None, x = None):
    if len(ctrl_bits) != len(ancilla_bits):
        print("Size mismatch")
        return None
    if side == 'front':
        for i in ancilla_bits:
            qc.h(i)
        if x is True:
            qc.x(ctrl_bits)
        qc.ry(phase_y, ctrl_bits)
        qc.rz(phase_z, ctrl_bits)
        for j,k in zip(ctrl_bits, ancilla_bits):
            qc.cz(j, k)
        qc.rz(-phase_z, ctrl_bits)
        qc.ry(-phase_y, ctrl_bits)
        if x is True:
            qc.x(ctrl_bits)
    elif side == 'end':
        qc.rz(-phase_z, ctrl_bits)
        qc.ry(-phase_y, ctrl_bits)
        for j,k in zip(ctrl_bits, ancilla_bits):
            qc.cz(j, k)
        qc.ry(phase_y, ctrl_bits)
        qc.rz(phase_z, ctrl_bits)
        for i in ancilla_bits:
            qc.h(i)
    else:
        print("Side undefined")

def create_check_circuit(ansatz, qubit_id, pre_list, post_list):
    #first create the circuit with same quantum registers
    qc = QuantumCircuit(ansatz.qubits)
    ancilla = QuantumRegister(1, "ancilla")
    qc.add_register(ancilla)
    qc.h(ancilla)
    
    targ_qr = ansatz.qubits[qubit_id]
    #apply the single-qubit gates in pre_list
    for node in pre_list:
        qc.append(node.op, qargs = [targ_qr])
    qc.cz(targ_qr, ancilla)
    #apply the inverse of the single-qubit gates
    for node in pre_list[::-1]:
        qc.append(node.op.inverse(), qargs = [targ_qr])
    
    #add the original circuit
    qc.compose(ansatz, inplace=True)
    
    #apply the inverse of the single-qubit gates in post_list
    for node in post_list[::-1]:
        qc.append(node.op.inverse(), qargs = [targ_qr])
    qc.cz(targ_qr, ancilla)
    #apply the single-qubit gates in post_list
    for node in post_list:
        qc.append(node.op, qargs = [targ_qr])
    qc.h(ancilla)
    
    return qc
    
    
def create_final_circuit(ansatz, qubit_id):
    ansatz_dag = circuit_to_dag(ansatz)
    gen = ansatz_dag.nodes_on_wire(ansatz_dag.wires[qubit_id])
    
    #traverse the operations on a wire and store the nodes in a list:
    node_list = []
    for node in gen:
        if type(node) is DAGOpNode:
            node_list.append(node)
            
    #find the first sequence of single-qubit operations:
    pre_list = []
    for node in node_list:
        if node.op.num_qubits == 1:
            pre_list.append(node)
        elif node.op.num_qubits > 1:
            break
            
    #find the last sequence of single-qubit operations:
    post_list = []
    for node in node_list[::-1]:
        if node.op.num_qubits == 1:
            post_list.append(node)
        elif node.op.num_qubits > 1:
            break   
    post_list = post_list[::-1]
    
    final_qc = create_check_circuit(ansatz, qubit_id, pre_list, post_list)
    
    #calculate the index for the two cuts:
    init_cut_index = len(pre_list) * 2 + 1
    final_cut_index = init_cut_index + len(node_list)
            
    return final_qc, init_cut_index, final_cut_index