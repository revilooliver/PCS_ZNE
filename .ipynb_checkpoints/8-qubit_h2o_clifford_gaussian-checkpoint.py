#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time, random
import numpy as np
# import pennylane as qml
# from qiskit import Aer, transpile, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Pauli, Statevector
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, edgeitems=10, linewidth=150, suppress=True)


# In[ ]:


import qiskit
import itertools
from qiskit import *
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.synthesis import synth_clifford_full
from qiskit.quantum_info import hellinger_fidelity as hf

from utils.pauli_checks import ChecksFinder, add_pauli_checks, add_meas_pauli_checks, add_linear_meas_pauli_checks,  search_for_pauli_list
from utils.pauli_checks import gen_initial_layout, gen_final_layout, complete_postprocess, filter_results

from utils.utils import norm_dict, total_counts
# from utils.vqe_utils import evaluation


# In[ ]:


print(qiskit.version.get_version_info())


# #### I. Calibrating $\tilde{f}$ in the noisy Clifford channel using hardware

# In[ ]:


total_trials = 10000
num_qubits = 8
def calibration_circuit(Clifford):
    qc = QuantumCircuit(num_qubits)
    
    clifford_circuit = Clifford.to_circuit()
    # qc.compose(clifford_circuit, qubits=[0,1,2,3], inplace=True)
    qc.compose(clifford_circuit, qubits=range(num_qubits), inplace=True)
    
    qc.measure_all()
    return qc


# In[ ]:


cali_C_list = []
for i in range(total_trials):
    # Clifford = random_clifford(4)
    Clifford = random_clifford(num_qubits)
    cali_C_list.append(Clifford)
    
cali_circs = []
for i in range(total_trials):
    circuit = calibration_circuit(cali_C_list[i])
    cali_circs.append(circuit)


# In[ ]:


print(cali_circs[0])


# Set noise model and topolgoy

# In[ ]:

"""
homogeneous noise
"""
# # from qiskit_ibm_runtime import Session, Options, SamplerV2 as Sampler
# from qiskit_ibm_runtime import Session, Sampler, Options
# from qiskit_ibm_runtime.fake_provider import *
# from qiskit_aer import AerSimulator
# import qiskit_aer.noise as noise
# from itertools import combinations

# # Make a noise model
# fake_backend = FakeCairo()
# # noise_model = noise.NoiseModel.from_backend(fake_backend)

# prob_1 = 0.002  # 1-qubit gate
# prob_2 = 0.02   # 2-qubit gate

# error_1 = noise.depolarizing_error(prob_1, 1)
# error_2 = noise.depolarizing_error(prob_2, 2)

# noise_model = noise.NoiseModel()
# noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'sx', 'x'])
# noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

# options = Options(optimization_level=2, resilience_level=1) # choose the proper levels on hardware
# options.simulator = {
#     "noise_model": noise_model,
#     "basis_gates": fake_backend.configuration().basis_gates,
#     # "coupling_map": fake_backend.configuration().coupling_map,
#     "seed_simulator": 42
# }

# # backend = service.get_backend("") 
# # backend = "ibmq_qasm_simulator" # use the simulator for now
# backend = AerSimulator()

"""
gaussian noise
"""
# from qiskit_ibm_runtime import Session, Options, SamplerV2 as Sampler
from qiskit_ibm_runtime import Session, Sampler, Options
from qiskit_ibm_runtime.fake_provider import *
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
from itertools import combinations

fake_backend = FakeCairo()

noisy_qubits = [0, 1, 2, 3, 4, 5, 6, 7]
noisy_pairs = list(combinations(noisy_qubits, 2))

mean_prob_1, std_dev_1 = 0.002, 0.0005  # 1-qubit gates
mean_prob_2, std_dev_2 = 0.02, 0.005   # 2-qubit gates

# Generate random error rates following a Gaussian distribution
np.random.seed(42) 
error_rates_1 = np.random.normal(mean_prob_1, std_dev_1, len(noisy_qubits))
error_rates_2 = np.random.normal(mean_prob_2, std_dev_2, len(noisy_pairs))

error_rates_1 = np.clip(error_rates_1, 0, 1)
error_rates_2 = np.clip(error_rates_2, 0, 1)

print("1-qubit error rates:", error_rates_1)
print("2-qubit error rates:", error_rates_2)

# Create a noise model
noise_model = noise.NoiseModel()

for i, noisy_qbt in enumerate(noisy_qubits):
    error_1 = noise.depolarizing_error(error_rates_1[i], 1)
    noise_model.add_quantum_error(error_1, ['u1', 'u2', 'u3', 'sx', 'x'], [noisy_qbt])

for i, noisy_pair in enumerate(noisy_pairs):
    error_2 = noise.depolarizing_error(error_rates_2[i], 2)
    noise_model.add_quantum_error(error_2, ['cx'], list(noisy_pair))
    noise_model.add_quantum_error(error_2, ['cx'], list(reversed(noisy_pair)))

options = Options(optimization_level=2, resilience_level=1) # choose the proper levels on hardware
options.simulator = {
    "noise_model": noise_model,
    "basis_gates": fake_backend.configuration().basis_gates,
#     "coupling_map": fake_backend.configuration().coupling_map,
    "seed_simulator": 42
}
#backend = service.get_backend("") 
# backend = "ibmq_qasm_simulator" # use the simulator for now
backend = AerSimulator()


# In[ ]:


print(noise_model)


# In[ ]:


with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)

    # define physical qubits to be used in the layout arguement
    job = sampler.run(cali_circs, shots=100)
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")

    result = job.result()

    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()


# In[ ]:


cali_b_lists = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits]: result.quasi_dists[i].binary_probabilities().get(key)})
    cali_b_lists.append(di)


# In[ ]:


def calibrating_f(cali_b_lists, cali_C_list, num_qubits):
    d = 2**num_qubits
    num_snapshots = len(cali_C_list)
    
    f_tilde = 0.
    for b_dict, clifford in zip(cali_b_lists, cali_C_list):
        F = computing_F(b_dict, clifford, num_qubits)
        f_tilde += np.real((d*F - 1) / (d - 1))
    
    return f_tilde / num_snapshots


def computing_F(b_dict, clifford, num_qubits):
    zero_state = state_reconstruction('0'*num_qubits)
    U = clifford.to_matrix()
    
    F = 0. + 0.j
    denom = 0.
    for b_state in list(b_dict.keys()):
        F += np.trace(zero_state @ U.T.conj() @ state_reconstruction(b_state) @ U) * b_dict.get(b_state)
        denom += b_dict.get(b_state)
    return F / denom


def state_reconstruction(b_str: str):
    '''
    '''
    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0], [0,1]])
    rho = [1]
    for i in b_str:
        state_i = zero_state if i=='0' else one_state
        rho = np.kron(rho, state_i)
    return rho


# In[ ]:


f_tilde = calibrating_f(cali_b_lists, cali_C_list, num_qubits)
print(f'The calibrated f_tilde is {f_tilde}; while the noiseless reference is {1/(2**num_qubits+1)}')

# #### II. Perform the standard shadow experiments

# In[ ]:


from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector

def construct_qcc_circuit(entanglers: list):
    '''This function defines the QCC ansatz circuit for VQE. Here we construct exponential blocks using
    entanglers from QMF state as a proof of principle demonstration.
    
    Args:
        entanglers: list storing Pauli words for construction of qcc_circuit.
        backend: statevector, qasm simulator or a real backend.
        truncation: a threshold number to truncate the blocks. Default: None.
    Returns:
        qcc_circuit
    '''
    num_blocks = len(entanglers)
    # p = ParameterVector('p', num_blocks)
    p = [1.16692654, 0.27223177, -0.93402707, -0.92067998, 0.06852241, -0.42444632, -0.41270851, -0.01068001]
    
    num_qubits = len(entanglers[0])
    qcc_circuit = QuantumCircuit(num_qubits)
    for i in range(num_blocks):
        circuit = QuantumCircuit(num_qubits)
        key = entanglers[i]
        coupler_map = []
        
        # We first construct coupler_map according to the key.
        for j in range(num_qubits):
            if key[num_qubits-1-j] != 'I':
                coupler_map.append(j)
                
        # Then we construct the circuit.
        if len(coupler_map) == 1:
            # there is no CNOT gate.
            c = coupler_map[0]
            if key[num_qubits-1-c] == 'X':
                circuit.h(c)
                circuit.rz(p[i], c)
                circuit.h(c)
            elif key[num_qubits-1-c] == 'Y':
                circuit.rx(-np.pi/2, c)
                circuit.rz(p[i], c)
                circuit.rx(np.pi/2, c)
                
            qcc_circuit += circuit
        else:
            # Here we would need CNOT gate.
            for j in coupler_map:
                if key[num_qubits-1-j] == 'X':
                    circuit.h(j)
                elif key[num_qubits-1-j] == 'Y':
                    circuit.rx(-np.pi/2, j)
                    
            for j in range(len(coupler_map) - 1):
                circuit.cx(coupler_map[j], coupler_map[j+1])
                
            param_gate = QuantumCircuit(num_qubits)
            param_gate.rz(p[i], coupler_map[-1])
            
            #qcc_circuit += circuit + param_gate + circuit.inverse()
            qcc_circuit.compose(circuit, inplace=True)
            qcc_circuit.compose(param_gate, inplace=True)
            qcc_circuit.compose(circuit.inverse(), inplace=True)
    
    return qcc_circuit



# In[ ]:


# define the ansatz circuit

def hf_circ(num_qubits, num_checks):
    total_qubits = num_qubits + num_checks
    hf_circuit = QuantumCircuit(total_qubits)

    hf_circuit.x(0)
    hf_circuit.x(1)
    hf_circuit.x(2)
    hf_circuit.x(3)
    
    entanglers = ['XXIIIIXY', 'IIXXXYII', 'IXXIXIIY', 'XIIXIXYI',
                  'XXIIXYII', 'IXIXIXIY', 'XIXIXIYI', 'IIXXIIXY']

    parameterized_circuit = hf_circuit.compose(construct_qcc_circuit(entanglers))
    
    return parameterized_circuit 

def hydrogen_shadow_circuit(Clifford, num_qubits):
    qc = hf_circ(num_qubits, num_checks=0)
    
    clifford_circuit = Clifford.to_circuit()
    # qc.compose(clifford_circuit, qubits=[0,1,2,3], inplace=True)
    qc.barrier()
    qc.compose(clifford_circuit, qubits=range(num_qubits), inplace=True)
    qc.barrier()
    
    qc.measure_all()
    return qc

def hydrogen_shadow_PCS_circuit(Clifford, num_qubits, num_checks):
    total_qubits = num_qubits + num_checks
    qc = hf_circ(num_qubits, num_checks)

    clif_qc = Clifford.to_circuit()
    
    characters = ['I', 'Z']
    strings = [''.join(p) for p in itertools.product(characters, repeat=num_qubits)]
    
    test_finder = ChecksFinder(num_qubits, clif_qc)
    p1_list = []
    for string in strings:
        string_list = list(string)
        result = test_finder.find_checks_sym(pauli_group_elem = string_list)
        #print(result.p1_str, result.p2_str)
        p1_list.append([result.p1_str, result.p2_str])
        
    sorted_list = sorted(p1_list, key=lambda s: s[1].count('I'))
    pauli_list = sorted_list[-num_qubits -1:-1]
    
    #
    initial_layout = {}
    for i in range(0, num_qubits):
        initial_layout[i] = [i]

    final_layout = {}
    for i in range(0, num_qubits):
        final_layout[i] = [i]
        
    #add pauli check on two sides:
    #specify the left and right pauli strings
    pcs_qc_list = []
    sign_list = []
    pl_list = []
    pr_list = []

    for i in range(0, num_checks):
        pl = pauli_list[i][0][2:]
        pr = pauli_list[i][1][2:]
        if i == 0:
            temp_qc = add_pauli_checks(clif_qc, pl, pr, initial_layout, final_layout, False, False, False, False, False)
            save_qc = add_pauli_checks(clif_qc, pl, pr, initial_layout, final_layout, False, False, False, False, False)
            prev_qc = temp_qc
        else:
            temp_qc = add_pauli_checks(prev_qc, pl, pr, initial_layout, final_layout, False, False, False, False, False)
            save_qc = add_pauli_checks(prev_qc, pl, pr, initial_layout, final_layout, False, False, False, False, False) 
            prev_qc = temp_qc
        pl_list.append(pl)
        pr_list.append(pr)
        sign_list.append(pauli_list[i][0][:2])
        pcs_qc_list.append(save_qc)

    
    qc.compose(pcs_qc_list[-1], qubits=[i for i in range(0, total_qubits)], inplace=True)
    
    qc.measure_all()
    return sign_list, qc


# In[ ]:


num_qubits = 8
num_checks = 4
C_list = []
for i in range(total_trials):
    # Clifford = random_clifford(4)
    Clifford = random_clifford(num_qubits)
    C_list.append(Clifford)
circs_list = []
signs_list = []
for check_id in range(1, num_checks + 1):
    circs = []
    signs = []
    for i in range(total_trials):
        print(check_id, i)
        sign, circuit = hydrogen_shadow_PCS_circuit(C_list[i], num_qubits, check_id)
        signs.append(sign)
        circs.append(circuit)
    circs_list.append(circs)
    signs_list.append(signs)
    
orign_circs = []
for i in range(total_trials):
    circuit = hydrogen_shadow_circuit(C_list[i], num_qubits)
    orign_circs.append(circuit)


# In[ ]:


circs_list[0][-1].draw()


# In[ ]:


# import pickle

# # Save circs_list and origin_circs using pickle
# with open('circs_list.pkl', 'wb') as f:
#     pickle.dump(circs_list, f)

# with open('origin_circs.pkl', 'wb') as f:
#     pickle.dump(orign_circs, f)
    
    


# In[ ]:


# import pickle

# with open('circs_list.pkl', 'rb') as f:
#     circs_list = pickle.load(f)

# with open('origin_circs.pkl', 'rb') as f:
#     orign_circs = pickle.load(f)


# In[ ]:


print(circs_list[0][-2])


# In[ ]:


print(orign_circs[0])


# In[ ]:


def filter_results_reindex(dictionary, qubits, indexes, sign_list):
    new_dict = {}
    for key in dictionary.keys():
        new_key = ''
        for i in range(len(key)):
            meas_index = i
#             if i < len(sign_list):
#                 print(key, "index", i, key[i], sign_list[meas_index])
            if meas_index in indexes and key[i] == sign_list[meas_index]:
                #the key equals the sign, keep
                new_key = ''
                break
            if meas_index not in indexes:
                new_key += key[i]
        if new_key != '':
            new_dict[new_key] = dictionary[key]
    return new_dict


# In[ ]:


num_qubits = 8


# In[ ]:


b_lists_filtered = []
check_id = 1
# Submit hardware jobs via Qiskit Runtime;

with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)

    # same as the calibration process
    job = sampler.run(circs_list[check_id-1], shots=100, initial_layout=[])
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")

    result = job.result()

    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()

b_lists_check = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits + check_id]: result.quasi_dists[i].binary_probabilities().get(key)})
    b_lists_check.append(di)


filtered_b_lists = []
for i in range(total_trials):
    bit_list = ['1' if i == '+1' else '0' for i in signs_list[check_id-1][i][check_id - 1::-1]]
#     print(bit_list)
    filted_dist = filter_results_reindex(b_lists_check[i], num_qubits, [j for j in range(0, check_id)], bit_list)
    print(total_counts(filted_dist))
    filtered_b_lists.append(filted_dist)
b_lists_filtered.append(filtered_b_lists)


# In[ ]:


check_id = 2
# Submit hardware jobs via Qiskit Runtime;

with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)

    # same as the calibration process
    job = sampler.run(circs_list[check_id-1], shots=100, initial_layout=[])
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")

    result = job.result()

    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()

b_lists_check = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits + check_id]: result.quasi_dists[i].binary_probabilities().get(key)})
    b_lists_check.append(di)


filtered_b_lists = []
for i in range(total_trials):
    bit_list = ['1' if i == '+1' else '0' for i in signs_list[check_id-1][i][check_id - 1::-1]]
#     print(bit_list)
    filted_dist = filter_results_reindex(b_lists_check[i], num_qubits, [j for j in range(0, check_id)], bit_list)
    print(total_counts(filted_dist))
    filtered_b_lists.append(filted_dist)
b_lists_filtered.append(filtered_b_lists)


# In[ ]:


check_id = 3
# Submit hardware jobs via Qiskit Runtime;

with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)

    # same as the calibration process
    job = sampler.run(circs_list[check_id-1], shots=100, initial_layout=[])
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")

    result = job.result()

    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()

b_lists_check = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits + check_id]: result.quasi_dists[i].binary_probabilities().get(key)})
    b_lists_check.append(di)


filtered_b_lists = []
for i in range(total_trials):
    bit_list = ['1' if i == '+1' else '0' for i in signs_list[check_id-1][i][check_id - 1::-1]]
#     print(bit_list)
    filted_dist = filter_results_reindex(b_lists_check[i], num_qubits, [j for j in range(0, check_id)], bit_list)
    print(total_counts(filted_dist))
    filtered_b_lists.append(filted_dist)
b_lists_filtered.append(filtered_b_lists)


# In[ ]:


check_id = 4
# Submit hardware jobs via Qiskit Runtime;

with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)

    # same as the calibration process
    job = sampler.run(circs_list[check_id-1], shots=1024, initial_layout=[])
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")

    result = job.result()

    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()

b_lists_check = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits + check_id]: result.quasi_dists[i].binary_probabilities().get(key)})
    b_lists_check.append(di)


filtered_b_lists = []
for i in range(total_trials):
    bit_list = ['1' if i == '+1' else '0' for i in signs_list[check_id-1][i][check_id - 1::-1]]
#     print(bit_list)
    filted_dist = filter_results_reindex(b_lists_check[i], num_qubits, [j for j in range(0, check_id)], bit_list)
    print(total_counts(filted_dist))
    filtered_b_lists.append(filted_dist)
b_lists_filtered.append(filtered_b_lists)


# In[ ]:


# Submit hardware jobs via Qiskit Runtime;

with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)
    
    # same as the calibration process
    job = sampler.run(orign_circs, shots=100, initial_layout=[i for i in range(0, num_qubits)])
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")
    
    result = job.result()
    
    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()
    
b_lists = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits + num_checks]: result.quasi_dists[i].binary_probabilities().get(key)})
    b_lists.append(di)


# Noiseless Experiments on qiskitruntime

# In[ ]:


from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options

options = Options(optimization_level=2, resilience_level=1)
# backend = service.get_backend("ibmq_qasm_simulator")


# In[ ]:


with Session(backend=backend) as session:
    sampler = Sampler(session=session, options=options)

    job = sampler.run(orign_circs, shots=100)
    print(f"Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")
    
    result = job.result()
    
    # Close the session only if all jobs are finished
    # and you don't need to run more in the session.
    session.close()


# In[ ]:


import math
def filter_results(dictionary, qubits, indexes, sign_list):
    new_dict = {}
    for key in dictionary.keys():
        new_key = ''
        for i in range(len(key)):
            meas_index = i
            #print(key, meas_index, indexes)
            if meas_index in indexes and key[i] == sign_list[meas_index]:
                new_key = ''
                break
            if meas_index not in indexes:
                new_key += key[i]
        if new_key != '':
            new_dict[new_key] = dictionary[key]
    return new_dict


# In[ ]:


b_lists_noiseless = []

for i in range(total_trials):
    di = {}
    for key in list(result.quasi_dists[i].binary_probabilities().keys()):
        di.update({key[:num_qubits]: result.quasi_dists[i].binary_probabilities().get(key)})
    b_lists_noiseless.append(di)


# In[ ]:


def compute_expectation(b_lists, b_lists_checks, b_lists_noiseless, C_list, operator_list, num_qubits, f_tilde):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.
    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.
        operator (np.ndarray):
        num_qubits
    Returns:
        Numpy array with the reconstructed quantum state.
    """
    num_snapshots = len(b_lists)
    
    # Averaging over snapshot states.
    expectation_list = np.zeros(len(operator_list))
    expectation_list_r = np.zeros(len(operator_list))
    expectation_list_checks = [np.zeros(len(operator_list)) for i in range(len(b_lists_checks))]
    expectation_list_noiseless = np.zeros(len(operator_list))
    
    for i in range(num_snapshots):
        noisy, robust = expectation_snapshot(b_lists[i], C_list[i], operator_list, num_qubits, f_tilde)
        expectation_list += noisy
        expectation_list_r += robust
        
        for j in range(len(b_lists_checks)):
            check = expectation_snapshot_noiseless(b_lists_checks[j][i], C_list[i], operator_list, num_qubits)
            expectation_list_checks[j] += check 
            
        
        noiseless = expectation_snapshot_noiseless(b_lists_noiseless[i], C_list[i], operator_list, num_qubits)
        expectation_list_noiseless += noiseless
        
    expectation_list /= num_snapshots
    expectation_list_r /= num_snapshots
    for j in range(len(b_lists_checks)):
        expectation_list_checks[j] /= num_snapshots
    expectation_list_noiseless /= num_snapshots
    
    return expectation_list, expectation_list_r, expectation_list_checks, expectation_list_noiseless


def expectation_snapshot(b_dict, clifford, operator_list, num_qubits, f_tilde):
    """
    Helper function for `shadow_state_reconstruction` that reconstructs the overlap estimate from
    a single snapshot in a shadow. Implements Eq. (S23) from https://arxiv.org/pdf/2106.16235.pdf
    Args:
        b_dict (dict): The list of classical outcomes for the snapshot.
        clifford: Indices for the applied Pauli measurement.
        operator:
        num_qubits:
    Returns:
        Numpy array with the reconstructed snapshot.
    """
    f = 1/(2**num_qubits+1)
    # reconstructing the snapshot state from random Clifford measurements
    U = clifford.to_matrix()
    I = np.eye(2**num_qubits)
    
    # applying Eq. (S32), note that this expression is built upon random Clifford, so that inverting
    # the quantum channel follows Eq. (S29).
    snapshot_list = np.zeros(len(operator_list))
    snapshot_list_r = np.zeros(len(operator_list))
    denom = 0
    for b_state in list(b_dict.keys()):
        matrix_part = U.conj().T @ state_reconstruction(b_state) @ U
        
        interm = 1 / f * matrix_part
        interm -= (1 / f - 1)/2**num_qubits * I
        
        interm_r = 1 / f_tilde * matrix_part
        interm_r -= (1 / f_tilde - 1) / 2**num_qubits * I
        
        for index, operator in enumerate(operator_list):
            operator_matrix = operator.to_matrix()
            snapshot_list[index] += np.real(np.trace(operator_matrix @ interm) * b_dict.get(b_state))
            snapshot_list_r[index] += np.real(np.trace(operator_matrix @ interm_r) * b_dict.get(b_state))
            
        denom += b_dict.get(b_state)
    
    return snapshot_list / denom, snapshot_list_r / denom


def expectation_snapshot_noiseless(b_dict, clifford, operator_list, num_qubits):
    """
    Helper function for `shadow_state_reconstruction` that reconstructs the overlap estimate from
    a single snapshot in a shadow. Implements Eq. (S23) from https://arxiv.org/pdf/2106.16235.pdf
    Args:
        b_dict (dict): The list of classical outcomes for the snapshot.
        clifford: Indices for the applied Pauli measurement.
        operator:
        num_qubits:
    Returns:
        Numpy array with the reconstructed snapshot.
    """
    f = 1/(2**num_qubits+1)
    # reconstructing the snapshot state from random Clifford measurements
    U = clifford.to_matrix()
    I = np.eye(2**num_qubits)
    
    # applying Eq. (S32), note that this expression is built upon random Clifford, so that inverting
    # the quantum channel follows Eq. (S29).
    snapshot_list = np.zeros(len(operator_list))
    denom = 0
    for b_state in list(b_dict.keys()):
        interm = 1/f * U.conj().T @ state_reconstruction(b_state) @ U
        interm -= (1/f - 1)/2**num_qubits * I
        
        for index, operator in enumerate(operator_list):
            operator_matrix = operator.to_matrix()
            snapshot_list[index] += np.real(np.trace(operator_matrix @ interm) * b_dict.get(b_state))
            
        denom += b_dict.get(b_state)
        
    # print('denom =', denom)
    return snapshot_list / denom


# In[ ]:


# run the classical shadows postprocessing to get expectation values;

Paulis = ['XXXXXXXX', 'YYYYYYYY', 'XYXYXYXY', 'YXYXYXYX', 'YYYYXXXX', 
            'XXXXYYYY', 'ZZZZZZZZ', 'ZZZZIIII', 'IIIIZZZZ', 'ZZZZXXXX', 'XXXXZZZZ', 'ZXZXZXZX', 'XZXZXZXZ',
             'XXXXIIII', 'IIIIXXXX', 'XXIIXXII']

operator_list = []
for pauli in Paulis:
    operator_list.append(Pauli(pauli))

psi = Statevector(hf_circ(num_qubits, num_checks=0))
ref_list = []
for operator in operator_list:
    print(operator)
    expect = np.array(psi).T.conj() @ operator.to_matrix() @ np.array(psi)
    ref_list.append(expect)


# In[ ]:


num_of_runs = 20
shadow_range = [100, 400, 1000, 4000, 10000]
num_of_checks = 4

expectation_shadow = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
expectation_shadow_r = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
expectation_shadow_check1 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
expectation_shadow_check2 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
expectation_shadow_check3 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
expectation_shadow_check4 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
expectation_shadow_noiseless = np.zeros((len(shadow_range), len(Paulis), num_of_runs))

for j, num_snapshots in enumerate(shadow_range):
    print('num snapshots = ', num_snapshots)
    indices = random.sample(range(total_trials), num_snapshots)

    # Partition indices into 'num_of_runs' equally sized chunks
    partitions = np.array_split(indices, num_of_runs)
    print('paritions', partitions)
    print(len(partitions))
        
    for i, run_indices in enumerate(partitions):
        C_sublist = [C_list[k] for k in run_indices]
        b_sublists = [b_lists[k] for k in run_indices]
        b_sublists_check1 = [b_lists_filtered[0][k] for k in run_indices]
        b_sublists_check2 = [b_lists_filtered[1][k] for k in run_indices]
        b_sublists_check3 = [b_lists_filtered[2][k] for k in run_indices]
        b_sublists_check4 = [b_lists_filtered[3][k] for k in run_indices]
        b_sublists_checks = [b_sublists_check1,b_sublists_check2, b_sublists_check3, b_sublists_check4]
        b_sublists_noiseless = [b_lists_noiseless[k] for k in run_indices]

        expectation_list, expectation_list_r, expectation_list_checks, expectation_list_noiseless = compute_expectation(
            b_sublists, b_sublists_checks,  b_sublists_noiseless, C_sublist, operator_list, num_qubits, f_tilde
        )

        expectation_shadow[j, :, i] = np.real(expectation_list)
        expectation_shadow_r[j, :, i] = np.real(expectation_list_r)
        expectation_shadow_check1[j, :, i] = np.real(expectation_list_checks[0])
        expectation_shadow_check2[j, :, i] = np.real(expectation_list_checks[1])
        expectation_shadow_check3[j, :, i] = np.real(expectation_list_checks[2])
        expectation_shadow_check4[j, :, i] = np.real(expectation_list_checks[3])
        expectation_shadow_noiseless[j, :, i] = np.real(expectation_list_noiseless)


# In[ ]:


# # num_of_runs = 10
# num_of_runs = 10
# # shadow_range = [5, 10, 40, 100]#, 400]#, 1000]#, 4000]
# shadow_range = [100, 400, 800]
# expectation_shadow = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# expectation_shadow_r = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# expectation_shadow_check1 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# expectation_shadow_check2 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# expectation_shadow_check3 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# expectation_shadow_check4 = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# expectation_shadow_noiseless = np.zeros((len(shadow_range), len(Paulis), num_of_runs))
# for i in range(num_of_runs):
#     for j, num_snapshots in enumerate(shadow_range):
#         print(f'run # {i}, num snapshots = {num_snapshots}')
#         indices = random.sample(range(total_trials), num_snapshots)
#         C_sublist = [C_list[k] for k in indices]
#         b_sublists = [b_lists[k] for k in indices]
#         b_sublists_check1 = [b_lists_filtered[0][k] for k in indices]
#         b_sublists_check2 = [b_lists_filtered[1][k] for k in indices]
#         b_sublists_check3 = [b_lists_filtered[2][k] for k in indices]
#         b_sublists_check4 = [b_lists_filtered[3][k] for k in indices]
#         b_sublists_checks = [b_sublists_check1, b_sublists_check2, b_sublists_check3, b_sublists_check4]
#         b_sublists_noiseless = [b_lists_noiseless[k] for k in indices]
        
#         expectation_list, expectation_list_r, expectation_list_checks, expectation_list_noiseless = compute_expectation(
#             b_sublists, b_sublists_checks,  b_sublists_noiseless, C_sublist, operator_list, num_qubits, f_tilde
#         )
        
#         print('expectation_list', expectation_list)
#         print('expectation_list_r', expectation_list_r)
#         print('expectation_list_noiseless', expectation_list_noiseless)
        
#         expectation_shadow[j, :, i] = np.real(expectation_list)
#         expectation_shadow_r[j, :, i] = np.real(expectation_list_r)
#         expectation_shadow_check1[j, :, i] = np.real(expectation_list_checks[0])
#         expectation_shadow_check2[j, :, i] = np.real(expectation_list_checks[1])
#         expectation_shadow_check3[j, :, i] = np.real(expectation_list_checks[2])
#         expectation_shadow_check4[j, :, i] = np.real(expectation_list_checks[3])
#         expectation_shadow_noiseless[j, :, i] = np.real(expectation_list_noiseless)


# In[ ]:


print(ref_list)


# #### Calculate extrapolated checks

# In[ ]:


from numpy.polynomial.polynomial import Polynomial

medians = [np.median(check, axis=2) for check in [expectation_shadow_check1, expectation_shadow_check2, expectation_shadow_check3, expectation_shadow_check4]]
check_numbers = [1, 2, 3, 4]  # Original check layers
extrapolation_layers = [8]  # Layers you want to extrapolate to

expectation_check_limit = np.zeros((len(extrapolation_layers), len(shadow_range), len(Paulis)))

for layer_index, layer in enumerate(extrapolation_layers):
    for shadow_size_index in range(len(medians[0])):
        for pauli_index in range(medians[0].shape[1]):
            expectation_values = [median[shadow_size_index, pauli_index] for median in medians]
            polynomial = Polynomial.fit(check_numbers, expectation_values, 1)
            extrapolated_value = polynomial(layer) # Extrapolate the value for the current layer
            expectation_check_limit[layer_index, shadow_size_index, pauli_index] = extrapolated_value


# In[ ]:


print(expectation_check_limit.shape)


# In[ ]:


print(ref_list)


# In[ ]:


error = np.zeros(len(shadow_range))
error_r = np.zeros(len(shadow_range))
error_check1 = np.zeros(len(shadow_range))
error_check2 = np.zeros(len(shadow_range))
error_check3 = np.zeros(len(shadow_range))
error_check4 = np.zeros(len(shadow_range))
print(error_check4.shape)
print(expectation_shadow_check4.shape)
error_noiseless = np.zeros(len(shadow_range))

for i in range(len(shadow_range)):
    error[i] = np.mean([np.abs(np.median(expectation_shadow[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])
    error_r[i] = np.mean([np.abs(np.median(expectation_shadow_r[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])
    error_check1[i] = np.mean([np.abs(np.median(expectation_shadow_check1[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])
    error_check2[i] = np.mean([np.abs(np.median(expectation_shadow_check2[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])
    error_check3[i] = np.mean([np.abs(np.median(expectation_shadow_check3[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])
    error_check4[i] = np.mean([np.abs(np.median(expectation_shadow_check4[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])
    error_noiseless[i] = np.mean([np.abs(np.median(expectation_shadow_noiseless[i], axis=1)[j] - ref_list[j]) for j in range(len(ref_list))])


# In[ ]:


error_check_limit = np.zeros((len(extrapolation_layers), len(shadow_range)))

for layer_index, layer in enumerate(extrapolation_layers):
    for shadow_size_index in range(len(shadow_range)):
        # Calculate the mean error for this layer and shadow size across all Pauli indices
        error_check_limit[layer_index, shadow_size_index] = np.mean(
            [np.abs(expectation_check_limit[layer_index, shadow_size_index, pauli_index] - ref_list[pauli_index]) for pauli_index in range(len(Paulis))]
        )


# In[ ]:


print(error_check_limit)


# In[ ]:


print(error)


# In[ ]:


print(error_check1)


# In[ ]:


print(error_check3)


# In[ ]:


plt.figure(figsize=(5, 4), dpi=100)
plt.plot(shadow_range, error, '--o', ms=8, color='tab:orange', label='noisy')
plt.plot(shadow_range, error_check1, '--o', ms=8, color='tab:red', label='check1')
plt.plot(shadow_range, error_check2, '--o', ms=8, color='tab:purple', label='check2')
plt.plot(shadow_range, error_check3, '--o', ms=8, color='tab:olive', label='check3')
plt.plot(shadow_range, error_check4, '--o', ms=8, color='tab:pink', label='check4')
plt.plot(shadow_range, error_r, '--^', ms=8, color='tab:green', label='robust')
plt.plot(shadow_range, error_noiseless, '--x', ms=8, color='tab:blue', label='noiseless')

# Plotting each layer of extrapolated checks
colors = ['tab:brown', 'tab:gray', 'tab:cyan', 'tab:pink', 'tab:purple']
for layer_index, layer in enumerate(extrapolation_layers):
    plt.plot(shadow_range, error_check_limit[layer_index, :], '--o', ms=8, color=colors[layer_index % len(colors)], label=f'check {layer} (extrap)')

# Adjust the legend to be outside without altering the figure size
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xlabel('Shadow size', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=14)

# Note: The figure's layout isn't altered with plt.tight_layout() in this case
# Saving the figure with bbox_inches='tight' includes the external legend
plt.savefig('8-qubit_h2o_clifford_gaussian_noise.png', dpi=100, bbox_inches="tight")
plt.show()


# In[ ]:


plt.figure(figsize=(5, 4), dpi=100)
plt.plot(shadow_range, error, '--o', ms=8, color='tab:blue', label='Robust Shadow')
plt.legend(fontsize=14, loc='best')
plt.xlabel('Shadow size', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




