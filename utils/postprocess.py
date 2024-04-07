from utils.pauli_checks import update_cnot_dist
def singlecheck_postprocess(input_dict, right_checks, qubits, check_count, layer_index):
    output_dict = input_dict.copy()
    for index in range(0, len(right_checks)):
        check = right_checks[index]
        if check == 'Z':
            ctrl_index =  - qubits + index 
            targ_index = check_count - layer_index -1
            output_dict = update_cnot_dist(output_dict, ctrl_index, targ_index)
            #change the corresponding qubit in the distribution
    return output_dict

def rightchecks_postprocess(input_dist, qubits, check_count, pr_list):
    output_dict = input_dist
    for i in range(0, check_count):
        output_dict = singlecheck_postprocess(output_dict, right_checks = pr_list[i], qubits = qubits, check_count = check_count, layer_index = i)
    return output_dict 

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