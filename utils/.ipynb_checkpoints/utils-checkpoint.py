# author: Ji Liu (ji.liu@anl.gov)
import numpy as np

def total_counts(dictionary):
    total = 0
    for value in dictionary.values():
        total += value
    return total

def filter_results(dictionary, indexes):
    new_dict = {}
    for key in dictionary.keys():
        new_key = ''
        for i in range(len(key)):
            if i in indexes and key[i] == '1':
                new_key = ''
                break
            if i not in indexes:
                new_key += key[i]
        if new_key != '':
            new_dict[new_key] = dictionary[key]
    return new_dict

def dict_to_list(dictionary, size):
    ret_list = []
    total = total_counts(dictionary)
    format_str = '{0:0' + str(size) + 'b}'
    for i in range(0, 2 ** size):
        binary = format_str.format(i)
        try:
            ret_list.append(dictionary[binary]/total)
        except:
            ret_list.append(0.0)
    return ret_list

def norm_dict(dictionary):
    total = total_counts(dictionary)
    norm_dist = {}
    for i in dictionary.keys():
        norm_dist[i] = dictionary[i]/total
    return norm_dist

def H_distance(p, q):
    # distance between p an d
    # p and q are np array probability distributions
    n = len(p)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result

def H_distance_dict(p, q):
    # distance between p an d
    # p and q are np array probability distributions
    sum = 0.0
    for key in p.keys():
        sum += (np.sqrt(p[key]) - np.sqrt(q[key]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result

def qubits_index(temp_qc, wire_path_map):
    # return a list of renumbered qubit index in order
    index_list = [0] * num_qubits
    for i in range(0, len(wire_path_map)):
        qubit_maps = wire_path_map[temp_qc.qubits[i]]
        for mp in qubit_maps:
            if mp[1].register.__len__() > 3:
                index_list[mp[1].index] = i
    return index_list