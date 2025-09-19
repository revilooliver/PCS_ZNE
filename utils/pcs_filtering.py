"""
Simple PCS post-selection filtering utilities.
"""

from typing import Dict, List
from copy import deepcopy


def filter_counts(no_checks: int, sign_list_in: list, in_counts: dict) -> dict:
    """
    Adjusts for minus signs.
    
    Args:
        no_checks: Number of check qubits
        sign_list_in: List of signs for checks (e.g., ["+1", "+1", "-1"])
        in_counts: Input measurement counts
        
    Returns:
        Filtered counts with check qubits removed
    """
    sign_list = deepcopy(sign_list_in)
    sign_list.reverse()
    err_free_checks = ""
    for i in sign_list:
        if i == "+1":
            err_free_checks += "0"
        else:
            err_free_checks += "1"
            
    out_counts = {}
    for key in in_counts.keys():
        if err_free_checks == key[:no_checks]:
            new_key = key[no_checks:]
            out_counts[new_key] = in_counts[key]
    return out_counts


def compute_post_selection_rate(
    raw_counts: Dict[str, int], 
    filtered_counts: Dict[str, int]
) -> float:
    """
    Compute post-selection rate from raw and filtered counts.
    
    Args:
        raw_counts: Counts before filtering
        filtered_counts: Counts after filtering
        
    Returns:
        Post-selection rate (fraction of shots kept)
    """
    total_raw = sum(raw_counts.values())
    total_filtered = sum(filtered_counts.values())
    
    return total_filtered / total_raw if total_raw > 0 else 0.0