"""
Plotting utilities for visualizing experimental results and comparisons.

This module provides functions for creating publication-quality plots comparing
different error mitigation methods, including PCE, ZNE, and baseline measurements.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple


def plot_avg_errors_by_qubit(
    data: Dict[int, Dict[str, float]], 
    num_samples: Optional[int] = None, 
    num_circs: Optional[int] = None, 
    depth: Optional[int] = None, 
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_values: bool = True
) -> None:
    """
    Plot average errors for multiple methods grouped by number of qubits.
    
    Args:
        data: Dictionary mapping num_qubits to method errors
        num_samples: Number of samples, used in the plot title
        num_circs: Number of circuits, used in the plot title
        depth: Circuit depth, used in the plot title
        save_path: Path to save the resulting plot
        figsize: Figure size as (width, height)
        show_values: Whether to show values on top of bars
    """
    # Determine all methods present across the data
    methods = set()
    for errors in data.values():
        methods.update(errors.keys())
    methods = sorted(methods)
    num_methods = len(methods)
    
    # Get a sorted list of qubit counts
    qubit_list = sorted(data.keys())
    x = np.arange(len(qubit_list))
    width = 0.8 / num_methods  # Adjust width for grouped bars
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, method in enumerate(methods):
        # Extract error values for each qubit count for the given method
        method_errors = [data[q].get(method, None) for q in qubit_list]
        # Bar positions for this method
        bar_positions = x - 0.4 + i * width + width / 2
        bars = ax.bar(bar_positions, method_errors, width, label=method)
        
        # Add value labels on top of each bar
        if show_values:
            for bar in bars:
                height = bar.get_height()
                if height is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, 
                        height + 0.002,
                        f'{height:.3f}', 
                        ha='center', 
                        va='bottom', 
                        fontsize=8
                    )
    
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Average Absolute Error")
    
    # Build title from provided parameters
    title_parts = []
    if num_samples is not None:
        title_parts.append(f"Total # of samples = {num_samples}")
    if num_circs is not None:
        title_parts.append(f"# of circuits = {num_circs}")
    if depth is not None:
        title_parts.append(f"Depth = {depth}")
    title_str = " | ".join(title_parts) if title_parts else "Comparison to ZNE"
    ax.set_title(title_str)
    
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_list)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_avg_errors_by_depth(
    data: Dict[int, Dict[str, float]], 
    num_samples: Optional[int] = None, 
    num_circs: Optional[int] = None, 
    qubits: Optional[int] = None, 
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_values: bool = True
) -> None:
    """
    Plot average errors for multiple methods grouped by circuit depth.
    
    Args:
        data: Dictionary mapping depth to method errors
        num_samples: Number of samples, used in the plot title
        num_circs: Number of circuits, used in the plot title
        qubits: Number of qubits (fixed), used in the plot title
        save_path: Path to save the resulting plot
        figsize: Figure size as (width, height)
        show_values: Whether to show values on top of bars
    """
    # Determine all methods present across the data
    methods = set()
    for errors in data.values():
        methods.update(errors.keys())
    methods = sorted(methods)
    num_methods = len(methods)
    
    # Get a sorted list of circuit depths
    depths = sorted(data.keys())
    x = np.arange(len(depths))
    width = 0.8 / num_methods  # Adjust width for grouped bars
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, method in enumerate(methods):
        # Extract error values for each depth for the given method
        method_errors = [data[d].get(method, None) for d in depths]
        # Bar positions for this method
        bar_positions = x - 0.4 + i * width + width / 2
        bars = ax.bar(bar_positions, method_errors, width, label=method)
        
        # Add value labels on top of each bar
        if show_values:
            for bar in bars:
                height = bar.get_height()
                if height is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, 
                        height + 0.002,
                        f'{height:.3f}', 
                        ha='center', 
                        va='bottom', 
                        fontsize=8
                    )
    
    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Average Absolute Error")
    
    # Build title from provided parameters
    title_parts = []
    if num_samples is not None:
        title_parts.append(f"# of samples = {num_samples}")
    if num_circs is not None:
        title_parts.append(f"# of circuits = {num_circs}")
    if qubits is not None:
        title_parts.append(f"# of qubits = {qubits}")
    title_str = " | ".join(title_parts) if title_parts else "Comparison of Error vs Depth"
    ax.set_title(title_str)
    
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_error_comparison_line(
    x_values: List[Union[int, float]],
    error_data: Dict[str, List[float]],
    x_label: str = "Parameter",
    y_label: str = "Average Absolute Error",
    title: str = "Error Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    log_scale: bool = False
) -> None:
    """
    Plot line comparison of errors across different parameter values.
    
    Args:
        x_values: List of x-axis values (e.g., number of qubits, depth, etc.)
        error_data: Dictionary mapping method names to lists of error values
        x_label: Label for x-axis
        y_label: Label for y-axis  
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size as (width, height)
        log_scale: Whether to use logarithmic scale for y-axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for method, errors in error_data.items():
        ax.plot(x_values, errors, marker='o', label=method, linewidth=2)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_extrapolation_fit(
    check_numbers: List[int],
    expectation_values: List[float],
    fit_function: callable,
    extrap_points: Optional[List[int]] = None,
    method_name: str = "Extrapolation",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot the extrapolation fit for PCE method.
    
    Args:
        check_numbers: List of check numbers used for fitting
        expectation_values: Corresponding expectation values
        fit_function: Fitted function for extrapolation
        extrap_points: Additional points to show extrapolation
        method_name: Name of the extrapolation method
        save_path: Path to save the plot
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data points
    ax.scatter(check_numbers, expectation_values, color='blue', s=50, 
               label='Data Points', zorder=3)
    
    # Create smooth curve for fit
    if extrap_points:
        x_range = np.linspace(min(check_numbers), max(extrap_points), 100)
    else:
        x_range = np.linspace(min(check_numbers), max(check_numbers) * 1.2, 100)
    
    y_fit = [fit_function(x) for x in x_range]
    ax.plot(x_range, y_fit, color='red', linewidth=2, 
            label=f'{method_name} Fit', zorder=2)
    
    # Plot extrapolation points if provided
    if extrap_points:
        extrap_values = [fit_function(x) for x in extrap_points]
        ax.scatter(extrap_points, extrap_values, color='green', s=50, 
                   label='Extrapolated Points', marker='s', zorder=3)
    
    ax.set_xlabel('Number of Checks')
    ax.set_ylabel('Expectation Value')
    ax.set_title(f'{method_name} Extrapolation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# PCE vs ZNE SPECIFIC PLOTTING FUNCTIONS
# ============================================================================

def load_experimental_results(results_folder: str) -> Dict:
    """
    Load all experimental results from the specified folder.
    
    Args:
        results_folder: Path to the folder containing result CSV files
        
    Returns:
        Dictionary containing loaded DataFrames and metadata
    """
    results = {}
    
    try:
        # Load PCE statistics
        pce_stats_path = os.path.join(results_folder, "pce_results_statistics.csv")
        results['pce_stats'] = pd.read_csv(pce_stats_path)
        
        # Load ZNE results 
        zne_path = os.path.join(results_folder, "zne_results_multiple_runs.csv")
        results['zne'] = pd.read_csv(zne_path)
        
        # Load baseline results
        baseline_path = os.path.join(results_folder, "baseline_results.csv")
        results['baseline'] = pd.read_csv(baseline_path)
        
        # Load backward compatible PCE results
        pce_path = os.path.join(results_folder, "pce_results.csv")
        results['pce_compat'] = pd.read_csv(pce_path)
        
        results['success'] = True
        results['error_message'] = None
        
        print("Successfully loaded all experimental results!")
        print(f"PCE stats shape: {results['pce_stats'].shape}")
        print(f"ZNE results shape: {results['zne'].shape}")
        print(f"Baseline results shape: {results['baseline'].shape}")
        
    except FileNotFoundError as e:
        results['success'] = False
        results['error_message'] = f"Could not find results files: {e}"
        print(results['error_message'])
        print("Please check the results_folder path or run the experiments first")
    
    return results


def plot_method_comparison_with_baseline(results: Dict, num_checks: int, ideal_expectation: float,
                                       save_path: Optional[str] = None) -> None:
    """
    Plot method comparison including baseline with confidence intervals.
    
    Args:
        results: Dictionary containing experimental results
        num_checks: Number of checks used in the experiment  
        ideal_expectation: Ideal expectation value
        save_path: Optional path to save the plot
    """
    if not results['success']:
        print("Cannot plot: experimental results not loaded successfully")
        return
    
    df_pce_stats = results['pce_stats']
    df_zne = results['zne']
    df_baseline = results['baseline']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Extract results for full number of checks
    pce_stats_full = df_pce_stats[df_pce_stats['num_checks_used'] == num_checks].iloc[0]
    zne_row = df_zne.iloc[0]  # Assuming first row is the method we want
    baseline_row = df_baseline.iloc[0]
    
    # Get values and confidence intervals
    baseline_result = baseline_row['baseline_expectation']
    pce_linear = pce_stats_full['linear_mean']
    pce_exponential = pce_stats_full['exp_mean']
    zne_result = zne_row['zne_expectation']
    
    # Calculate error bars from confidence intervals
    baseline_err = (baseline_row['baseline_ci_high'] - baseline_row['baseline_ci_low']) / 2 if not pd.isna(baseline_row['baseline_ci_low']) else 0
    pce_linear_err = (pce_stats_full['linear_ci_high'] - pce_stats_full['linear_ci_low']) / 2 if not pd.isna(pce_stats_full['linear_ci_low']) else 0
    pce_exp_err = (pce_stats_full['exp_ci_high'] - pce_stats_full['exp_ci_low']) / 2 if not pd.isna(pce_stats_full['exp_ci_low']) else 0
    
    # Calculate ZNE error bar from raw values if available
    zne_err = 0
    if not pd.isna(zne_row['zne_std']) and zne_row['num_successful'] > 1:
        import scipy.stats as stats
        n = zne_row['num_successful'] 
        t_val = stats.t.ppf(0.975, n-1)  # 95% CI
        zne_err = (zne_row['zne_std'] / np.sqrt(n)) * t_val
    
    # Setup bar chart with baseline
    methods = ['Baseline', 'PCE Linear', 'PCE Exponential', 'ZNE Linear']
    values = [baseline_result, pce_linear, pce_exponential, zne_result]
    error_bars = [baseline_err, pce_linear_err, pce_exp_err, zne_err]
    colors = ['orange', 'blue', 'green', 'red']
    
    # Calculate errors from ideal
    errors = [abs(val - ideal_expectation) for val in values]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, alpha=0.7, yerr=error_bars, capsize=8, 
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add ideal line
    ax.axhline(ideal_expectation, color='gray', linestyle='--', 
              label=f'Ideal ({ideal_expectation:.3f})', linewidth=2)
    
    # Annotate bars with values and errors
    for i, (bar, val, err, err_bar) in enumerate(zip(bars, values, errors, error_bars)):
        height = bar.get_height()
        text_height = height + err_bar + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, text_height,
                f'{val:.3f}\n(Δ={err:.4f})', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Expectation Value', fontsize=12)
    ax.set_title('Method Comparison with Baseline and 95% Confidence Intervals', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved method comparison plot to {save_path}")
    
    plt.show()


def plot_pce_by_num_checks_with_baseline_zne(results: Dict, ideal_expectation: float, save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive comparison: Baseline, ZNE, and PCE results for different numbers of checks.
    
    Args:
        results: Dictionary containing experimental results
        ideal_expectation: Ideal expectation value
        save_path: Optional path to save the plot
    """
    if not results['success']:
        print("Cannot plot: experimental results not loaded successfully")
        return
    
    df_pce_stats = results['pce_stats']
    df_zne = results['zne']
    df_baseline = results['baseline']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    baseline_row = df_baseline.iloc[0]
    zne_row = df_zne.iloc[0]
    
    # Build method names and values
    methods = ['Baseline', 'ZNE Linear']
    values = [baseline_row['baseline_expectation'], zne_row['zne_expectation']]
    error_bars = []
    
    # Baseline error bar
    if not pd.isna(baseline_row['baseline_ci_low']) and not pd.isna(baseline_row['baseline_ci_high']):
        baseline_err = (baseline_row['baseline_ci_high'] - baseline_row['baseline_ci_low']) / 2
    else:
        baseline_err = 0
    error_bars.append(baseline_err)
    
    # ZNE error bar
    zne_err = 0
    if not pd.isna(zne_row['zne_std']) and zne_row['num_successful'] > 1:
        import scipy.stats as stats
        n = zne_row['num_successful']
        t_val = stats.t.ppf(0.975, n-1)  # 95% CI
        zne_err = (zne_row['zne_std'] / np.sqrt(n)) * t_val
    error_bars.append(zne_err)
    
    # Add PCE results for each number of checks
    for _, row in df_pce_stats.iterrows():
        num_checks_used = int(row['num_checks_used'])
        
        # PCE Linear
        if not pd.isna(row['linear_mean']):
            methods.append(f'PCE Linear ({num_checks_used} checks)')
            values.append(row['linear_mean'])
            if not pd.isna(row['linear_ci_low']) and not pd.isna(row['linear_ci_high']):
                linear_err = (row['linear_ci_high'] - row['linear_ci_low']) / 2
            else:
                linear_err = 0
            error_bars.append(linear_err)
        
        # PCE Exponential
        if not pd.isna(row['exp_mean']):
            methods.append(f'PCE Exp ({num_checks_used} checks)')
            values.append(row['exp_mean'])
            if not pd.isna(row['exp_ci_low']) and not pd.isna(row['exp_ci_high']):
                exp_err = (row['exp_ci_high'] - row['exp_ci_low']) / 2
            else:
                exp_err = 0
            error_bars.append(exp_err)
    
    # Calculate absolute errors
    abs_errors = [abs(val - ideal_expectation) for val in values]
    
    # Create color scheme
    colors = ['orange', 'red']  # Baseline, ZNE
    num_pce_methods = len(methods) - 2
    pce_colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_pce_methods // 2))  # Linear PCE
    exp_colors = plt.cm.Greens(np.linspace(0.4, 0.9, num_pce_methods // 2))  # Exponential PCE
    
    # Interleave linear and exponential colors
    for i in range(num_pce_methods // 2):
        colors.extend([pce_colors[i], exp_colors[i]])
    
    # Left plot: Expectation values
    x = np.arange(len(methods))
    bars1 = ax1.bar(x, values, color=colors, alpha=0.7, yerr=error_bars, capsize=5,
                    error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add ideal line
    ax1.axhline(ideal_expectation, color='gray', linestyle='--', 
               label=f'Ideal ({ideal_expectation:.3f})', linewidth=2)
    
    # Annotate bars with values
    for i, (bar, val, err_bar) in enumerate(zip(bars1, values, error_bars)):
        height = bar.get_height()
        text_height = height + err_bar + 0.02
        ax1.text(bar.get_x() + bar.get_width()/2, text_height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Expectation Value')
    ax1.set_title('Comprehensive Method Comparison with 95% Confidence Intervals')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Absolute errors
    bars2 = ax2.bar(x, abs_errors, color=colors, alpha=0.7)
    
    # Annotate error bars with values
    for i, (bar, err) in enumerate(zip(bars2, abs_errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                f'{err:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Absolute Errors from Ideal Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive comparison plot to {save_path}")
    
    plt.show()


def print_performance_summary(results: Dict, num_checks: int, ideal_expectation: float) -> None:
    """
    Print a comprehensive performance summary of all methods.
    
    Args:
        results: Dictionary containing experimental results
        num_checks: Number of checks used in the experiment
        ideal_expectation: Ideal expectation value
    """
    if not results['success']:
        print("Cannot print summary: experimental results not loaded successfully")
        return
    
    df_pce_stats = results['pce_stats']
    df_zne = results['zne']
    df_baseline = results['baseline']
    
    print("=== PERFORMANCE SUMMARY ===")
    
    # Extract values
    baseline_row = df_baseline.iloc[0]
    pce_stats_full = df_pce_stats[df_pce_stats['num_checks_used'] == num_checks].iloc[0]
    zne_row = df_zne.iloc[0]
    
    baseline_result = baseline_row['baseline_expectation']
    pce_linear = pce_stats_full['linear_mean']
    pce_exponential = pce_stats_full['exp_mean']
    zne_result = zne_row['zne_expectation']
    
    # Calculate errors
    baseline_error = abs(baseline_result - ideal_expectation)
    pce_linear_error = abs(pce_linear - ideal_expectation)
    pce_exp_error = abs(pce_exponential - ideal_expectation)
    zne_error = abs(zne_result - ideal_expectation)
    
    # Print results
    print(f"Ideal expectation: {ideal_expectation:.6f}")
    print(f"\nBaseline: {baseline_result:.6f} (error: {baseline_error:.6f})")
    print(f"PCE Linear: {pce_linear:.6f} (error: {pce_linear_error:.6f})")
    print(f"PCE Exponential: {pce_exponential:.6f} (error: {pce_exp_error:.6f})")
    print(f"ZNE Linear: {zne_result:.6f} (error: {zne_error:.6f})")
    
    # Calculate improvements
    print(f"\n=== IMPROVEMENTS OVER BASELINE ===")
    methods = ["PCE Linear", "PCE Exponential", "ZNE Linear"]
    errors = [pce_linear_error, pce_exp_error, zne_error]
    
    for method, error in zip(methods, errors):
        improvement = baseline_error - error
        improvement_pct = (improvement / baseline_error) * 100 if baseline_error > 0 else 0
        print(f"{method}: {improvement:.6f} improvement ({improvement_pct:.1f}%)")
    
    # Find best method
    all_methods = ["Baseline"] + methods
    all_errors = [baseline_error] + errors
    best_idx = np.argmin(all_errors)
    print(f"\nBest performing method: {all_methods[best_idx]} (error: {all_errors[best_idx]:.6f})")


def plot_ratio_comparison_with_baseline(results: Dict, num_checks: int, save_path: Optional[str] = None) -> None:
    """
    Plot method comparison showing approximation ratios (noisy/ideal) instead of raw values.
    
    Args:
        results: Dictionary containing experimental results
        num_checks: Number of checks used in the experiment  
        save_path: Optional path to save the plot
    """
    if not results['success']:
        print("Cannot plot: experimental results not loaded successfully")
        return
    
    df_pce_stats = results['pce_stats']
    df_zne = results['zne']
    df_baseline = results['baseline']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Extract results for full number of checks
    pce_stats_full = df_pce_stats[df_pce_stats['num_checks_used'] == num_checks].iloc[0]
    zne_row = df_zne.iloc[0]  # Assuming first row is the method we want
    baseline_row = df_baseline.iloc[0]
    
    # Get ratio values
    baseline_ratio = baseline_row['baseline_ratio']
    pce_linear_ratio = pce_stats_full['linear_ratio_mean']
    pce_exponential_ratio = pce_stats_full['exp_ratio_mean']
    zne_ratio = zne_row['zne_ratio']
    
    # Calculate error bars from confidence intervals (for ratios)
    baseline_ratio_err = 0
    if not pd.isna(baseline_row['baseline_ci_low']) and not pd.isna(baseline_row['baseline_ci_high']):
        # Convert expectation CI to ratio CI
        baseline_ideal = baseline_row['ideal_expectation']
        baseline_ratio_ci_low = baseline_row['baseline_ci_low'] / baseline_ideal
        baseline_ratio_ci_high = baseline_row['baseline_ci_high'] / baseline_ideal
        baseline_ratio_err = (baseline_ratio_ci_high - baseline_ratio_ci_low) / 2
    
    pce_linear_ratio_err = (pce_stats_full['linear_ratio_ci_high'] - pce_stats_full['linear_ratio_ci_low']) / 2 if not pd.isna(pce_stats_full['linear_ratio_ci_low']) else 0
    pce_exp_ratio_err = (pce_stats_full['exp_ratio_ci_high'] - pce_stats_full['exp_ratio_ci_low']) / 2 if not pd.isna(pce_stats_full['exp_ratio_ci_low']) else 0
    
    # Calculate ZNE ratio error bar
    zne_ratio_err = 0
    if not pd.isna(zne_row['zne_std']) and zne_row['num_successful'] > 1:
        import scipy.stats as stats
        n = zne_row['num_successful'] 
        t_val = stats.t.ppf(0.975, n-1)  # 95% CI
        zne_expectation_err = (zne_row['zne_std'] / np.sqrt(n)) * t_val
        zne_ratio_err = zne_expectation_err / baseline_row['ideal_expectation']  # Convert to ratio error
    
    # Setup bar chart with baseline
    methods = ['Baseline', 'PCE Linear', 'PCE Exponential', 'ZNE Linear']
    ratio_values = [baseline_ratio, pce_linear_ratio, pce_exponential_ratio, zne_ratio]
    error_bars = [baseline_ratio_err, pce_linear_ratio_err, pce_exp_ratio_err, zne_ratio_err]
    colors = ['orange', 'blue', 'green', 'red']
    
    x = np.arange(len(methods))
    bars = ax.bar(x, ratio_values, color=colors, alpha=0.7, yerr=error_bars, capsize=8, 
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add ideal line at ratio = 1.0
    ax.axhline(1.0, color='gray', linestyle='--', 
              label='Ideal (ratio = 1.0)', linewidth=2)
    
    # Annotate bars with values
    for i, (bar, val, err_bar) in enumerate(zip(bars, ratio_values, error_bars)):
        height = bar.get_height()
        text_height = height + err_bar + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, text_height,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Approximation Ratio (Noisy/Ideal)', fontsize=12)
    ax.set_title('Method Comparison: Approximation Ratios with 95% Confidence Intervals', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ratio comparison plot to {save_path}")
    
    plt.show()


def plot_pce_ratios_by_num_checks(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot PCE approximation ratios for different numbers of checks alongside baseline and ZNE.
    
    Args:
        results: Dictionary containing experimental results
        save_path: Optional path to save the plot
    """
    if not results['success']:
        print("Cannot plot: experimental results not loaded successfully")
        return
    
    df_pce_stats = results['pce_stats']
    df_zne = results['zne']
    df_baseline = results['baseline']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data
    baseline_row = df_baseline.iloc[0]
    zne_row = df_zne.iloc[0]
    
    # Build method names and ratio values
    methods = ['Baseline', 'ZNE Linear']
    ratio_values = [baseline_row['baseline_ratio'], zne_row['zne_ratio']]
    error_bars = []
    
    # Baseline ratio error bar
    baseline_ratio_err = 0
    if not pd.isna(baseline_row['baseline_ci_low']) and not pd.isna(baseline_row['baseline_ci_high']):
        baseline_ideal = baseline_row['ideal_expectation']
        baseline_ratio_ci_low = baseline_row['baseline_ci_low'] / baseline_ideal
        baseline_ratio_ci_high = baseline_row['baseline_ci_high'] / baseline_ideal
        baseline_ratio_err = (baseline_ratio_ci_high - baseline_ratio_ci_low) / 2
    error_bars.append(baseline_ratio_err)
    
    # ZNE ratio error bar
    zne_ratio_err = 0
    if not pd.isna(zne_row['zne_std']) and zne_row['num_successful'] > 1:
        import scipy.stats as stats
        n = zne_row['num_successful']
        t_val = stats.t.ppf(0.975, n-1)  # 95% CI
        zne_expectation_err = (zne_row['zne_std'] / np.sqrt(n)) * t_val
        zne_ratio_err = zne_expectation_err / baseline_row['ideal_expectation']
    error_bars.append(zne_ratio_err)
    
    # Add PCE results for each number of checks
    for _, row in df_pce_stats.iterrows():
        num_checks_used = int(row['num_checks_used'])
        
        # PCE Linear
        if not pd.isna(row['linear_ratio_mean']):
            methods.append(f'PCE Linear ({num_checks_used} checks)')
            ratio_values.append(row['linear_ratio_mean'])
            if not pd.isna(row['linear_ratio_ci_low']) and not pd.isna(row['linear_ratio_ci_high']):
                linear_ratio_err = (row['linear_ratio_ci_high'] - row['linear_ratio_ci_low']) / 2
            else:
                linear_ratio_err = 0
            error_bars.append(linear_ratio_err)
        
        # PCE Exponential
        if not pd.isna(row['exp_ratio_mean']):
            methods.append(f'PCE Exp ({num_checks_used} checks)')
            ratio_values.append(row['exp_ratio_mean'])
            if not pd.isna(row['exp_ratio_ci_low']) and not pd.isna(row['exp_ratio_ci_high']):
                exp_ratio_err = (row['exp_ratio_ci_high'] - row['exp_ratio_ci_low']) / 2
            else:
                exp_ratio_err = 0
            error_bars.append(exp_ratio_err)
    
    # Create color scheme
    colors = ['orange', 'red']  # Baseline, ZNE
    num_pce_methods = len(methods) - 2
    pce_colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_pce_methods // 2))  # Linear PCE
    exp_colors = plt.cm.Greens(np.linspace(0.4, 0.9, num_pce_methods // 2))  # Exponential PCE
    
    # Interleave linear and exponential colors
    for i in range(num_pce_methods // 2):
        colors.extend([pce_colors[i], exp_colors[i]])
    
    # Create plot
    x = np.arange(len(methods))
    bars = ax.bar(x, ratio_values, color=colors, alpha=0.7, yerr=error_bars, capsize=5,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add ideal line
    ax.axhline(1.0, color='gray', linestyle='--', 
               label='Ideal (ratio = 1.0)', linewidth=2)
    
    # Annotate bars with values
    for i, (bar, val, err_bar) in enumerate(zip(bars, ratio_values, error_bars)):
        height = bar.get_height()
        text_height = height + err_bar + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, text_height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Approximation Ratio (Noisy/Ideal)')
    ax.set_title('Comprehensive Approximation Ratio Comparison with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive ratio comparison plot to {save_path}")
    
    plt.show()


def create_all_plots(results_folder: str, num_qubits: int, deg: int, p: int, 
                    ideal_expectation: float, num_checks: Optional[int] = None) -> None:
    """
    Create all standard plots for PCE vs ZNE analysis.
    
    Args:
        results_folder: Path to folder containing experimental results
        num_qubits: Number of qubits in the experiment
        deg: Degree of the graph
        p: QAOA depth parameter
        ideal_expectation: Ideal expectation value
        num_checks: Number of checks (defaults to num_qubits)
    """
    if num_checks is None:
        num_checks = num_qubits
    
    # Load results
    results = load_experimental_results(results_folder)
    
    if not results['success']:
        return
    
    # Create save paths
    plot_paths = {
        'comparison': os.path.join(results_folder, "method_comparison_with_baseline.png"),
        'comprehensive': os.path.join(results_folder, "comprehensive_pce_comparison.png"),
        'ratio_comparison': os.path.join(results_folder, "ratio_comparison_with_baseline.png"),
        'ratio_comprehensive': os.path.join(results_folder, "comprehensive_ratio_comparison.png"),
    }
    
    # Create plots
    print(f"Creating plots for {num_qubits}-qubit QAOA experiment...")
    
    # Original expectation value plots
    plot_method_comparison_with_baseline(results, num_checks, ideal_expectation, plot_paths['comparison'])
    plot_pce_by_num_checks_with_baseline_zne(results, ideal_expectation, plot_paths['comprehensive'])
    
    # New ratio plots
    plot_ratio_comparison_with_baseline(results, num_checks, plot_paths['ratio_comparison'])
    plot_pce_ratios_by_num_checks(results, plot_paths['ratio_comprehensive'])
    
    # Print performance summary
    print_performance_summary(results, num_checks, ideal_expectation)
    
    print(f"\n=== PLOTS SAVED ===")
    for plot_type, path in plot_paths.items():
        print(f"{plot_type.capitalize()}: {path}")