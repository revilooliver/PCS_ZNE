"""
Error mitigation utilities for ZNE and PCE.
"""

from typing import List, Callable, Tuple
import numpy as np
from functools import partial
from mitiq import zne
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# def mitigate_zne(
#     circuit,
#     backend,
#     pauli_string: str,
#     shots: int,
#     method: str,
#     scale_factors: List[float],
#     fold_method: Callable,
#     layout=None,
#     enable_error_mitigation: bool = False,
#     mthree=None
# ) -> float:
#     """
#     Apply Zero Noise Extrapolation to mitigate errors.
    
#     Args:
#         circuit: Quantum circuit to mitigate
#         backend: Backend to execute on
#         pauli_string: Observable Pauli string
#         shots: Number of shots per scale factor
#         method: ZNE method ("linear", "richardson", "exp", "poly")
#         scale_factors: List of noise scaling factors
#         fold_method: Noise scaling function (e.g., zne.scaling.fold_global)
#         layout: Qubit layout
#         enable_error_mitigation: Enable additional error mitigation
#         mthree: M3 mitigation object
        
#     Returns:
#         Extrapolated expectation value
#     """
#     from .quantum_execution import execute_circuit, apply_pauli_basis, compute_z_expectation
    
#     # Create executor for ZNE
#     def zne_executor(circ):
#         # Apply basis rotations for measurement
#         rotated_circ = apply_pauli_basis(circ, pauli_string)
#         rotated_circ.measure_all()
        
#         # Execute circuit
#         counts = execute_circuit(
#             rotated_circ, backend, shots,
#             layout=layout,
#             twirling=enable_error_mitigation,
#             mthree=mthree
#         )
        
#         # Compute expectation value
#         return compute_z_expectation(counts)
    
#     # Select ZNE factory
#     if method == "richardson":
#         factory = zne.inference.RichardsonFactory(scale_factors=scale_factors)
#     elif method == "linear":
#         factory = zne.inference.LinearFactory(scale_factors=scale_factors)
#     elif method == "exp":
#         factory = zne.inference.ExpFactory(scale_factors=scale_factors, asymptote=0.5)
#     elif method == "poly":
#         factory = zne.inference.PolyFactory(scale_factors=scale_factors, order=2)
#     else:
#         raise ValueError(f"Unsupported ZNE method: {method}")
    
#     # Execute ZNE
#     mitigated_value = zne.execute_with_zne(circuit, zne_executor, factory=factory, scale_noise=fold_method)
    
#     print("ZNE expectation values:", factory.get_expectation_values())
#     fig = factory.plot_fit()
#     plt.show()
    
#     return mitigated_value


def extrapolate_zne(
    expectation_values: List[float],
    scale_factors: List[float], 
    method: str = 'linear',
    show_plot: bool = False
) -> float:
    """
    Core ZNE extrapolation with optional plotting.
    
    Args:
        expectation_values: Measured expectation values at different scale factors
        scale_factors: Corresponding noise scale factors
        method: Extrapolation method ('linear', 'richardson', 'polynomial', 'exponential')
        show_plot: Whether to show extrapolation plot
        
    Returns:
        Extrapolated zero-noise value
    """
    if method == 'linear':
        factory = zne.inference.LinearFactory(scale_factors=scale_factors)
    elif method == 'richardson':
        factory = zne.inference.RichardsonFactory(scale_factors=scale_factors)
    elif method == 'polynomial':
        factory = zne.inference.PolyFactory(scale_factors=scale_factors, order=2)
    elif method == 'exponential':
        factory = zne.inference.ExpFactory(scale_factors=scale_factors)
    else:
        raise ValueError(f"Unknown ZNE method: {method}")
    
    extrapolated_value = factory.extrapolate(scale_factors, expectation_values)
    
    if show_plot:
        try:
            # Use run_classical to populate factory's internal state for plotting
            factory.run_classical(lambda x: np.interp(x, scale_factors, expectation_values))
            
            # Now reduce() to get the extrapolation curve
            factory.reduce()
            
            # Plot the fit
            fig = factory.plot_fit()
            plt.title(f'ZNE Extrapolation ({method.title()}) → {extrapolated_value:.4f}')
            plt.show()
        except Exception as e:
            print(f"Could not create plot: {e}")
    
    return extrapolated_value


def extrapolate_checks(
    num_checks_to_fit: int,
    extrap_checks: List[int],
    expectation_values: List[float],
    method: str = 'linear',
    show_plot: bool = True
) -> Tuple[List[float], Callable]:
    """
    Extrapolate PCS results to higher check numbers.
    
    Args:
        num_checks_to_fit: Number of data points to use for fitting
        extrap_checks: Check numbers to extrapolate to
        expectation_values: Measured expectation values
        method: Extrapolation method ("linear" or "exponential")
        show_plot: Whether to show fit plot
        
    Returns:
        (extrapolated_values, fit_function)
    """
    # Prepare fitting data
    check_numbers = np.array(range(1, num_checks_to_fit + 1))
    y_data = np.array(expectation_values[:num_checks_to_fit])
    
    # Fit model
    if method == 'linear':
        fit_func, params = _fit_linear_model(check_numbers, y_data)
        if show_plot:
            _plot_linear_fit(check_numbers, y_data, fit_func, params, extrap_checks)
    elif method == 'exponential':
        fit_func, params = _fit_exponential_model(check_numbers, y_data)
        if show_plot:
            _plot_exponential_fit(check_numbers, y_data, fit_func, params, extrap_checks)
    else:
        raise ValueError("Method must be 'linear' or 'exponential'")
    
    # Extrapolate to target check numbers
    extrapolated_values = [fit_func(c) for c in extrap_checks]
    
    return extrapolated_values, fit_func


def _fit_linear_model(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Callable, np.ndarray]:
    """Fit linear model to data."""
    coeffs = np.polyfit(x_data, y_data, 1)
    fit_func = np.poly1d(coeffs)
    return fit_func, coeffs


def _fit_exponential_model(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Callable, np.ndarray]:
    """Fit exponential model E(m) = a * b^m + c to data."""
    def exp_model(m, a, b, c):
        return a * (b ** m) + c
    
    initial_guess = [1.0, 0.9, 0.0]
    lower_bounds = [-np.inf, 0.6, -np.inf]
    upper_bounds = [np.inf, 1.2, np.inf]
    
    try:
        popt, _ = curve_fit(
            exp_model, x_data, y_data, 
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds), 
            maxfev=10000
        )
        fit_func = lambda m: exp_model(m, *popt)
        return fit_func, popt
    except Exception as e:
        raise RuntimeError(f"PCE exponential fitting failed: {e}") from e


def _plot_linear_fit(x_data, y_data, fit_func, coeffs, extrap_points):
    """Plot linear fit visualization."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, label='Data Points', s=50)
    
    x_fit = np.linspace(x_data.min(), max(extrap_points), 100)
    plt.plot(x_fit, fit_func(x_fit), 
             label=f'Linear fit: slope={coeffs[0]:.3f}', color='red')
    
    plt.xlabel('Number of Checks')
    plt.ylabel('Expectation Value')
    plt.title('PCE Linear Extrapolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def _plot_exponential_fit(x_data, y_data, fit_func, popt, extrap_points):
    """Plot exponential fit visualization."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, label='Data Points', s=50)
    
    x_fit = np.linspace(x_data.min(), max(extrap_points), 100)
    plt.plot(x_fit, fit_func(x_fit), 
             label=f'Exp fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}',
             color='red')
    
    plt.xlabel('Number of Checks')
    plt.ylabel('Expectation Value')
    plt.title('PCE Exponential Extrapolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()