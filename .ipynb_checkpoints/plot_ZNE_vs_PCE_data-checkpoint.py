import os
import csv
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_data_from_csvs(data_dir):
    """
    Load data from all CSV files in the specified directory.
    Returns a dictionary with number of qubits as keys and error values for each method.
    """
    data = defaultdict(dict)
    pattern = re.compile(r"avg_errors_n=(\d+)_num_circs=\d+_num_samp=\d+\.csv")
    
    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if not match:
            continue
            
        num_qubits = int(match.group(1))
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                method, error = row
                data[num_qubits][method] = float(error)
    
    return data

def create_bar_plot(data, save_path=None):
    """
    Create a bar plot comparing ZNE and PCE errors for different numbers of qubits.
    """
    # Sort by number of qubits
    num_qubits = sorted(data.keys())
    zne_errors = [data[n]['ZNE'] for n in num_qubits]
    pce_errors = [data[n]['PCE'] for n in num_qubits]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(num_qubits))
    width = 0.35
    
    # Create bars
    rects1 = ax.bar(x - width/2, zne_errors, width, label='ZNE', color='skyblue')
    rects2 = ax.bar(x + width/2, pce_errors, width, label='PCE', color='lightcoral')
    
    # Customize plot
    ax.set_ylabel('Average Absolute Error')
    ax.set_xlabel('Number of Qubits')
    ax.set_title('Comparison of ZNE and PCE Error Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(num_qubits)
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Adjust layout and save/show plot
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def create_line_plot(data, save_path=None):
    """
    Create a line plot comparing ZNE and PCE errors for different numbers of qubits.
    """
    num_qubits = sorted(data.keys())
    zne_errors = [data[n]['ZNE'] for n in num_qubits]
    pce_errors = [data[n]['PCE'] for n in num_qubits]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_qubits, zne_errors, 'o-', label='ZNE', color='skyblue')
    plt.plot(num_qubits, pce_errors, 'o-', label='PCE', color='lightcoral')
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Average Absolute Error')
    plt.title('Error Rates vs Number of Qubits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for x, y in zip(num_qubits, zne_errors):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    for x, y in zip(num_qubits, pce_errors):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    # Configuration
    data_dir = "data_PCE_vs_ZNE/rand_cliffs"
    output_dir = "plots"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data_from_csvs(data_dir)
    
    if not data:
        print("No data files found!")
        return
    
    # Create and save bar plot
    bar_plot_path = os.path.join(output_dir, "error_comparison_bar.png")
    create_bar_plot(data, bar_plot_path)
    
    # Create and save line plot
    line_plot_path = os.path.join(output_dir, "error_comparison_line.png")
    create_line_plot(data, line_plot_path)

if __name__ == "__main__":
    main()