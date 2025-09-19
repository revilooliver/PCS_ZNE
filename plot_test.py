import matplotlib
# matplotlib.use('Agg')  # Use the non-interactive backend Agg
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.figure()
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Save the plot to a file
plt.savefig('plot_test.png', dpi=100)
print("Plot saved successfully.")
