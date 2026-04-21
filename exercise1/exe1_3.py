import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 200)
Z_values = [1, 3, 5]

def dirac_delta(x, Z):
    return 2 * np.sin(Z * x) / x

for Z in Z_values:
    y = dirac_delta(x, Z)
    plt.plot(x, y, label=f"Z = {Z}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Dirac Delta Approximation")
plt.legend()
plt.grid()

plt.show()