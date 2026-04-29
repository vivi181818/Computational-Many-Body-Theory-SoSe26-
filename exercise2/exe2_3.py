import numpy as np
import matplotlib.pyplot as plt

def heaviside(x):
    return np.piecewise(x, [x < 0, x == 0, x > 0], [0, 0.5, 1])


x_values = np.linspace(-5, 5, 1000)
y_values1 = heaviside(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values1, label=r'$\Theta(x)$', color='blue', lw=2)
plt.title('Heaviside Step Function $\Theta(x)$')
plt.legend()
plt.grid()
plt.show()

B_values = [1, 3, 5, 10]
miu_value = 1

def fermifunction(x,b,miu):
    return 1 / (1 + np.exp((x-miu)*b))

plt.figure(figsize=(10, 6))
for B in B_values:
    y_values2 = fermifunction(x_values, B, miu_value)
    plt.plot(x_values, y_values2, label=f" β = {B}") 
plt.title('Fermi function with various β')
plt.legend()
plt.grid()
plt.show()