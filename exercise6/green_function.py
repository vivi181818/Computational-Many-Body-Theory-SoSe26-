import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters ---
t = 1.0        # Hopping parameter
a = 1.0        # Lattice constant
delta = 0.05   # Small imaginary part

# Define a specific k-point, e.g., k = (pi/2, pi/2)
kx = np.pi /2
ky = np.pi /2

# Frequency range (omega)
omega = np.linspace(-6.0, 6.0, 1000)

# --- 2. Calculate Dispersion and Green's Function ---
# Dispersion relation epsilon(k)
eps_k = -2.0 * t * (np.cos(kx * a) + np.cos(ky * a))
print(f"Dispersion value epsilon(k) for k=({kx/np.pi:.2f}*pi, {ky/np.pi:.2f}*pi) = {eps_k:.4f}")

# Retarded Green's function: G_0(k, omega) = 1 / (omega - epsilon_k + i * delta)
denominator = omega - eps_k + 1j * delta
G_k_omega = 1.0 / denominator

# Extract real and imaginary parts
real_part = np.real(G_k_omega)
imag_part = np.imag(G_k_omega)

# --- 3. Plotting ---
plt.figure(figsize=(10, 6))

plt.plot(omega, real_part, label=r'$\mathrm{Re } \, G_0(\mathbf{k}, \omega)$', color='#1f77b4', linewidth=2)
plt.plot(omega, imag_part, label=r'$\mathrm{Im } \, G_0(\mathbf{k}, \omega)$', color='#ff7f0e', linewidth=2, linestyle='--')

# Styling the plot
plt.title(r'Retarded Green function $G_0(\mathbf{k}, \omega)$ for $\mathbf{k} = (' + f'{kx/np.pi:.2f}' + r'\pi, ' + f'{ky/np.pi:.2f}' + r'\pi)$', fontsize=14)
plt.xlabel(r'Frequency $\omega$', fontsize=12)
plt.ylabel(r'$G_0(\mathbf{k}, \omega)$', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(eps_k, color='red', linewidth=1.0, linestyle='-.', label=r'$\omega = \varepsilon(\mathbf{k})$')

plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(-1/delta - 2, 1/delta + 2) # Dynamically scales based on 1/delta peak height

# Show the plot
plt.tight_layout()
plt.show()