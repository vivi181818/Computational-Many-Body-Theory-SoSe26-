import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

# --- 1. Parameters ---
t = 1.0        # Hopping parameter
a = 1.0        # Lattice constant
delta = 0.05   # Broadening parameter (infinitesimal)
mu = 0.0       # Chemical potential
temp = 0.0     # Finite temperature (smears the step function)

# Frequency range (omega)
omega = np.linspace(-6.0, 6.0, 2000)

# --- 2. Brillouin Zone Summation & Convergence Analysis ---
Nk_values = [20, 50, 100, 200]
plt.figure(figsize=(12, 7))

for Nk in Nk_values:
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    Kx, Ky = np.meshgrid(kx, ky)
    
    # Dispersion over mesh relative to chemical potential: omega - epsilon(k) - mu
    eps_k_mesh = -2.0 * t * (np.cos(Kx * a) + np.cos(Ky * a))
    
    G_loc = np.zeros_like(omega, dtype=np.complex128)
    for idx, w in enumerate(omega):
        denominator = w - eps_k_mesh - mu + 1j * delta
        G_loc[idx] = np.sum(1.0 / denominator) / (Nk * Nk)
        
    dos = -np.imag(G_loc) / np.pi
    
    if Nk == 20:
        plt.plot(omega, dos, label=r'$N_k=20$', linestyle=':', alpha=0.7)
    elif Nk == 50:
        plt.plot(omega, dos, label=r'$N_k=50$', linestyle='-.', alpha=0.8)
    elif Nk == 100:
        plt.plot(omega, dos, label=r'$N_k=100$', linestyle='--', alpha=0.9)
    elif Nk == 200:
        plt.plot(omega, dos, label=r'$N_k=200$ (Converged)', color='#d62728', linewidth=2)

plt.title(r'Convergence of Local Density of States $\rho(\omega)$ with Grid Size $N_k$', fontsize=14)
plt.xlabel(r'Frequency $\omega$', fontsize=12)
plt.ylabel(r'DOS $\rho(\omega)$', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(-5, 5)
plt.ylim(0, 0.5)
plt.tight_layout()
plt.show()

# --- 3. Verification of Sum Rule & Calculations (Nk = 200) ---
Nk_final = 200
kx = np.linspace(-np.pi, np.pi, Nk_final)
ky = np.linspace(-np.pi, np.pi, Nk_final)
Kx, Ky = np.meshgrid(kx, ky)
eps_k_mesh = -2.0 * t * (np.cos(Kx * a) + np.cos(Ky * a))

G_loc_final = np.zeros_like(omega, dtype=np.complex128)
for idx, w in enumerate(omega):
    G_loc_final[idx] = np.sum(1.0 / (w - eps_k_mesh - mu + 1j * delta)) / (Nk_final**2)

# Spectral function A(omega) = -1/pi * Im(G_loc)
spectral_function = -np.imag(G_loc_final) / np.pi

# Applying Fermi-Dirac distribution
if temp == 0.0:
    f_omega = np.where(omega <= mu, 1.0, 0.0)
else:
    # Use (omega - mu) in the exponent for correct chemical potential shifting
    f_omega = 1.0 / (np.exp((omega - mu) / temp) + 1.0)

# Number of electrons: n = \int A(\omega) f(\omega) d\omega
n_electrons = trapz(spectral_function * f_omega, omega)

print(f"\n--- Analysis & Verification ---")
print(f"Calculated electron filling (n) for μ={mu}, T={temp}: {n_electrons:.4f}")

# --- 4. Plotting Final Spectral and Green Functions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(omega, np.real(G_loc_final), label=r'$\mathrm{Re } \, G_{\mathrm{loc}}(\omega)$', color='#1f77b4', linewidth=2)
axes[0].plot(omega, np.imag(G_loc_final), label=r'$\mathrm{Im } \, G_{\mathrm{loc}}(\omega)$', color='#ff7f0e', linewidth=2, linestyle='--')
axes[0].set_title(r'Local Retarded Green\'s Function $G_{\mathrm{loc}}(\omega)$', fontsize=14)
axes[0].set_xlabel(r'Frequency $\omega$', fontsize=12)
axes[0].set_ylabel(r'$G_{\mathrm{loc}}(\omega)$', fontsize=12)
axes[0].axhline(0, color='black', linewidth=0.5, linestyle=':')
axes[0].axvline(mu, color='red', linewidth=1.0, linestyle='-.', label=r'Chem. Potential $\mu$')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Plotting temperature smeared occupied states
occupied_states = spectral_function * f_omega
axes[1].plot(omega, spectral_function, color='#2ca02c', linewidth=2.5, label=r'$A(\omega)$')
axes[1].plot(omega, occupied_states, color='#17becf', linewidth=2.0, linestyle='-', label=r'Occupied States $A(\omega)f(\omega)$')
axes[1].fill_between(omega, occupied_states, color='#17becf', alpha=0.2)

axes[1].set_title(r'Spectral Function & Thermal Occupation ($T=$' + f'{temp}' + r')', fontsize=14)
axes[1].set_xlabel(r'Frequency $\omega$', fontsize=12)
axes[1].set_ylabel(r'$A(\omega)$', fontsize=12)
axes[1].axvline(mu, color='red', linewidth=1.0, linestyle='-.', label=r'Chem. Potential $\mu$')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_xlim(-5, 5)

plt.tight_layout()
plt.show()