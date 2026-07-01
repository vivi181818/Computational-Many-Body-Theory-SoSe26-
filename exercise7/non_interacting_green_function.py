import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters ---
t = 1.0          # Hopping parameter
a = 1.0          # Lattice constant
mu = 0.0         # Chemical potential
beta = 20.0      # Inverse temperature (beta = 1/T)
n_max = 50       # Number of positive Matsubara frequencies to compute
n_spin = 2       # Spin index size (0 for spin-up, 1 for spin-down)

# Define Matsubara frequencies
n_indices = np.arange(-n_max, n_max)
V_n = (2 * n_indices + 1) * np.pi / beta

# Initialize the array with indices (sigma, n)
G_sigma_n = np.zeros((n_spin, len(V_n)), dtype=np.complex128)

# Brillouin Zone Summation - Convergence
Nk_values = [8, 20, 200]

print("Running Brillouin Zone integration and convergence analysis...")
plt.figure(figsize=(12, 8))

# Store converged temporary local array for symmetry check
g_loc_converged = None

for Nk in Nk_values:
    kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    Kx, Ky = np.meshgrid(kx, ky)
    
    # 2D Tight-binding dispersion
    eps_k = -2.0 * t * (np.cos(Kx * a) + np.cos(Ky * a))
    
    # Preallocate temporary Green function array for current grid size
    G_loc = np.zeros(len(V_n), dtype=np.complex128)
    
    # Utilize Symmetry: Calculate only for positive n_indices 
    for idx, n in enumerate(range(n_max, 2 * n_max)):
        w_n = 1j * V_n[n]
        denominator = w_n - eps_k + mu
        G_loc[n] = np.sum(1.0 / denominator) / (Nk * Nk)
        
    # Apply symmetry relations for negative frequencies
    for n in range(0, n_max):
        G_loc[n] = np.conj(G_loc[2*n_max - 1 - n])

    if Nk == 200:
        g_loc_converged = G_loc
        # Save into the (sigma, n) array for both spin channels (sigma=0 and sigma=1)
        G_sigma_n[0, :] = G_loc
        G_sigma_n[1, :] = G_loc  # Unpolarized: spin-up and spin-down are identical
        
        plt.plot(V_n, np.real(G_sigma_n[0, :]), label=r'$\mathrm{Re } \, G_{\sigma=0}(iV_n)$ - $N_k=200$', color='blue')
        plt.plot(V_n, np.imag(G_sigma_n[0, :]), label=r'$\mathrm{Im } \, G_{\sigma=0}(iV_n)$ - $N_k=200$', color='orange')
    elif Nk == 20:
        plt.plot(V_n, np.imag(G_loc), linestyle='--', color='green', alpha=0.5, label=r'$\mathrm{Im } \, G_{\sigma}(iV_n)$ ($N_k=20$)')
    elif Nk == 8:
        plt.plot(V_n, np.imag(G_loc), linestyle='--', color='red', alpha=0.5, label=r'$\mathrm{Im } \, G_{\sigma}(iV_n)$ ($N_k=8$)')

plt.title(r'Local Matsubara Green Function $G(iV_n)$ & $N_k$ Convergence', fontsize=14)
plt.xlabel(r'Matsubara Frequency $V_n$', fontsize=12)
plt.ylabel(r'$G(iV_n)$', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# --- 3. Symmetry Verification ---
print("\n--- Verifying Symmetry Properties ---")
re_check = np.allclose(np.real(g_loc_converged), np.real(g_loc_converged[::-1]))
im_check = np.allclose(np.imag(g_loc_converged), -np.imag(g_loc_converged[::-1]))

print(f"Real part is symmetric (even function): {re_check}")
print(f"Imaginary part is antisymmetric (odd function): {im_check}")
print(f"Shape of array with indices (sigma, n): {G_sigma_n.shape}")