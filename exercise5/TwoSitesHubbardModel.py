import numpy as np
import matplotlib.pyplot as plt

#plot energies as function of t/U
U = 1.0
t_U = np.linspace(0.01, 1.0, 500)
t = t_U * U

# 4 Energy Levels
E0 = np.zeros_like(t_U)                         # Triplet states 
E1 = np.ones_like(t_U) * U                      # Non-bonding singlet 
E2 = 0.5 * (U - np.sqrt(U**2 + 16 * t**2))      # Ground singlet
E3 = 0.5 * (U + np.sqrt(U**2 + 16 * t**2))      # Excited singlet

#plot
plt.figure(figsize=(8, 5))
plt.plot(t_U, E0, label='Triplet States', color='black')
plt.plot(t_U, E1, label='Non-bonding Singlet', color='blue')
plt.plot(t_U, E2, label='Ground Singlet', color='green')
plt.plot(t_U, E3, label='Excited Singlet', color='red')
plt.xlabel(r'$t/U$')
plt.ylabel('Energy (in units of U)')
plt.title('Two Sites Hubbard Model')
plt.legend()
plt.grid(True)
plt.show()

# Limits t << U and t >> U

# t << U 
t_U_small = np.linspace(0.01, 0.3, 500)
U_small = 1 
t_small = t_U_small * U_small
ground_exact_small = 0.5 * (U_small - np.sqrt(U_small**2 + 16 * t_small**2))  
ground_approx_small = -4 * t_U_small**2

# t >> U
U_t_large = np.linspace(0.01, 3.0, 300)
t_large = 1.0
U_large = U_t_large * t_large
ground_exact_large = 0.5 * (U_large - np.sqrt(U_large**2 + 16 * t_large**2)) 
ground_approx_large = (-2 * t_large) + (U_large / 2) - ((U_large**2) / (16 * t_large))

# Plotting
plt.figure(figsize=(12, 5))

# Subplot 1: Strongly Correlated Limit (t << U)
plt.subplot(1, 2, 1)
plt.plot(t_U_small, ground_exact_small, label='Exact Ground State Energy', color='green', linewidth=2)
plt.plot(t_U_small, ground_approx_small, label=r'2nd Order: $-4t^2/U$', color='red', linestyle='--')
plt.xlabel('$t/U$')
plt.ylabel('Energy (in units of U)')
plt.title(r'Ground State Energy: $t \ll U$')
plt.grid(True)
plt.legend()
# Subplot 2: Weakly Correlated Limit (t >> U)
plt.subplot(1, 2, 2)
plt.plot(U_t_large, ground_exact_large, label='Exact Ground State Energy', color='green', linewidth=2)
plt.plot(U_t_large, ground_approx_large, label=r'2nd Order: $-2t + U/2 - U^2/16t$', color='red', linestyle='--')
plt.xlabel('$U/t$')
plt.ylabel('Energy (in units of t)')
plt.title(r'Ground State Energy: $t \gg U$')
plt.grid(True)
plt.legend()

plt.show()