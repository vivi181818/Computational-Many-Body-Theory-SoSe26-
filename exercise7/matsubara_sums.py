import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters ---
t    = 1.0
a    = 1.0
mu   = 0.0
beta = 20.0

# kpoint grid
Nk = 200
kx = -np.pi + 2*np.pi*np.arange(Nk)/Nk
ky = -np.pi + 2*np.pi*np.arange(Nk)/Nk
Kx, Ky = np.meshgrid(kx, ky)
eps_k = -2.0 * t * (np.cos(Kx * a) + np.cos(Ky * a))

# tau grid
N_tau    = 800
tau_pos  = np.linspace(0, beta, N_tau, endpoint=False)
tau_eps  = 1e-4

n_max_values = [50, 100, 250, 500]

# use symmetry to compute the Green function
def compute_G_loc(n_max):
    n_indices = np.arange(-n_max, n_max)
    V_n  = (2 * n_indices + 1) * np.pi / beta
    G_loc = np.zeros(len(V_n), dtype=np.complex128)
    for n in range(n_max, 2 * n_max):
        w_n     = 1j * V_n[n]
        G_loc[n] = np.mean(1.0 / (w_n - eps_k + mu))
    for n in range(n_max):
        G_loc[n] = np.conj(G_loc[2 * n_max - 1 - n])
    return V_n, G_loc


def G_tau_from_Matsubara(V_n, G_loc, use_tail=False):
    tau_mesh = tau_pos[:, None]
    V_mesh   = V_n[None, :]
    phase    = np.exp(-1j * tau_mesh * V_mesh)

    if use_tail:
        iw_n         = 1j * V_n
        summed       = G_loc[None, :] - (1.0 / iw_n[None, :])
        G_tau_pos    = (1.0 / beta) * np.sum(summed * phase, axis=1) - 0.5
    else:
        G_tau_pos    = (1.0 / beta) * np.sum(G_loc[None, :] * phase, axis=1)

    tau_neg  = tau_pos[1:] - beta    # shape N_tau-1, spans (-beta, 0)
    G_tau_neg = -G_tau_pos[1:]       # exact anti-periodic partners

    tau_full = np.concatenate([tau_neg, tau_pos])
    G_full   = np.concatenate([G_tau_neg, G_tau_pos])
    idx      = np.argsort(tau_full)
    return tau_full[idx], np.real(G_full[idx]), np.real(G_tau_pos), np.real(G_tau_neg)



# Figure 1 — Plain Fourier sum (no tail subtraction)
fig, ax = plt.subplots(figsize=(12, 7))

for n_max in n_max_values:
    V_n, G_loc = compute_G_loc(n_max)
    tau_sorted, G_sorted, _, _ = G_tau_from_Matsubara(V_n, G_loc, use_tail=False)
    ax.plot(tau_sorted, G_sorted, label=rf'$n_{{max}}={n_max}$')

ax.axvline(0,         color='black',  alpha=0.4)
ax.axvline(-beta,     color='gray',   ls='--', alpha=0.5)
ax.axvline(beta,      color='gray',   ls='--', alpha=0.5)
ax.axvline(tau_eps,   color='red',    ls=':', label=r'$\tau=0^+$')
ax.axvline(beta-tau_eps, color='purple', ls=':', label=r'$\tau=\beta^-$')

ax.set_title(r'Plain Fourier sum: $G^0_\sigma(\tau) = \frac{1}{\beta}\sum_n G^0_\sigma(\nu_n)e^{-i\tau\nu_n}$', fontsize=13)
ax.set_xlabel(r'Imaginary Time $\tau$', fontsize=12)
ax.set_ylabel(r'Re$\,G^0_\sigma(\tau)$', fontsize=12)
ax.set_xlim(-beta, beta)
ax.set_xticks([-beta, -beta/2, 0, beta/2, beta],
              [r'$-\beta$', r'$-\beta/2$', r'$0$', r'$\beta/2$', r'$\beta$'])
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()



# Figure 2 — Improved: tail subtraction
fig, ax = plt.subplots(figsize=(12, 7))

for n_max in n_max_values:
    V_n, G_loc = compute_G_loc(n_max)
    tau_sorted, G_sorted, _, _ = G_tau_from_Matsubara(V_n, G_loc, use_tail=True)
    ax.plot(tau_sorted, G_sorted, label=rf'$n_{{max}}={n_max}$')

ax.axvline(0,         color='black',  alpha=0.4)
ax.axvline(-beta,     color='gray',   ls='--', alpha=0.5)
ax.axvline(beta,      color='gray',   ls='--', alpha=0.5)
ax.axvline(tau_eps,   color='red',    ls=':', label=r'$\tau=0^+$')
ax.axvline(beta-tau_eps, color='purple', ls=':', label=r'$\tau=\beta^-$')

ax.set_title(r'Improved (tail subtraction): $G^0_\sigma(\tau) = -\frac{1}{2} + \frac{1}{\beta}\sum_n[G^0_\sigma(\nu_n)-\frac{1}{i\nu_n}]e^{-i\tau\nu_n}$', fontsize=11)
ax.set_xlabel(r'Imaginary Time $\tau$', fontsize=12)
ax.set_ylabel(r'Re$\,G^0_\sigma(\tau)$', fontsize=12)
ax.set_xlim(-beta, beta)
ax.set_xticks([-beta, -beta/2, 0, beta/2, beta],
              [r'$-\beta$', r'$-\beta/2$', r'$0$', r'$\beta/2$', r'$\beta$'])
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()


plt.show()



# Convergence check 
print("\n--- Convergence Check at tau = 0+ ---")
print(f"{'n_max':>8} | {'Plain sum G(0+)':>18} | {'Tail-sub G(0+)':>18}")
print("-" * 52)

for n_max in n_max_values:
    V_n, G_loc = compute_G_loc(n_max)
    zero_idx = np.argmin(np.abs(tau_pos - tau_eps))

    _, _, G_pos_plain, _ = G_tau_from_Matsubara(V_n, G_loc, use_tail=False)
    _, _, G_pos_tail,  _ = G_tau_from_Matsubara(V_n, G_loc, use_tail=True)

    print(f"{n_max:>8d} | {G_pos_plain[zero_idx]:>18.6f} | {G_pos_tail[zero_idx]:>18.6f}")

n_sigma = 0.5
print(f"\nTheoretical:  G(0+) = {-1 + n_sigma:.2f},  G(beta-) = {-n_sigma:.2f}")



# Boundary conditions and anti-periodicity — plain sum
V_n, G_loc = compute_G_loc(n_max_values[-1])
tau_sorted, G_sorted, G_tau_pos, G_tau_neg = G_tau_from_Matsubara(V_n, G_loc, use_tail=False)

zero_plus_idx  = np.argmin(np.abs(tau_sorted - tau_eps))
beta_minus_idx = np.argmin(np.abs(tau_sorted - (beta - tau_eps)))

print("\n--- Boundary Conditions (plain sum, n_max=500) ---")
print(f"G(0+)    = {G_sorted[zero_plus_idx]:.6f}   (expected {-1+n_sigma:.2f})")
print(f"G(beta-) = {G_sorted[beta_minus_idx]:.6f}   (expected {-n_sigma:.2f})")

anti_periodic = np.allclose(G_tau_neg, -G_tau_pos[1:], atol=1e-6)
print(f"Anti-periodicity G(tau-beta) = -G(tau): {anti_periodic}")

# Boundary conditions and anti-periodicity — improved version

V_n, G_loc = compute_G_loc(n_max_values[-1])
tau_sorted, G_sorted, G_tau_pos, G_tau_neg = G_tau_from_Matsubara(V_n, G_loc, use_tail=True)

zero_plus_idx  = np.argmin(np.abs(tau_sorted - tau_eps))
beta_minus_idx = np.argmin(np.abs(tau_sorted - (beta - tau_eps)))

print("\n--- Boundary Conditions (tail-subtracted, n_max=500) ---")
print(f"G(0+)    = {G_sorted[zero_plus_idx]:.6f}   (expected {-1+n_sigma:.2f})")
print(f"G(beta-) = {G_sorted[beta_minus_idx]:.6f}   (expected {-n_sigma:.2f})")

anti_periodic = np.allclose(G_tau_neg, -G_tau_pos[1:], atol=1e-6)
print(f"Anti-periodicity G(tau-beta) = -G(tau): {anti_periodic}")

# Self-consistent chemical potential

def compute_n_e(mu_val, n_max=500):
    n_indices = np.arange(-n_max, n_max)
    V_n  = (2 * n_indices + 1) * np.pi / beta
    G_loc = np.zeros(len(V_n), dtype=np.complex128)
    for n in range(n_max, 2 * n_max):
        w_n      = 1j * V_n[n]
        G_loc[n] = np.mean(1.0 / (w_n - eps_k + mu_val))
    for n in range(n_max):
        G_loc[n] = np.conj(G_loc[2 * n_max - 1 - n])

    # G(0+) via tail-subtracted sum evaluated at tau -> 0+
    iw_n    = 1j * V_n
    G_0plus = (1.0 / beta) * np.sum(G_loc - 1.0 / iw_n) - 0.5

    n_sigma = 1.0 + np.real(G_0plus)   # per spin: G(0+) = -1 + n_sigma
    n_e     = 2.0 * n_sigma             # total (spin up + down)
    return n_e, np.real(G_0plus)


def self_consistent_mu(N_e_target, alpha=0.5, mu_init=0.0,
                        max_iter=200, tol=1e-6):
    mu = mu_init
    for i in range(max_iter):
        n_e, _ = compute_n_e(mu)
        error  = n_e - N_e_target
        if abs(error) < tol:
            return mu, n_e, i + 1, True
        mu_trial = mu - error
        mu = (1.0 - alpha) * mu + alpha * mu_trial
    return mu, n_e, max_iter, False


# --- Verify N_e = 1 -> mu = 0 ---
mu_sc, n_e_sc, iters, conv = self_consistent_mu(N_e_target=1.0, alpha=0.5, mu_init=0.0)
print(f"\n--- Self-consistent mu (N_e=1, alpha=0.5) ---")
print(f"mu = {mu_sc:.6f}  (expected 0.0) | n_e = {n_e_sc:.6f}  (expected 1.0) | iters = {iters} | converged = {conv}")

# --- Effect of mixing parameter ---
print("\n=== Effect of mixing parameter alpha ===")
print(f"{'alpha':>8} | {'mu':>12} | {'n_e':>12} | {'iters':>8} | {'converged':>10}")
print("-" * 58)
for alpha in [0.1, 0.5, 0.9]:
    mu_sc, n_e_sc, iters, conv = self_consistent_mu(
        N_e_target=1.0, alpha=alpha, mu_init=0.5, max_iter=200, tol=1e-5)
    print(f"{alpha:>8.1f} | {mu_sc:>12.6f} | {n_e_sc:>12.6f} | {iters:>8d} | {str(conv):>10}")