import numpy as np
import matplotlib.pyplot as plt

#Hartee Fock self energy
def compute_G_loc_selfenergy(mu, n, m, U, eps_k, beta, n_max=500):

    Sigma_up = U * (0.5 * n - m)
    Sigma_down = U * (0.5 * n + m)

    n_indices = np.arange(-n_max, n_max)
    V_n = (2*n_indices + 1) * np.pi / beta
    G_loc_up = np.zeros(len(V_n), dtype=np.complex128)
    G_loc_down = np.zeros(len(V_n), dtype=np.complex128)

    for n in range(n_max, 2 * n_max):
        w_n     = 1j * V_n[n]
        G_loc_up[n] = np.mean(1.0 / (w_n - eps_k - Sigma_up + mu))
        G_loc_down[n] = np.mean(1.0 / (w_n - eps_k - Sigma_down + mu))
    for n in range(n_max):
        G_loc_up[n] = np.conj(G_loc_up[2 * n_max - 1 - n])
        G_loc_down[n] = np.conj(G_loc_down[2 * n_max - 1 - n])
    return V_n, G_loc_up, G_loc_down

def compute_n_sigma_redefined(mu_val, n_val, m_val, n_max=500):
    
    V_n, G_loc_up, G_loc_down = compute_G_loc_selfenergy(mu_val, n_val, m_val, U, eps_k, beta, n_max)

    iw_n = 1j * V_n

    # G(0+) via tail-subtracted sum evaluated at tau -> 0+
    # phase: exp(0)=1
    G_0plus_up   = (1.0 / beta) * np.sum(G_loc_up   - 1.0 / iw_n) - 0.5
    G_0plus_down = (1.0 / beta) * np.sum(G_loc_down - 1.0 / iw_n) - 0.5

    n_up   = 1.0 + np.real(G_0plus_up)     # per spin: G(0+) = -1 + n_sigma
    n_down = 1.0 + np.real(G_0plus_down)
    return n_up, n_down


def self_consistent_mu_redefined(N_e_target, m_val, U, alpha=0.5, mu_init=0.0,
                        max_iter=200, tol=1e-8):
    mu_HF = N_e_target * U * 0.5
    mu = mu_init + mu_HF
    for i in range(max_iter):
        n_up, n_down = compute_n_sigma_redefined(mu, N_e_target, m_val)
        N_e = n_up + n_down
        error  = N_e - N_e_target
        if abs(error) < tol:
            return mu, N_e, i + 1, True, mu - mu_HF
        mu_trial = mu - error
        mu = (1.0 - alpha) * mu + alpha * mu_trial
    return mu, N_e, max_iter, False, mu - mu_HF  

#  parameters
t    = 1.0
a    = 1.0
U    = 2.0          
beta = 20.0          

# kpoint grid
Nk = 200
kx = -np.pi + 2*np.pi*np.arange(Nk)/Nk
ky = -np.pi + 2*np.pi*np.arange(Nk)/Nk
Kx, Ky = np.meshgrid(kx, ky)
eps_k = -2.0 * t * (np.cos(Kx * a) + np.cos(Ky * a)) 

N_e_target = 1.0     # half-filling

m_values = [0.0, 0.2, 0.5]  

for m in m_values:
    mu_val, N_e, iters, converged, mu_shift = self_consistent_mu_redefined(N_e_target, m, U)
    print(f"m = {m:.2f}:  mu = {mu_val:.6f}, mu - mu_HF = {mu_shift:.6f}, "
          f"N_e = {N_e:.6f}, iters = {iters}, converged = {converged}")

    # local Green's function at the self-consistent mu for this m
    V_n, G_loc_up, G_loc_down = compute_G_loc_selfenergy(
        mu_val, N_e_target, m, U, eps_k, beta
    )
