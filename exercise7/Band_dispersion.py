import numpy as np
import matplotlib.pyplot as plt

def dispersion_energy_calc(k_vec, t, lat_vec):
    energy = 0.0
    for lat_vec_ele in lat_vec:
        k_dot = np.dot(k_vec, lat_vec_ele)
        energy += np.cos(k_dot)

    dispersion_energy = -t * energy 
    
    return dispersion_energy

# ==========================================
#  EXECUTION
# ==========================================

# parameter
t = 1.0
a = 1.0

# Cubic Lattice Nearest Neighbors
lat_vec_1d = np.array([[a], [-a]])
lat_vec_2d = np.array([[a, 0.0], [-a, 0.0], [0.0, a], [0.0, -a]])
lat_vec_3d = np.array([
    [a, 0.0, 0.0], [-a, 0.0, 0.0],
    [0.0, a, 0.0], [0.0, -a, 0.0],
    [0.0, 0.0, a], [0.0, 0.0, -a]
])

#  1D DISPERSION PLOT
k_vals_1d = np.linspace(-np.pi / a, np.pi / a, 300)
E_vals_1d = []
for k in k_vals_1d:
    E_vals_1d.append(dispersion_energy_calc(np.array([k]), t, lat_vec_1d))

plt.figure(figsize=(7, 5))
plt.plot(k_vals_1d / np.pi, E_vals_1d, color='#1f77b4', linewidth=2.5)
plt.xlabel('$k_x / \pi$ (units of $1/a$)', fontsize=12)
plt.ylabel('Energy $E(k_x)$', fontsize=12)
plt.title('1D Chain Tight-Binding Dispersion', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlim(-1.0, 1.0)
plt.tight_layout()
plt.savefig('1d_disp_separate.png', dpi=300)
plt.show()


#  2D BZ HEATMAP
k_axis_2d = np.linspace(-np.pi / a, np.pi / a, 300)
kx, ky = np.meshgrid(k_axis_2d, k_axis_2d) #(300, 300)
energy_2d = np.zeros_like(kx)

for i in range(kx.shape[0]):
    for j in range(kx.shape[1]):
        k_vec = np.array([kx[i, j], ky[i, j]])
        energy_2d[i, j] = dispersion_energy_calc(k_vec, t, lat_vec_2d)

plt.figure(figsize=(7, 6))
c = plt.pcolormesh(kx / np.pi, ky / np.pi, energy_2d, cmap='viridis', shading='auto')
plt.colorbar(c, label='Energy $E(k_x, k_y)$')
plt.xlabel('$k_x / \pi$', fontsize=12)
plt.ylabel('$k_y / \pi$', fontsize=12)
plt.title('2D First Brillouin Zone (Top-Down View)', fontsize=14)
plt.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
plt.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#  2D BAND STRUCTURE ALONG Gamma-X-M-Gamma PATH
N_pts = 100

# Segment 1: Gamma to X (kx goes 0 to pi/a, ky = 0)
kx_1 = np.linspace(0, np.pi / a, N_pts)
ky_1 = np.zeros(N_pts)
path_g_x = np.array([[kx, ky] for kx, ky in zip(kx_1, ky_1)])

# Segment 2: X to M (kx = pi/a, ky goes 0 to pi/a)
kx_2 = np.full(N_pts, np.pi / a)
ky_2 = np.linspace(0, np.pi / a, N_pts)
path_x_m = np.array([[kx, ky] for kx, ky in zip(kx_2, ky_2)])

# Segment 3: M to Gamma (kx goes pi/a to 0, ky goes pi/a to 0)
kx_3 = np.linspace(np.pi / a, 0, N_pts)
ky_3 = np.linspace(np.pi / a, 0, N_pts)
path_m_g = np.array([[kx, ky] for kx, ky in zip(kx_3, ky_3)])

# Combine path segments (skipping overlapping points at boundaries)
# Compute energy along the path
k_path = np.vstack((path_g_x, path_x_m[1:], path_m_g[1:]))
E_vals_2d = []
for kv in k_path:
    E_vals_2d.append(dispersion_energy_calc(kv, t, lat_vec_2d))

# Calculate cumulative physical distance along the path for the x-axis
dist = np.zeros(k_path.shape[0])
for i in range(1, len(dist)):
    dx = k_path[i, 0] - k_path[i-1, 0]
    dy = k_path[i, 1] - k_path[i-1, 1]
    dist[i] = dist[i-1] + np.sqrt(dx**2 + dy**2)

# Plotting the band structure
plt.figure(figsize=(9, 5))
plt.plot(dist, E_vals_2d, color='crimson', linewidth=2.5)
plt.xlabel('High-Symmetry Path', fontsize=12)
plt.ylabel('Energy $E(k)$', fontsize=12)
plt.title('2D Tight-Binding Band Structure along $\mathbf{\Gamma}$ - X - M - $\mathbf{\Gamma}$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Set custom ticks at Gamma, X, M, Gamma positions
idx_x = N_pts - 1
idx_m = idx_x + (N_pts - 1)
idx_gamma_end = len(dist) - 1
tick_positions = [dist[0], dist[idx_x], dist[idx_m], dist[idx_gamma_end]]
plt.xticks(tick_positions, ['$\mathbf{\Gamma}$', 'X', 'M', '$\mathbf{\Gamma}$'])
plt.xlim(dist[0], dist[-1])
plt.tight_layout()
plt.show()


fig_2d = plt.figure(figsize=(10, 7))
ax_2d = fig_2d.add_subplot(111, projection='3d')
surf = ax_2d.plot_surface(kx / np.pi, ky / np.pi, energy_2d, cmap='viridis', edgecolor='none')
ax_2d.set_xlabel('$k_x / \pi$', fontsize=12)
ax_2d.set_ylabel('$k_y / \pi$', fontsize=12)
ax_2d.set_zlabel('Energy $E$', fontsize=12)
ax_2d.set_title('2D Tight-Binding Energy Surface', fontsize=14)
fig_2d.colorbar(surf, shrink=0.5, aspect=5, label='Energy')
plt.show()








# ==========================================
# 3D BAND STRUCTURE
# Gamma -> X -> M -> R -> Gamma

N_pts = 100

# Gamma -> X
kx_1 = np.linspace(0, np.pi/a, N_pts)
ky_1 = np.zeros(N_pts)
kz_1 = np.zeros(N_pts)
path_g_x = np.array([[kx, ky, kz]
                     for kx, ky, kz in zip(kx_1, ky_1, kz_1)])

# X -> M
kx_2 = np.full(N_pts, np.pi/a)
ky_2 = np.linspace(0, np.pi/a, N_pts)
kz_2 = np.zeros(N_pts)
path_x_m = np.array([[kx, ky, kz]
                     for kx, ky, kz in zip(kx_2, ky_2, kz_2)])

# M -> R
kx_3 = np.full(N_pts, np.pi/a)
ky_3 = np.full(N_pts, np.pi/a)
kz_3 = np.linspace(0, np.pi/a, N_pts)
path_m_r = np.array([[kx, ky, kz]
                     for kx, ky, kz in zip(kx_3, ky_3, kz_3)])

# R -> Gamma
kx_4 = np.linspace(np.pi/a, 0, N_pts)
ky_4 = np.linspace(np.pi/a, 0, N_pts)
kz_4 = np.linspace(np.pi/a, 0, N_pts)
path_r_g = np.array([[kx, ky, kz]
                     for kx, ky, kz in zip(kx_4, ky_4, kz_4)])

k_path_3d = np.vstack((
    path_g_x,
    path_x_m[1:],
    path_m_r[1:],
    path_r_g[1:]
))

E_vals_3d = []

for kv in k_path_3d:
    E_vals_3d.append(
        dispersion_energy_calc(kv, t, lat_vec_3d)
    )

dist = np.zeros(len(k_path_3d))

for i in range(1, len(dist)):
    dk = k_path_3d[i] - k_path_3d[i-1]
    dist[i] = dist[i-1] + np.linalg.norm(dk)

plt.figure(figsize=(9, 5))
plt.plot(dist, E_vals_3d,
         color='darkgreen',
         linewidth=2.5)

plt.xlabel('High-Symmetry Path')
plt.ylabel('Energy $E(k)$')
plt.title(
    '3D Tight-Binding Band Structure along '
    '$\\mathbf{\\Gamma}$-X-M-R-$\\mathbf{\\Gamma}$'
)

idx_x = N_pts - 1
idx_m = idx_x + (N_pts - 1)
idx_r = idx_m + (N_pts - 1)

tick_positions = [
    dist[0],
    dist[idx_x],
    dist[idx_m],
    dist[idx_r],
    dist[-1]
]

plt.xticks(
    tick_positions,
    ['$\\mathbf{\\Gamma}$', 'X', 'M', 'R', '$\\mathbf{\\Gamma}$']
)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ==========================================
# 3D FERMI SURFACE (isosurface E = 0)
# ==========================================

from skimage import measure
from mpl_toolkits.mplot3d import Axes3D

# k-space grid (IMPORTANT: increase for smoother surface)
N = 50  # 40–80 recommended depending on speed

k_axis = np.linspace(-np.pi/a, np.pi/a, N)
kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing='ij')

energy_3d_grid = np.zeros_like(kx)

# compute dispersion on full 3D grid
for i in range(N):
    for j in range(N):
        for k in range(N):
            k_vec = np.array([kx[i, j, k], ky[i, j, k], kz[i, j, k]])
            energy_3d_grid[i, j, k] = dispersion_energy_calc(k_vec, t, lat_vec_3d)

# shift grid so coordinates are physical k-space
spacing = k_axis[1] - k_axis[0]

# extract isosurface E = 0
verts, faces, normals, values = measure.marching_cubes(
    energy_3d_grid,
    level=0.0,
    spacing=(spacing, spacing, spacing)
)

# shift vertices to correct k-space origin
verts += k_axis[0]

# plot Fermi surface
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

mesh = ax.plot_trisurf(
    verts[:, 0] / np.pi,
    verts[:, 1] / np.pi,
    faces,
    verts[:, 2] / np.pi,
    cmap='coolwarm',
    lw=0.2,
    alpha=0.9
)

ax.set_xlabel('$k_x / \pi$')
ax.set_ylabel('$k_y / \pi$')
ax.set_zlabel('$k_z / \pi$')
ax.set_title('3D Fermi Surface (E = 0)')

plt.tight_layout()
plt.show()