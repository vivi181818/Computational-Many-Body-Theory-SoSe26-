import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(1e-20, 15, 500)
theta = np.linspace(0, np.pi, 100) #pole to pole
phi = np.linspace(0, 2 * np.pi, 100) #equator
n = 2
ao = 1
l_values = [0, 1]

def radial_part(n, l, ao, r):
    if l == 0:
        return np.sqrt(2/ao) * (r/(n*ao))**(l+1) * (1-(r/(2*ao))) * np.exp(-r/(2*ao)) 
    if l == 1:
        return np.sqrt(3/(3*ao)) * (r/(n*ao))**(l+1) * 1 * np.exp(-r/(2*ao)) 
    return 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for l in l_values:
    u_y = radial_part(n, l, ao, r)
    r_y = u_y / r
    ax1.plot(r, u_y, label=fr"$U_{{{n},{l}}}(r)$")
    ax2.plot(r, r_y, label=fr"$R_{{{n},{l}}}(r)$")


ax1.set_title(f"Reduced Radial Wavefunction $U(r)$")
ax1.set_xlabel(r"Radius $r$")
ax1.set_ylabel(r"$U(r)$")
ax1.legend() 
ax1.grid()
ax2.set_title(f"Radial Wavefunction $R(r)$")
ax2.set_xlabel(r"Radius $r$")
ax2.set_ylabel(r"$R(r)$")
ax2.legend() 
ax2.grid()
plt.show()

def angular_part(l, m, theta, phi):
    if l == 0 and m == 0:
        return np.sqrt(1 / (4 * np.pi))  * np.ones_like(theta)
    if l == 1:
        if m == 0:
            return np.sqrt(3 / (4 * np.pi)) * np.cos(theta) 
        if m == -1:
            return np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(-1j * phi)
        if m == 1:
            return -1* np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(1j * phi) 
    return 0


def plot_orbital(axis, values, Theta, Phi, title):

    R_mag = np.abs(values)

    X = R_mag * np.sin(Theta) * np.cos(Phi)
    Y = R_mag * np.sin(Theta) * np.sin(Phi)
    Z = R_mag * np.cos(Theta)

    face_colors = np.empty(values.shape + (3,)) 
    face_colors[values > 0] = [1, 0, 0]   
    face_colors[values < 0] = [0, 0, 1]   
    face_colors[values == 0] = [1, 1, 1]  
    
    colors_final = face_colors[:-1, :-1]

    axis.plot_surface(
        X, Y, Z,
        facecolors=colors_final,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    axis.set_title(title)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')


Theta, Phi = np.meshgrid(theta, phi)

S = np.real(angular_part(0,0,Theta, Phi))
Pz = np.real(angular_part(1,0,Theta, Phi))
Px = np.real(np.sqrt(0.5) * (angular_part(1,-1,Theta, Phi) - angular_part(1,1,Theta, Phi)))
Py = -np.imag(np.sqrt(0.5) * (angular_part(1,-1,Theta, Phi) + angular_part(1,1,Theta, Phi)))
    
fig2, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(10, 10))
plot_orbital(axes[0, 0], S,  Theta, Phi, "2s orbital")
plot_orbital(axes[0, 1], Pz, Theta, Phi, "2pz orbital")
plot_orbital(axes[1, 0], Px, Theta, Phi, "2px orbital")
plot_orbital(axes[1, 1], Py, Theta, Phi, "2py orbital")
plt.show()



limit_val = 15
x_ext = np.linspace(-limit_val, limit_val, 400)
y_ext = np.linspace(-limit_val, limit_val, 400)
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

R_ext = np.sqrt(X_ext**2 + Y_ext**2)
Theta_ext = np.pi / 2
Phi_ext = np.arctan2(Y_ext, X_ext)


R_s = radial_part(2, 0, ao, R_ext) / R_ext
R_p = radial_part(2, 1, ao, R_ext) / R_ext

orbitals_2d = {
    "2s": R_s * np.real(angular_part(0, 0, Theta_ext, Phi_ext)),
    "2pz": R_p * np.real(angular_part(1, 0, Theta_ext, Phi_ext)),
    "2px": R_p * np.real(np.sqrt(0.5) * (angular_part(1, -1, Theta_ext, Phi_ext) - 
                                        angular_part(1, 1, Theta_ext, Phi_ext))),
    "2py": R_p * -np.imag(np.sqrt(0.5) * (angular_part(1, -1, Theta_ext, Phi_ext) + 
                                         angular_part(1, 1, Theta_ext, Phi_ext)))
}


fig3, axes3 = plt.subplots(2, 2, figsize=(12, 12))
titles = list(orbitals_2d.keys())
solid_cmap = plt.matplotlib.colors.ListedColormap(['blue', 'red'])
for i, ax in enumerate(axes3.flat):
    name = titles[i]
    wavefunc = orbitals_2d[name]
    max_val = np.max(np.abs(wavefunc))
    threshold = max_val * 0.1  
    
    if max_val > 1e-15:
        color_zones = [-max_val, -threshold, threshold, max_val]
        custom_color = ['blue', 'white', 'red']
        ax.contourf(X_ext, Y_ext, wavefunc, levels=color_zones, colors=custom_color)
    else:
        ax.text(0, 0, "Nodal Plane (Zero)", ha='center', color='gray')

    ax.set_title(f"{name} orbital")
    ax.set_aspect('equal')
    ax.set_xlim(-limit_val, limit_val)
    ax.set_ylim(-limit_val, limit_val)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()