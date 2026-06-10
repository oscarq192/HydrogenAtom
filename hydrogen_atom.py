import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Radial equation finite difference solver
N = 10000
r_min = 1e-6
r_max = 100
h = (r_max - r_min) / (N + 1)

n = 4
l = 0
m = 0

r = np.linspace(r_min, r_max, N + 2)
r_i = r[1:-1]

diagonal = (1 / h ** 2) - (1 / r_i) + (l * (l + 1) / (2 * r_i ** 2))
off_diagonal = - 1 / (2 * h ** 2) * np.ones(N - 1)

values, vectors = sp.linalg.eigh_tridiagonal(diagonal, off_diagonal, check_finite=False)
R = vectors / r_i

# Monte Carlo rejection sampling
points = 10000

scatter_x = []
scatter_y = []
scatter_z = []

radial_probability = np.abs(R[:, n - l - 1]) ** 2 * (r_i ** 2)
radial_prob_max = radial_probability.max()

polar = np.linspace(0, np.pi, 180)
azimuthal = np.linspace(0, 2 * np.pi, 360)
theta, phi = np.meshgrid(polar, azimuthal, indexing="ij")

sph_harm = sp.special.sph_harm_y(l, m, theta, phi)
angular_probability = np.abs(sph_harm) ** 2 * np.sin(theta)
angular_prob_max = angular_probability.max()

while len(scatter_x) < points:
    radial_idx = np.random.choice(len(r_i))

    # Radial test
    if np.random.rand() < radial_probability[radial_idx] / radial_prob_max:

        # Angular test
        polar_idx = np.random.choice(len(polar))
        azimuthal_idx = np.random.choice(len(azimuthal))
        if np.random.rand() < angular_probability[polar_idx][azimuthal_idx] / angular_prob_max:
            if abs(r_i[radial_idx] * np.sin(polar[polar_idx]) * np.sin(azimuthal[azimuthal_idx])) < 1:
                scatter_x.append(r_i[radial_idx] * np.sin(polar[polar_idx]) * np.cos(azimuthal[azimuthal_idx]))
                scatter_y.append(r_i[radial_idx] * np.sin(polar[polar_idx]) * np.sin(azimuthal[azimuthal_idx]))
                scatter_z.append(r_i[radial_idx] * np.cos(polar[polar_idx]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(scatter_x, scatter_y, scatter_z, s=1)
plt.axis("equal")
plt.show()



