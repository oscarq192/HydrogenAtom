import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Radial equation finite difference solver
N = 10000
r_min = 1e-6
r_max = 100
h = (r_max - r_min) / (N + 1)

n = 3
l = 1
m = 0
n_r = n - l - 1     # Radial quantum number, specifying number of radial nodes

r = np.linspace(r_min, r_max, N + 2)    # Radial indexes
r_i = r[1:-1]

diagonal = (1 / h ** 2) - (1 / r_i) + (l * (l + 1) / (2 * r_i ** 2))
off_diagonal = - 1 / (2 * h ** 2) * np.ones(N - 1)

values, vector = sp.linalg.eigh_tridiagonal(diagonal,
                                             off_diagonal,
                                             select="i",
                                             select_range=(n_r, n_r),
                                             check_finite=False)
R = vector / r_i

radial_probability = np.abs(R[:, 0]) ** 2 * (r_i ** 2)
total_radial_probability = np.sum(radial_probability)

normalised_radial_probability = radial_probability / total_radial_probability
radial_cdf = np.cumsum(normalised_radial_probability)

polar = np.linspace(0, np.pi, 180)
azimuthal = np.linspace(0, 2 * np.pi, 360)
theta, phi = np.meshgrid(polar, azimuthal, indexing="ij")   # Angular indexes

sph_harm = sp.special.sph_harm_y(l, m, theta, phi)
angular_probability = np.abs(sph_harm) ** 2 * np.sin(theta)

total_angular_probability = np.sum(angular_probability)

normalised_angular_probability = angular_probability / total_angular_probability
angular_cdf = np.cumsum(normalised_angular_probability.ravel())

points = 10000

# Monte Carlo rejection sampling

# angular_prob_max = angular_probability.max()        # Denominator of fraction for probability
# radial_prob_max = radial_probability.max()      # Denominator of fraction for rejection sampling

# scatter_x = []
# scatter_y = []
# scatter_z = []

# while len(scatter_x) < points:
#     radial_index = np.random.choice(len(r_i))
#
#     # Radial test
#     if np.random.rand() < radial_probability[radial_index] / radial_prob_max:
#
#         # Angular test
#         polar_idx = np.random.choice(len(polar))
#         azimuthal_idx = np.random.choice(len(azimuthal))
#         if np.random.rand() < angular_probability[polar_idx][azimuthal_idx] / angular_prob_max:
#             if abs(r_i[radial_index] * np.sin(polar[polar_idx]) * np.sin(azimuthal[azimuthal_idx])) < 1:
#                 scatter_x.append(r_i[radial_index] * np.sin(polar[polar_idx]) * np.cos(azimuthal[azimuthal_idx]))
#                 scatter_y.append(r_i[radial_index] * np.sin(polar[polar_idx]) * np.sin(azimuthal[azimuthal_idx]))
#                 scatter_z.append(r_i[radial_index] * np.cos(polar[polar_idx]))

# Distribution Sampling

u_radial = np.random.rand(points)
u_angular = np.random.rand(points)

point_radii = np.searchsorted(radial_cdf, u_radial)
angle_indices = np.searchsorted(angular_cdf, u_angular)
point_thetas = theta.ravel()[angle_indices]
point_phis = phi.ravel()[angle_indices]

scatter_x = point_radii * np.sin(point_thetas) * np.cos(point_phis)
scatter_y = point_radii * np.sin(point_thetas) * np.sin(point_phis)
scatter_z = point_radii * np.cos(point_thetas)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(scatter_x, scatter_y, scatter_z, s=1)
plt.axis("equal")
plt.show()



