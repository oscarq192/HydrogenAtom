import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyvista as pv

class QuantumCloud:
    def __init__(self, n=1, l=0, m=0, points=100000, N=10000, r_min=1e-6, r_max=35):
        self.phase = None
        self.colour_prob = None
        self.angular_wavefunction = None
        self.radial_wavefunction = None
        self.radii = None
        self.theta = None
        self.phi = None
        self.x = None
        self.y = None
        self.z = None
        self.radial_cdf = None
        self.angular_cdf = None
        self.R = None
        self.r = None
        self.thetas = None
        self.phis = None
        self.n = n
        self.l = l
        self.m = m
        self.points = points

        self.N = N
        self.r_min = r_min
        self.r_max = r_max

        self.calculate()

    # Radial equation finite difference solver

    def solve_radial_equation(self):
        h = (self.r_max - self.r_min) / (self.N + 1)
        n_r = self.n - self.l - 1  # Radial quantum number, specifying number of radial nodes

        r = np.linspace(self.r_min, self.r_max, self.N + 2)  # Radial indexes
        r_i = r[1:-1]

        diagonal = ((1 / h ** 2)
                    - (1 / r_i)
                    + (self.l * (self.l + 1) / (2 * r_i ** 2)))
        off_diagonal = - 1 / (2 * h ** 2) * np.ones(self.N - 1)

        values, vector = sp.linalg.eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(n_r, n_r),
            check_finite=False
        )
        R = vector / r_i

        return r_i, R[:, 0]

    def calculate_radial_probability(self, r, R):
        radial_probability = np.abs(R) ** 2 * (r ** 2)
        radial_probability /= np.sum(radial_probability)

        radial_cdf = np.cumsum(radial_probability)

        return radial_cdf

    # Angular equation solver

    def calculate_angular_distribution(self):
        polar = np.linspace(0, np.pi, 180)
        azimuthal = np.linspace(0, 2 * np.pi, 360)
        theta, phi = np.meshgrid(polar, azimuthal, indexing="ij")  # Angular indexes

        sph_harm = sp.special.sph_harm_y(self.l, self.m, theta, phi)

        angular_probability = np.abs(sph_harm) ** 2 * np.sin(theta)
        angular_probability /= np.sum(angular_probability)

        angular_cdf = np.cumsum(angular_probability.ravel())

        return theta, phi, angular_cdf

    # Distribution Sampling

    def sample_points(self, r, theta, phi, radial_cdf, angular_cdf):
        u_radial = np.random.rand(self.points)
        u_angular = np.random.rand(self.points)

        point_radii = r[np.searchsorted(radial_cdf, u_radial)]
        angle_indices = np.searchsorted(angular_cdf, u_angular)
        point_thetas = theta.ravel()[angle_indices]
        point_phis = phi.ravel()[angle_indices]

        x = point_radii * np.sin(point_thetas) * np.cos(point_phis)
        y = point_radii * np.sin(point_thetas) * np.sin(point_phis)
        z = point_radii * np.cos(point_thetas)

        return x, y, z, point_radii, point_thetas, point_phis

    # Coordinate all the maths above

    def calculate(self):
        self.r, self.R = self.solve_radial_equation()

        self.radial_cdf = self.calculate_radial_probability(self.r, self.R)

        self.theta, self.phi, self.angular_cdf = self.calculate_angular_distribution()

        (self.x,
         self.y,
         self.z,
         self.radii,
         self.thetas,
         self.phis) = self.sample_points(self.r, self.theta, self.phi, self.radial_cdf, self.angular_cdf)

        self.radial_wavefunction = np.interp(self.radii, self.r, self.R)
        self.angular_wavefunction = sp.special.sph_harm_y(self.l, self.m, self.thetas, self.phis)

        self.colour_prob = np.abs(self.angular_wavefunction) ** 2 * np.abs(self.radial_wavefunction) ** 2
        self.phase = np.angle(self.angular_wavefunction)

    # PyVista rendering

    def create_cloud(self, mask_enabled=False):
        x, y, z = self.x, self.y, self.z
        colour_prob = self.colour_prob
        phase = self.phase

        if mask_enabled:
            mask = np.abs(y) < 1

            x = x[mask]
            y = y[mask]
            z = z[mask]
            colour_prob = colour_prob[mask]
            phase = phase[mask]

        scatter = np.column_stack((x, y, z))
        cloud = pv.PolyData(scatter)
        cloud["probability"] = colour_prob
        cloud["phase"] = phase

        return cloud


cloud.plot(scalars="probability", render_points_as_spheres=True, point_size=3, cmap="plasma")





# Monte Carlo rejection sampling

# angular_prob_max = angular_probability.max()        # Denominator of fraction for probability
# radial_prob_max = radial_probability.max()      # Denominator of fraction for rejection sampling

# x = []
# y = []
# z = []

# while len(x) < points:
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
#                 x.append(r_i[radial_index] * np.sin(polar[polar_idx]) * np.cos(azimuthal[azimuthal_idx]))
#                 y.append(r_i[radial_index] * np.sin(polar[polar_idx]) * np.sin(azimuthal[azimuthal_idx]))
#                 z.append(r_i[radial_index] * np.cos(polar[polar_idx]))

# Matplotlib rendering

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(x, y, z, s=1)
# plt.axis("equal")
# plt.show()

