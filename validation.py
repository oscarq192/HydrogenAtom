import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

n = 1
l = 0
m = 0
r_min = 1e-6
r_max = 35
N = (np.geomspace(100, 10000, 100)).astype(int)
H = np.zeros_like(N, dtype=float)
E_fd = np.zeros_like(N, dtype=float)


for i, j in enumerate(N):
    h = (r_max - r_min) / (j + 1)
    H[i] = h
    n_r = n - l - 1  # Radial quantum number, specifying number of radial nodes

    r = np.linspace(r_min, r_max, j + 2)  # Radial indexes
    r_i = r[1:-1]

    diagonal = ((1 / h ** 2)
                - (1 / r_i)
                + (l * (l + 1) / (2 * r_i ** 2)))
    off_diagonal = - 1 / (2 * h ** 2) * np.ones(j - 1)

    values, vector = sp.linalg.eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(n_r, n_r),
        check_finite=False
    )

    E_fd[i] = values[0]

E_n = -1 / (2 * n ** 2)

error = np.abs(E_fd - E_n)

plt.loglog(H, error, label="Finite difference error")

reference = error[0] * (H / H[0])**2
plt.loglog(H, reference, "--", label=r"$O(h^2)$", color="r")

plt.xlabel("Grid spacing (h)")
plt.ylabel("Absolute energy error")
plt.title("Finite Difference Convergence of the Energy Eigenvalue")

gradient, intercept = np.polyfit(np.log(H), np.log(error), 1)
fit = np.exp(intercept) * H ** gradient
plt.loglog(H, fit, ":", label=f"Fit: slope = {gradient:.2f}", color="g")

plt.legend()
plt.show()

