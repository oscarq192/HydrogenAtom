import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

L = 1
hbar = 1
m = 1
N = 100
h = L / (N + 1)

H = np.zeros((N, N))
np.fill_diagonal(H, (hbar ** 2) / (m * h ** 2))

i = np.arange(N - 1)
H[i, i + 1] = - (hbar ** 2) / (2 * m * h ** 2)
H[i + 1, i] = - (hbar ** 2) / (2 * m * h ** 2)

print(H)

values, vectors = sp.linalg.eigh(H)

x = np.linspace(h, L - h, N)
plt.plot(x, vectors[:, 0])
plt.plot(x, 10 * abs(vectors[:, 0]) ** 2)
plt.show()

