import numpy as np
import matplotlib.pyplot as plt

data = np.load('data.npy')
d, n = data.shape

U, sigma, VT = np.linalg.svd(data, full_matrices=False)
sigma = sigma.reshape((d, 1))

print(f'shape of U: {U.shape}')
print(f'shape of sigma: {sigma.shape}')
print(f'shape of V: {VT.shape}')

r_vals = [50, 100, 150, 200, 250, 300]


for r in r_vals:
    L_r = U[:, :r]
    RT_r = sigma[:r] * VT[:r, :]

    np.save(f'compressed/L_{r}.npy', L_r)
    np.save(f'compressed/LT_{r}.npy', L_r.T)
    np.save(f'compressed/R_{r}.npy', RT_r.T)

data = plt.imread('cat.jpg').mean(axis = -1)
data /= np.linalg.norm(data, axis = 0)
d, n = data.shape

U, sigma, VT = np.linalg.svd(data, full_matrices=False)
sigma = sigma.reshape((d, 1))

print(f'shape of U: {U.shape}')
print(f'shape of sigma: {sigma.shape}')
print(f'shape of V: {VT.shape}')

r_vals = [10, 50, 100, 500, 1000, 1500]


for r in r_vals:
    L_r = U[:, :r]
    RT_r = sigma[:r] * VT[:r, :]

    plt.title(f'{r}-rank approx.')
    plt.imshow(L_r @ RT_r)
    plt.show()