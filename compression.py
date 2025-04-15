import numpy as np

data = np.load('data.npy')
d, n = data.shape

U, sigma, VT = np.linalg.svd(data, full_matrices=False)
sigma = sigma.reshape((d, 1))

print(f'shape of U: {U.shape}')
print(f'shape of sigma: {sigma.shape}')
print(f'shape of V: {VT.shape}')

r_vals = [10, 20, 50, 75, 100]


for r in r_vals:
    L_r = U[:, :r]
    R_r = sigma[:r] * VT[:r, :]

    np.save(f'compressed/L_{r}.npy', L_r)
    np.save(f'compressed/R_{r}.npy', R_r.T)