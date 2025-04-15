import numpy as np
import matplotlib.pyplot as plt

d = 100
n = 10000

data = np.random.normal(0, 1, (d, n))
np.save('data.npy', data)

plt.title(f'data (d = {d}, n = {n})')
plt.imshow(data, aspect = 'auto')
plt.show()