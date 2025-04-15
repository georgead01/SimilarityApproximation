import numpy as np
import matplotlib.pyplot as plt

d = 100
num_queries = 100

queries = np.random.normal(0.5, 1, (d, num_queries))
queries[:, np.random.choice(num_queries, num_queries//2)] = np.random.normal(-0.5, 1, (d, num_queries//2))

np.save('queries.npy', queries)

plt.title(f'queries (d = {d}, |Q| = {num_queries})')
plt.imshow(queries, aspect='auto')
plt.show()