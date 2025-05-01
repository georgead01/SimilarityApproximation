import numpy as np
import matplotlib.pyplot as plt

d = 300
num_queries = 100

queries = np.random.normal(0, 1, (d, num_queries))
train_queries = np.random.normal(0, 1, (d, num_queries))

queries /= np.linalg.norm(queries, axis = 0)
train_queries /= np.linalg.norm(train_queries, axis = 0)

np.save('queries.npy', queries)
np.save('train_queries.npy', train_queries)

plt.title(f'queries (d = {d}, |Q| = {num_queries})')
plt.imshow(queries, aspect='auto')
plt.show()