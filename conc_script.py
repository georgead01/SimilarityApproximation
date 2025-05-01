import numpy as np
import matplotlib.pyplot as plt
import time
from evaluation import mean_reciprocal_rank_k, get_relevant_docs

queries = np.load('queries.npy')
d, num_quereis = queries.shape
data = np.load('data.npy')
d, n = data.shape

def matrix_multi(A, B):
    m, _ = A.shape
    p, n = B.shape

    result = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            for k in range(p):
                result[i][j] += A[i,k]*B[k,j]

    return result
    
start_time = time.time()
target = matrix_multi(data.T, queries)
# target = data.T@queries
full_time = time.time()-start_time
print(f'full time: {full_time} sec')