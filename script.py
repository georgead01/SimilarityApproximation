import numpy as np
import matplotlib.pyplot as plt
import time
from evaluation import mean_reciprocal_rank_k, get_relevant_docs

r_vals = [50, 100, 150, 200, 250]

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
#target = matrix_multi(data.T, queries)
target = data.T@queries
full_time = time.time()-start_time
print(f'full time: {full_time} sec')

k_docs = 100
rel_docs = get_relevant_docs(target, k_docs)

mrr = {r: [] for r in r_vals}
times = {r: 0 for r in r_vals}
for r in r_vals:
    LT = np.load(f'compressed/LT_{r}.npy')
    R = np.load(f'compressed/R_{r}.npy')

    print(f'LT_{r} shape: {LT.shape}')
    print(f'R_{r} shape: {R.shape}')

    start_time = time.time()
    #output = matrix_multi(R, matrix_multi(LT, queries))
    output = R @ LT @ queries
    exec_time = time.time()-start_time
    rel_time = exec_time/full_time
    times[r] = rel_time

    print(f'r = {r}, relative time: {rel_time}, time: {exec_time} sec')

    rankings = get_relevant_docs(output, k_docs)
    for k in [i*5 for i in range(1, 20)]:
        # for q_idx in range(num_quereis):
        #     ranked = list(np.nonzero(output[:, q_idx].argsort(axis = 0) >= n-k)[0])
        #     ranked.sort(key=lambda doc: output[doc][q_idx], reverse=True)
        #     rankings.append(ranked)
        #     docs = np.nonzero(target[:, q_idx].argsort(axis = 0) >= n-k_docs)[0]
        #     rel_docs.append(docs)
        mrr_k = mean_reciprocal_rank_k(rankings, rel_docs, k)
        print(f'k: {k}, MRR @ k = {mrr_k}')
        mrr[r].append(mrr_k)
    
plt.title(f'MRR @ k vs. k')
for r in r_vals:
    plt.plot([i*5 for i in range(1, 20)], mrr[r], label = f'r = {r}')
plt.legend()
plt.show()