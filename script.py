import numpy as np
import matplotlib.pyplot as plt
import time
from evaluation import mean_reciprocal_rank_k, get_relevant_docs, mean_avg_precision_k

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

k_docs = n//100
rel_docs = get_relevant_docs(target, k_docs)

mrr = {r: [] for r in r_vals}
map = {r: [] for r in r_vals}
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
    for k in [i*5 for i in range(1, 21)]:
        # for q_idx in range(num_quereis):
        #     ranked = list(np.nonzero(output[:, q_idx].argsort(axis = 0) >= n-k)[0])
        #     ranked.sort(key=lambda doc: output[doc][q_idx], reverse=True)
        #     rankings.append(ranked)
        #     docs = np.nonzero(target[:, q_idx].argsort(axis = 0) >= n-k_docs)[0]
        #     rel_docs.append(docs)
        mrr_k = mean_reciprocal_rank_k(rankings, rel_docs, k)
        map_k = mean_avg_precision_k(rankings, rel_docs, k)
        print(f'k: {k}, MRR @ k = {mrr_k}')
        print(f'k: {k}, MAP @ k = {map_k}')
        mrr[r].append(mrr_k)
        map[r].append(map_k)

k_vals = [i*5 for i in range(1, 21)]
probs = [1]
mrr_bench = []

for k in k_vals:
    bench = 0
    for i in range(1, k+1):
        prob = probs[i-1]
        probs.append(prob*(1-k_docs/(n-i+1)))
        bench += (1/i)*prob*(k_docs/(n-i+1))
    mrr_bench.append(bench)
    
plt.title(f'MRR @ k vs. k')
plt.plot(k_vals, mrr_bench, 'r--', label = 'benchmark')
for r in r_vals:
    plt.plot(k_vals, mrr[r], label = f'r = {r}')
plt.legend()
plt.show()

plt.title(f'Precision @ k vs. k')
plt.hlines(k_docs/n, 0, max(k_vals), 'r', '--', label = 'benchmark')
for r in r_vals:
    plt.plot(k_vals, map[r], label = f'r = {r}')
plt.legend()
plt.show()