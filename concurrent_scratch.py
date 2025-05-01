import asyncio
import numpy as np

queries = np.load('queries.npy')
d, num_quereis = queries.shape
data = np.load('data.npy')
d, n = data.shape

async def matmul(A, B):
    return A @ B

async def main():
    partitions = 10
    chunk_size = n//partitions

    tasks = []
    for p in range(partitions):
        task = asyncio.create_task(matmul(data.T[p*chunk_size: min((p+1)*chunk_size, n)], queries))
        tasks.append(task)
    
    result = np.zeros((n, num_quereis))
    for p in range(partitions):
        task = tasks[p]
        result[p*chunk_size: min((p+1)*chunk_size, n)] = await task

    return result

print(asyncio.run(main()))