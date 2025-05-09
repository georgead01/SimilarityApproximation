import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from evaluation import mean_reciprocal_rank_k, get_relevant_docs, mean_avg_precision_k, mean_precision_k

class LowRankApprox(nn.Module):
    def __init__(self, d, n, rank):
        super().__init__()
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(rank, n, bias=False)
        nn.init.normal_(self.U.weight, mean=0, std=1.0)
        nn.init.normal_(self.V.weight, mean=0, std=1.0)
        self.activation = nn.Softmax()

    def forward(self, x):
        return self.V(self.U(x))
    
if __name__ == '__main__':
    train_queries = torch.from_numpy(np.load('train_queries.npy')).float()
    print(train_queries.dtype)
    d, num_train = train_queries.shape
    queries = np.load('queries.npy')
    d, num_quereis = queries.shape
    data = np.load('data.npy')
    #data = plt.imread('cat.jpg').mean(axis = -1)
    #print(data.shape)
    d, n = data.shape
    #data /= np.linalg.norm(data, axis = 0)
    #print(data.max())

    #plt.imshow(data)
    #plt.show()

    r_vals = [50, 100, 150, 200, 250]
    #r_vals = [300]
    #r_vals = [10, 50, 100, 500, 1000, 1500]

    k_docs = n//100
    rank_target = data.T@queries
    rel_docs = get_relevant_docs(rank_target, k_docs)

    mrrs = {}
    maps = {}
    errs = []
    for r in r_vals:
        model = LowRankApprox(d, n, r)
        lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        
        #loss_fn = nn.MultiLabelMarginLoss()
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        #loss_fn = nn.MSELoss(reduction='sum')
        #loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

        #target = torch.from_numpy(data.T).float()@train_queries
        target = torch.from_numpy(data.T).float()
        #print(target.shape)
        #target = torch.from_numpy(data.T).argsort(dim = 0)

        epochs = 1000
        for epoch in range(epochs):
            #output = model(train_queries.T)
            optimizer.zero_grad()
            # output =
            output = model(torch.from_numpy(np.eye(d)).float())
            #print(output.shape)
            #loss = loss_fn(output.argsort(dim = 1), target.T)
            #print(output.shape)
            loss = loss_fn(output, target.T)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

        U_matrix = model.U.weight.detach().numpy()
        V_matrix = model.V.weight.detach().numpy()
        print(f'U_matrix shape: {U_matrix.shape}')
        print(f'V_matrix shape: {V_matrix.shape}')

        output = V_matrix @ U_matrix @ queries
        #output = V_matrix @ U_matrix
        #plt.imshow(output)
        #plt.show()

        LT = np.load(f'compressed/LT_{r}.npy')
        R = np.load(f'compressed/R_{r}.npy')

        #svd_output = R @ LT @ queries

        #svd_err = np.linalg.norm(svd_output-output)
        #svd_err = 0
        #errs.append(svd_err)

        rankings = get_relevant_docs(output, k_docs)

        mrrs[r] = []
        maps[r] = []
        for k in [i*5 for i in range(1, 21)]:
            # for q_idx in range(num_quereis):
            #     ranked = list(np.nonzero(output[:, q_idx].argsort(axis = 0) >= n-k)[0])
            #     ranked.sort(key=lambda doc: output[doc][q_idx], reverse=True)
            #     rankings.append(ranked)
            #     docs = np.nonzero(target[:, q_idx].argsort(axis = 0) >= n-k_docs)[0]
            #     rel_docs.append(docs)
            mrr_k = mean_reciprocal_rank_k(rankings, rel_docs, k)
            map_k = mean_precision_k(rankings, rel_docs, k)
            # mrr_k = 0
            print(f'k: {k}, MRR @ k = {mrr_k}')
            mrrs[r].append(mrr_k)
            print(f'k: {k}, MAP @ k = {map_k}')
            maps[r].append(map_k)
    
    #plt.title(f'||SVD approx. - MSE trained approx.|| vs r')
    #plt.xlabel('r')
    #plt.ylabel('diff. norm')
    #plt.plot(r_vals, errs)
    #plt.legend()
    #plt.show()

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
        plt.plot(k_vals, mrrs[r], label = f'r = {r}')
    plt.legend()
    plt.show()

    plt.title(f'Precision @ k vs. k')
    plt.hlines(k_docs/n, 0, max(k_vals), 'r', '--', label = 'benchmark')
    for r in r_vals:
        plt.plot(k_vals, maps[r], label = f'r = {r}')
    plt.legend()
    plt.show()