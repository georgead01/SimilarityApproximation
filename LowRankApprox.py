import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from evaluation import mean_reciprocal_rank_k, get_relevant_docs

class LowRankApprox(nn.Module):
    def __init__(self, d, n, rank):
        super().__init__()
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(rank, n, bias=False)

    def forward(self, x):
        return self.V(self.U(x))
    
if __name__ == '__main__':
    train_queries = torch.from_numpy(np.load('train_queries.npy')).float()
    print(train_queries.dtype)
    d, num_train = train_queries.shape
    queries = np.load('queries.npy')
    d, num_quereis = queries.shape
    data = np.load('data.npy')
    d, n = data.shape

    #r_vals = [50, 100, 150, 200, 250]
    r_vals = [300]

    k_docs = 100
    rank_target = data.T@queries
    rel_docs = get_relevant_docs(rank_target, k_docs)

    mrrs = {}
    for r in r_vals:
        model = LowRankApprox(d, n, r)
        lr = 0.05
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        #loss_fn = nn.MultiLabelMarginLoss()
        #loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.MSELoss()

        #target = torch.from_numpy(data.T).float()@train_queries
        target = torch.from_numpy(data.T).float().argsort(dim = 0)

        epochs = 1000
        for epoch in range(epochs):
            #output = model(train_queries.T)
            output = model(torch.from_numpy(np.eye(d)).float())
            loss = loss_fn(output.argsort(dim = 1), target.T)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

        U_matrix = model.U.weight.detach().numpy()
        V_matrix = model.V.weight.detach().numpy()
        print(f'U_matrix shape: {U_matrix.shape}')
        print(f'V_matrix shape: {V_matrix.shape}')

        output = V_matrix @ U_matrix @ queries
        rankings = get_relevant_docs(output, k_docs)

        mrrs[r] = []
        for k in [i*5 for i in range(1, 21)]:
            # for q_idx in range(num_quereis):
            #     ranked = list(np.nonzero(output[:, q_idx].argsort(axis = 0) >= n-k)[0])
            #     ranked.sort(key=lambda doc: output[doc][q_idx], reverse=True)
            #     rankings.append(ranked)
            #     docs = np.nonzero(target[:, q_idx].argsort(axis = 0) >= n-k_docs)[0]
            #     rel_docs.append(docs)
            mrr_k = mean_reciprocal_rank_k(rankings, rel_docs, k)
            print(f'k: {k}, MRR @ k = {mrr_k}')
            mrrs[r].append(mrr_k)
    
    plt.title(f'MRR @ k vs. k')
    for r in r_vals:
        plt.plot(mrrs[r], label = f'r = {r}')
    plt.legend()
    plt.show()