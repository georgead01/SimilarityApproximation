import numpy as np

# methods
def reciprocal_rank(ranking, relevant_docs):
    '''
    Computes the Reciprocal Rank for a ranking given the indices relevant documents.

    Attributes:
    - ranking: a list of document indices (integers) order in decreasing predicted relevance, must not be empty
    - relevant_docs: a list of document indices (integers) that are relevant, must not be empty

    Returns:
    - The Reciprocal Rank value (integer, must be in range [0, 1])
    '''
    k = len(ranking)

    for idx in range(k):
        if ranking[idx] in relevant_docs:
            return 1/(idx+1)
        
    return 0

def reciprocal_rank_k(ranking, relevant_docs, k):
    '''
    Computes the Reciprocal Rank @ k for a ranking given the indices relevant documents.

    Attributes:
    - ranking: a list of document indices (integers) order in decreasing predicted relevance, must not be empty
    - relevant_docs: a list of document indices (integers) that are relevant, must not be empty

    Returns:
    - The Reciprocal Rank value (integer, must be in range [0, 1])
    '''
    return reciprocal_rank(ranking[:k], relevant_docs[:k])

def mean_reciprocal_rank_k(rankings, relevant_docs, k):
    '''
    Computes the Mean Reciprocal Rank @ k for a ranking given the indices relevant documents.

    Attributes:
    - rankings: a list of lists of document indices (integers) order in decreasing predicted relevance, must not be empty
    - relevant_docs: a list of lists of document indices (integers) that are relevant, must be of same length as rankings

    Returns:
    - The Mean Reciprocal Rank value (integer, must be in range [0, 1])
    '''
    num_queries = len(rankings)
    assert len(relevant_docs) == num_queries, 'rankings and relevant_docs must be of the same length'

    mrr = 0
    for idx in range(num_queries):
        mrr += reciprocal_rank_k(rankings[idx], relevant_docs[idx], k)

    mrr /= num_queries
    return mrr

def precision_k(ranking, relevant_docs, k):
    '''
    Computes the Precision @ k for a ranking given the indices relevant documents.

    Attributes:
    - ranking: a list of document indices (integers) order in decreasing predicted relevance, must not be empty
    - relevant_docs: a list of document indices (integers) that are relevant, must not be empty

    Returns:
    - The Precision @ k value (integer, must be in range [0, 1])
    '''

    count = 0

    for idx in range(k):
        if ranking[idx] in relevant_docs:
            count += 1
    
    return count/k

def avg_precision_k(ranking, relevant_docs, k):
    '''
    Computes the Average Precision @ k for a ranking given the indices relevant documents.

    Attributes:
    - ranking: a list of document indices (integers) order in decreasing predicted relevance, must not be empty
    - relevant_docs: a list of document indices (integers) that are relevant, must not be empty

    Returns:
    - The AP @ k value (integer, must be in range [0, 1])
    '''

    count = 0
    avg_p = 0

    for idx in range(k):
        if ranking[idx] in relevant_docs:
            count += 1
            avg_p += precision_k(ranking, relevant_docs, idx+1)

    if not count:
        return 0
    return avg_p/count

def mean_precision_k(rankings, relevant_docs, k):
    num_queries = len(rankings)
    assert len(relevant_docs) == num_queries, 'rankings and relevant_docs must be of the same length'

    mp = 0
    for idx in range(num_queries):
        mp += precision_k(rankings[idx], relevant_docs[idx], k)

    mp /= num_queries
    return mp

def mean_avg_precision_k(rankings, relevant_docs, k):
    '''
    Computes the Mean Average Precision @ k for a ranking given the indices relevant documents.

    Attributes:
    - rankings: a list of lists of document indices (integers) order in decreasing predicted relevance, must not be empty
    - relevant_docs: a list of document indices (integers) that are relevant, must not be empty

    Returns:
    - The MAP @ k value (integer, must be in range [0, 1])
    '''

    num_queries = len(rankings)
    assert len(relevant_docs) == num_queries, 'rankings and relevant_docs must be of the same length'

    map = 0
    for idx in range(num_queries):
        map += avg_precision_k(rankings[idx], relevant_docs[idx], k)

    map /= num_queries
    return map

def get_relevant_docs(sim_scores, num_docs):
    '''
    Given a similarity score matrix, where M[i, j] is the similarity between the i-th doc and j-th query, return a list of relevant docs.

    Attributes:
    - sim_scores: a matrix of dimensions (n x num_queries)
    - num_docs: number of desired relevant documents

    Returns:
    - A list of lists of indices (integers) of the k_docs most similar documents for each query sorted from most to least similar.
    '''
    n, num_queries = sim_scores.shape

    result = []

    for q_idx in range(num_queries):
        doc_axis, q_axis = np.nonzero(sim_scores.argsort(axis = 0) >= n-num_docs)
        rel_docs = doc_axis[q_axis == q_idx].tolist()
        rel_docs.sort(key = lambda doc: sim_scores[doc, q_idx], reverse = True)
        result.append(rel_docs)

    return result


# tests
def test_reciprocal_rank():
    print('testing reciprocal_rank:')

    rank1 = [1, 2, 3, 4, 5]
    docs1 = [1, 2, 3, 4, 5]
    assert reciprocal_rank(rank1, docs1) == 1, f'failed test 1'
    print('passed test 1')

    rank2 = [9, 8, 7, 6, 5]
    docs2 = [1, 2, 3, 4, 5]
    assert reciprocal_rank(rank2, docs2) == 0.2, f'failed test 2'
    print('passed test 2')

    rank3 = [9, 8, 7, 6]
    docs3 = [1, 2, 3, 4, 5]
    assert reciprocal_rank(rank3, docs3) == 0, f'failed test 3'
    print('passed test 3')

    rank4 = [9, 8, 7, 6, 10]
    docs4 = [9, 8, 7, 6, 10]
    assert reciprocal_rank(rank4, docs4) == 1, f'failed test 4'
    print('passed test 4')

def test_reciprocal_rank_k():
    print('testing reciprocal_rank_k:')

    rank1 = [1, 2, 3, 4, 5]
    docs1 = [4, 5]
    k1 = 3
    assert reciprocal_rank_k(rank1, docs1, k1) == 0, f'failed test 1'
    print('passed test 1')

    rank2 = [1, 2, 3, 4, 5]
    docs2 = [4, 5]
    k2 = 4
    assert reciprocal_rank_k(rank2, docs2, k2) == 0.25, f'failed test 2'
    print('passed test 2')

def test_mean_reciprocal_rank_k():

    print('testing mean_reciprocal_rank_k:')

    rank1 = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    docs1 = [[4, 5], [2, 3]]
    k1 = 3
    assert mean_reciprocal_rank_k(rank1, docs1, k1) == 0.25, f'failed test 1'
    print('passed test 1')

    rank2 = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    docs2 = [[4, 5], [1, 2]]
    k2 = 2
    assert mean_reciprocal_rank_k(rank2, docs2, k2) == 0, f'failed test 2'
    print('passed test 2')

    rank3 = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    docs3 = [[1, 2], [4, 5]]
    k3 = 2
    assert mean_reciprocal_rank_k(rank3, docs3, k3) == 1, f'failed test 2'
    print('passed test 3')

def test_get_relevant_docs():

    print('testing get_relevant_docs:')

    sim_scores1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.2], [0.3, 0.2, 0.1]])
    k_docs1 = 1
    assert get_relevant_docs(sim_scores1, k_docs1) == [[2], [1], [0]], 'failed test 1'
    print('passed test 1')

    sim_scores2 = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.2], [0.3, 0.3, 0.1]])
    k_docs2 = 2
    assert get_relevant_docs(sim_scores2, k_docs2) == [[2, 1], [1, 2], [0, 1]]
    print('passed test 2')

    sim_scores3 = np.array([[0.3], [0.2], [0.3]])
    k_docs3 = 1
    docs3 = get_relevant_docs(sim_scores3, k_docs3)[0]
    assert 0 in docs3 or 2 in docs3, 'failed test 3'
    print('passed test 3')

    sim_scores4 = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.2], [0.3, 0.3, 0.1]])
    k_docs4 = 3
    assert get_relevant_docs(sim_scores4, k_docs4) == [[2, 1, 0], [1, 2, 0], [0, 1, 2]], 'failed test 4'
    print('passed test 4')


def test_precision_k():
    print('testing precision_k:')

    rank1 = [1, 2, 4, 5]
    docs1 = [4, 5]
    k1 = 3
    assert precision_k(rank1, docs1, k1) == 1/3, f'failed test 1'
    print('passed test 1')

    rank2 = [1, 2, 4, 5]
    docs2 = [4, 5]
    k2 = 2
    assert precision_k(rank2, docs2, k2) == 0, f'failed test 2'
    print('passed test 2')

    rank3 = [4, 5, 1, 2]
    docs3 = [4, 5, 3]
    k3 = 2
    assert precision_k(rank3, docs3, k3) == 1, f'failed test 3'
    print('passed test 3')

def test_avg_precision_k():
    print('testing avg_precision_k:')

    rank1 = [1, 2, 4, 5]
    docs1 = [4, 5]
    k1 = 3
    assert avg_precision_k(rank1, docs1, k1) == 1/3, f'failed test 1'
    print('passed test 1')

    rank2 = [1, 2, 4, 5]
    docs2 = [4, 5]
    k2 = 2
    assert avg_precision_k(rank2, docs2, k2) == 0, f'failed test 2'
    print('passed test 2')

    rank3 = [4, 5, 1, 2]
    docs3 = [4, 5, 3]
    k3 = 2
    assert avg_precision_k(rank3, docs3, k3) == 1, f'failed test 3'
    print('passed test 3')

    rank4 = [4, 5, 1, 2]
    docs4 = [4, 5, 3]
    k4 = 3
    assert avg_precision_k(rank4, docs4, k4) == 1, f'failed test 4'
    print('passed test 4')

def test_mean_avg_precision_k():
    epsilon = 10**-6

    print('testing mean_avg_precision_k:')

    rank1 = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    docs1 = [[4, 5], [2, 3]]
    k1 = 3
    assert abs(7/24-mean_avg_precision_k(rank1, docs1, k1)) < epsilon, f'failed test 1'
    print('passed test 1')

    rank2 = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    docs2 = [[4, 5], [1, 2]]
    k2 = 2
    assert mean_avg_precision_k(rank2, docs2, k2) == 0, f'failed test 2'
    print('passed test 2')

    rank3 = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    docs3 = [[1, 2], [4, 5]]
    k3 = 2
    assert mean_avg_precision_k(rank3, docs3, k3) == 1, f'failed test 2'
    print('passed test 3')



if __name__ == '__main__':
    test_reciprocal_rank()
    test_reciprocal_rank_k()
    test_mean_reciprocal_rank_k()
    test_get_relevant_docs()
    test_precision_k()
    test_avg_precision_k()
    test_mean_avg_precision_k()