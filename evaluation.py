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

# methods
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

if __name__ == '__main__':
    test_reciprocal_rank()
    test_reciprocal_rank_k()
    test_mean_reciprocal_rank_k()