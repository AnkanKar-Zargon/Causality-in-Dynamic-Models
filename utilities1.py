import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform

def compute_rank_matrix_a(data, metric="euclidean"):
    """
    Computes the pairwise distance matrix of the input data.

    Args:
        data (np.ndarray): array of shape (N,D), with N points and D features
        metric (str, default="euclidean"): the distance metric to use

    Returns:
        distance_maatrix (np.ndarray): array of shape (N,N), where entry [i,j]
                                      is the distance between point i and point j.
                                      Diagonal entries are set to np.inf.
    """
    distance_matrix = squareform(pdist(data, metric=metric))
    np.fill_diagonal(distance_matrix, np.inf)
    return distance_matrix

def nns_index_array(distance_matrix, k=1):
    """
    Computes the indices of the k nearest neighbors using a distance matrix.

    Args:
        distance_matrix (np.ndarray): array of shape (N,N), pairwise distances with np.inf on diagonal

        k (int): number of nearest neighbors to find

    Returns:
        NNs (np.ndarray): array of shape (N,k), where each row contains indices of the k nearest neighbors
    """
    N = distance_matrix.shape[0]
    NNs = np.zeros((N, k), dtype=int)
    for i in range(N):
        NNs[i, :] = np.argpartition(distance_matrix[i], np.arange(k))[:k]
    return np.array(NNs)

def compute_info_imbalance(data_A, data_B, k_A=1, k_B=1, metric="euclidean"):
    """
    Computes the Information Imbalances Delta(A->B) and Delta(B->A)

    Args:
        data_A (np.ndarray): array of shape (N,D1) with N points and D1 features

        data_A (np.ndarray): array of shape (N,D2) with N points and D2 features

        k_A (int, default=1): number of nearest neighbor to compute Delta(A->B)

        k_B (int, default=1): number of nearest neighbor to compute Delta(B->A)

        metric (str, default="euclidean"): name of distance employed

    Returns:
        imb_A_to_B (float): Information Imbalance Delta(A->B)

        imb_B_to_A (float): Information Imbalance Delta(B->A)
    """
    if data_A.shape[0] != data_B.shape[0]:
        raise ValueError("Number of points must be the same in the two representations!")
    N = data_A.shape[0]
    rank_matrix_A = compute_rank_matrix_a(data_A, metric=metric)
    rank_matrix_B = compute_rank_matrix_a(data_B, metric=metric)

    # Find the nn indices in each space
    nns_A = nns_index_array(rank_matrix_A, k=k_A)
    nns_B = nns_index_array(rank_matrix_B, k=k_B)

    # Find conditional ranks in two spaces
    conditional_ranks_B = np.zeros((N, k_A))
    for i_point in range(N):
        rank_B = rank_matrix_B[i_point][nns_A[i_point]]
        conditional_ranks_B[i_point] = rank_B
    conditional_ranks_B = conditional_ranks_B.flatten()

    conditional_ranks_A = np.zeros((N, k_B))
    for i_point in range(N):
        rank_A = rank_matrix_A[i_point][nns_B[i_point]]
        conditional_ranks_A[i_point] = rank_A
    conditional_ranks_A = conditional_ranks_A.flatten()

    # The information imbalances:
    imb_A_to_B = 2/N * np.mean(conditional_ranks_B)
    imb_B_to_A = 2/N * np.mean(conditional_ranks_A)

    return imb_A_to_B, imb_B_to_A

def construct_time_delay_embedding(X, E, tau_e, sample_times=None):
    """
    Computes the time-delay embeddings of X, with embedding length E and embedding time tau_e

    Args:
        X (np.ndarray): one-dimensional array with N points

        E (int): embedding dimension

        tau_e (int): embedding time

        sample_times (np.ndarray or list(int), default=None):
            whether to construct one embedding for each point in X (if sample_times==None)
            or to use only the points in the array sample_times (if sample_times!=None)
    Returns:
        X_time_delay (np.ndarray): time-delay embedding of X, with shape (N,E)
    """

    if sample_times is None:
        N = len(X)
        start_time = E*tau_e

        X_time_delay = np.zeros((N-start_time, E))
        X_time_delay[:, 0] = X[start_time:]
        for i_dim in range(1, E):
            X_time_delay[:, i_dim] = X[start_time-i_dim*tau_e:-i_dim*tau_e]

    else:
        N = len(sample_times)
        X_time_delay = np.zeros((N, E))

        for i_sample in range(N):
            embedding_times = np.arange(sample_times[i_sample],
                                        sample_times[i_sample] - tau_e*E,
                                        -tau_e)

            X_time_delay[i_sample, :] = X[embedding_times]

    return X_time_delay


def compute_pearson_correlation(X, Y):
    """
    Computes the Pearson correlation coefficient between X and Y

    Args:
        X (np.ndarray): one-dimensional array

        Y (np.ndarray): one-dimensional array

    Returns:
        rho (float): Pearson correlation rho(X,Y)
    """
    X_standardized = (X - np.mean(X)) / np.std(X)
    Y_standardized = (Y - np.mean(Y)) / np.std(Y)
    rho = np.mean(X_standardized*Y_standardized)
    return rho

