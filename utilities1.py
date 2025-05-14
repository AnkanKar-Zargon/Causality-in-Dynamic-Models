import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform

def compute_rank_matrix(data, metric="euclidean"):
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
    Computes the Information Imbalances Delta(A->B) and Delta(B->A) using distances.

    Args:
        data_A (np.ndarray): shape (N,D1)
        data_B (np.ndarray): shape (N,D2)
        k_A (int): number of neighbors for A->B
        k_B (int): number of neighbors for B->A
        metric (str): distance metric

    Returns:
        imb_A_to_B (float): Information Imbalance from A to B
        imb_B_to_A (float): Information Imbalance from B to A
    """
    if data_A.shape[0] != data_B.shape[0]:
        raise ValueError("Datasets must have the same number of samples.")

    N = data_A.shape[0]
    dist_matrix_A = compute_rank_matrix(data_A, metric)
    dist_matrix_B = compute_rank_matrix(data_B, metric)

    nns_A = nns_index_array(dist_matrix_A, k=k_A)
    nns_B = nns_index_array(dist_matrix_B, k=k_B)

    conditional_dists_B = np.zeros((N, k_A))
    for i in range(N):
        conditional_dists_B[i] = dist_matrix_B[i, nns_A[i]]
    conditional_dists_B = conditional_dists_B.flatten()

    conditional_dists_A = np.zeros((N, k_B))
    for i in range(N):
        conditional_dists_A[i] = dist_matrix_A[i, nns_B[i]]
    conditional_dists_A = conditional_dists_A.flatten()
    
    imb_A_to_B = np.mean(conditional_dists_B) 
    imb_B_to_A = np.mean(conditional_dists_A)

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

