import numpy as np
import scipy
import powerlaw
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix, diags
from random import randint, sample
import networkx as nx
import time
import itertools
import math
import warnings
warnings.filterwarnings("ignore")
import random


def generate_PowerLaw1(N):  
    """
    This function creates a random network topology where the out-degree of nodes follows a power-law distribution. 
    The generation process avoids self-loops but allows asymmetric directed edges.
    
    Parameters:
    N (int): Number of nodes in the network.
    
    Returns:
    W : Row-normalized spatial weight matrix of shape (N, N).
    """
    #powerlaw.Power_Law(1,2.5).generate_random(N)
    #df=np.ceil(np.random.exponential(10,N).reshape(N,1))
    df=np.ceil(powerlaw.Power_Law(xmin=1,parameters=[7]).generate_random(N))
    #df=np.floor(powerlaw.Power_Law(xmin=1,parameters=[2.5]).generate_random(N))
    df[df>N]=N
    R=np.zeros((int(sum(df)),1))
    s1,s2=0,0
    for i in range(N):
        s1=s2
        s2=s2+int(df[i])
        R[s1:s2]=i
    C=np.zeros(())
    for i in range(N):
        pos=np.random.permutation(N).reshape(N,1)
        C=np.vstack((C,pos[0:int(df[i])]))
    C=C[1:]
    rr=R[R!=C]
    cc=C[R!=C]
    l=np.ones((rr.shape))
    A=sps.coo_matrix((l,(rr,cc)),shape=(N,N))
    Asum = np.array(A.sum(1))
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = sps.diags(r_inv)
    W = r_mat_inv.dot(A)
    
    return W


def generate_Dyad(N):
    """
    Generate a network based on the Dyad Independence Model.
    
    Parameters:
    N (int): Number of nodes in the network.
    
    Returns:
    W (scipy.sparse.csr_matrix): Row-normalized random walk spatial weight matrix.
    """
    # Initialize the adjacency matrix
    A = np.zeros((N, N))
    
    # Define dyad state probabilities
    p_mutual = 10 * N**(-2)      # Probability of a mutual dyad (1, 1)
    p_asym = N**(-0.8)           # Total probability of an asymmetric dyad (1, 0) or (0, 1)
    p_asym_each = p_asym / 2     # Individual probability for (1, 0) or (0, 1)
    p_null = 1 - p_mutual - p_asym # Probability of a null dyad (0, 0)
    
    # Generate a dyad state for each node pair (i < j)
    for i in range(N):
        for j in range(i+1, N):
            # Randomly sample the dyad state
            r = random.random()
            
            if r < p_mutual:
                # Mutual dyad: bidirectional edge
                A[i, j] = 1
                A[j, i] = 1
            elif r < p_mutual + p_asym_each:
                # Asymmetric dyad: directed edge from i to j
                A[i, j] = 1
                A[j, i] = 0
            elif r < p_mutual + 2 * p_asym_each:
                # Asymmetric dyad: directed edge from j to i
                A[i, j] = 0
                A[j, i] = 1
            # else: Null dyad (no edges), remains 0 as initialized
    
    # Ensure no self-loops exist
    np.fill_diagonal(A, 0)
    
    # Calculate out-degrees for row normalization
    Asum = np.array(A.sum(axis=1)).flatten()  
    Asum[Asum == 0] = 1  # Prevent division by zero for isolated nodes
    
    # Construct the row-normalized spatial weight matrix W
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = diags(r_inv)
    
    A_sparse = coo_matrix(A)
    W = r_mat_inv.dot(A_sparse)
    
    return W



def _get_num_pos_edges(c1_size, c2_size, same_cluster, self_loops, directed):
    """
    Compute the number of possible edges between two clusters.
    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param same_cluster: Whether these are the same cluster
    :param self_loops: Whether we will generate self loops
    :param directed: Whether we are generating a directed graph
    :return: the number of possible edges between these clusters
    """
    if not same_cluster:
        # The number is simply the product of the number of vertices
        return c1_size * c2_size
    else:
        # The base number is n choose 2
        possible_edges_between_clusters = int((c1_size * (c1_size - 1)) / 2)

        # If we are allowed self-loops, then add them on
        if self_loops:
            possible_edges_between_clusters += c1_size

        # The number is normally the same for undirected and directed graphs, unless the clusters are the same, in which
        # case the number for the directed graph is double since we need to consider both directions of each edge.
        if directed:
            possible_edges_between_clusters *= 2

        # But if we are allowed self-loops, then we shouldn't double them since there is only one 'direction'.
        if directed and self_loops:
            possible_edges_between_clusters -= c1_size

        return possible_edges_between_clusters


def _get_number_of_edges(c1_size, c2_size, prob, same_cluster, self_loops, directed):
    """
    Compute the number of edges there will be between two clusters.
    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param prob: The probability of an edge between the clusters
    :param same_cluster: Whether these are the same cluster
    :param self_loops: Whether we will generate self loops
    :param directed: Whether we are generating a directed graph
    :return: the number of edges to generate between these clusters
    """
    # We need to compute the number of possible edges
    possible_edges_between_clusters = _get_num_pos_edges(c1_size, c2_size, same_cluster, self_loops, directed)

    # Sample the number of edges from the binomial distribution
    return np.random.binomial(possible_edges_between_clusters, prob)


def _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=False):
    """
    Given a list of cluster sizes, and a square matrix Q, generates edges for a graph in the following way.
    For two vertices u and v where u is in cluster i and v is in cluster j, there is an edge between u and v with
    probability Q_{i, j}.
    For the undirected case, we assume that the matrix Q is symmetric (and in practice look only at the upper triangle).
    For the directed case, we generate edges (u, v) and (v, u) with probabilities Q_{i, j} and Q_{j, i} respectively.
    May return self-loops. The calling code can decide what to do with them.
    Returns edges as pairs (u, v) where u and v are integers giving the index of the respective vertices.
    :param cluster_sizes: a list giving the number of vertices in each cluster
    :param prob_mat_q: A square matrix where Q_{i, j} is the probability of each edge between clusters i and j. Should
                       be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :return: Edges (u, v).
    """
    # We will iterate over the clusters. This variable keeps track of the index of the first vertex in the current
    # cluster_1.
    c1_base_index = 0

    for cluster_1 in range(len(cluster_sizes)):
        # Keep track of the index of the first vertex in the current cluster_2
        c2_base_index = c1_base_index

        # If we are constructing a directed graph, we need to consider all values of cluster_2.
        # Otherwise, we will consider only the clusters with an index >= cluster_1.
        if directed:
            second_clusters = range(len(cluster_sizes))
            c2_base_index = 0
        else:
            second_clusters = range(cluster_1, len(cluster_sizes))

        for cluster_2 in second_clusters:
            # Compute the number of edges between these two clusters
            num_edges = _get_number_of_edges(cluster_sizes[cluster_1],
                                             cluster_sizes[cluster_2],
                                             prob_mat_q[cluster_1][cluster_2],
                                             cluster_1 == cluster_2,
                                             True,
                                             directed)

            # Sample this number of edges. TODO: correct for possible double-sampling of edges
            num_possible_edges = (cluster_sizes[cluster_1] * cluster_sizes[cluster_2]) - 1
            for i in range(num_edges):
                edge_idx = random.randint(0, num_possible_edges)
                u = c1_base_index + int(edge_idx / cluster_sizes[cluster_1])
                v = c2_base_index + (edge_idx % cluster_sizes[cluster_1])
                yield u, v

            # Update the base index for the second cluster
            c2_base_index += cluster_sizes[cluster_2]

        # Update the base index of this cluster
        c1_base_index += cluster_sizes[cluster_1]


def sbm_adjmat(cluster_sizes, prob_mat_q, directed=False, self_loops=False):
    """
    Generate a graph from the stochastic block model.
    The list cluster_sizes gives the number of vertices inside each cluster and the matrix Q gives the probability of
    each edge between pairs of clusters.
    For two vertices u and v where u is in cluster i and v is in cluster j, there is an edge between u and v with
    probability Q_{i, j}.
    For the undirected case, we assume that the matrix Q is symmetric (and in practice look only at the upper triangle).
    For the directed case, we generate edges (u, v) and (v, u) with probabilities Q_{i, j} and Q_{j, i} respectively.
    Returns the adjacency matrix of the graph as a sparse scipy matrix in the CSR format.
    :param cluster_sizes: The number of vertices in each cluster.
    :param prob_mat_q: A square matrix where Q_{i, j} is the probability of each edge between clusters i and j. Should
                       be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :param self_loops: Whether to generate self-loops (default is false).
    :return: The sparse adjacency matrix of the graph.
    """
    # Initialize the adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((sum(cluster_sizes), sum(cluster_sizes)))

    # Generate the edges in the graph
    for (u, v) in _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=directed):
        if u != v or self_loops:
            # Add this edge to the adjacency matrix.
            adj_mat[u, v] = 1

            if not directed:
                adj_mat[v, u] = 1

    # Reformat the output matrix to the CSR format
    return adj_mat.tocsr()


def ssbm_adjmat(n, k, p, q, directed=False):
    """
    Generate a graph from the symmetric stochastic block model.
    Generates a graph with n vertices and k clusters. Every cluster will have floor(n/k) vertices. The probability of
    each edge inside a cluster is given by p. The probability of an edge between two different clusters is q.
    :param n: The number of vertices in the graph.
    :param k: The number of clusters.
    :param p: The probability of an edge inside a cluster.
    :param q: The probability of an edge between clusters.
    :param directed: Whether to generate a directed graph.
    :return: The sparse adjacency matrix of the graph.
    """
    # Every cluster has the same size.
    cluster_sizes = [int(n/k)] * k

    # Construct the k*k probability matrix Q. The off-diagonal entries are all q and the diagonal entries are all p.
    prob_mat_q = []
    for row_num in range(k):
        new_row = [q] * k
        new_row[row_num] = p
        prob_mat_q.append(new_row)

    # Call the general sbm method.
    return sbm_adjmat(cluster_sizes, prob_mat_q, directed=directed)


def generate_Block(N):
    Nblock=20
    A = ssbm_adjmat(N, Nblock, 20/N, 2/N, directed=False)
    Asum = np.array(A.sum(1))
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = sps.diags(r_inv)
    W = r_mat_inv.dot(A)
    
    return W

def generate(N,p,beta,rho,type_w):
    """
    Generate synthetic data for the Spatial Autoregressive (SAR) model under standard assumptions.
    
    The response vector Y is generated according to the DGP: Y = (I - \rho W)^{-1} (X\beta + E).
    Both the covariates X and the error terms E are drawn from a standard normal distribution N(0, 1).
    
    Parameters:
    N: Sample size (number of nodes).
    p: Dimension of covariates.
    beta: True covariate coefficients.
    rho: True spatial autoregressive parameter.
    type_w: Type of network topology ('PowerLaw', 'Dyad', or 'SBM').
    
    Returns:
    X (ndarray): Generated covariates matrix (N x p).
    Y (ndarray): Generated response vector (N x 1).
    W (scipy.sparse/ndarray): weight matrix (N x N).
    """
    #np.random.seed(123)
    if type_w == 'PowerLaw':
        W=generate_PowerLaw1(N)
    elif type_w == 'Dyad':
        W=generate_Dyad(N)
    elif type_w == 'SBM':
        W=generate_Block(N)
    
    sigma2=1
    
    S = np.eye(N)-rho*W
    S1=np.linalg.inv(S)

    X=np.random.normal(0,1,N*p).reshape(N,p) 
    
    E=np.random.normal(0,sigma2**0.5,N).reshape(N,1)
    
    Y=S1.dot(X.dot(beta)+E)
    return X,Y,W

def generate_compare(N,p,beta,rho,type_w):
    """
    Generate synthetic data for the SAR model, specifically designed to align with the empirical setup and assumptions of Wang & Song (2024) for fair comparison.
    
    To satisfy their strict bounded covariate assumption required for the Functional Mechanism, the covariates X are drawn from a bounded uniform distribution U(-1, 1). 
    Additionally, the error variance is reduced to 0.5 to match their experimental design of the response.
    """
    if type_w == 'PowerLaw':
        W=generate_PowerLaw1(N)
    elif type_w == 'Dyad':
        W=generate_Dyad(N)
    elif type_w == 'SBM':
        W=generate_Block(N)
    
    sigma2=0.5
    
    S = np.eye(N)-rho*W
    S1=np.linalg.inv(S)

    X = np.random.uniform(-1, 1, N*p).reshape(N, p)
    
    
    E=np.random.normal(0,sigma2**0.5,N).reshape(N,1)
    
    Y=S1.dot(X.dot(beta)+E)
    return X,Y,W