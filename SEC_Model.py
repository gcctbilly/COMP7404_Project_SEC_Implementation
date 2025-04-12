import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from itertools import product
from scipy.optimize import linear_sum_assignment
import numpy as np

"""
    This file the SEC model and its parameter tuning function.
    It also contains the clustering accuracy function.
"""

def sec_out_of_sample_extension(x_new, W, b):
    """
    Given a new data point x_new in R^d, and the learned W, b in R^{d x c}, R^c,
    produce a cluster assignment.
    
    x_new: shape (d,) single data vector
    W: shape (d, c)
    b: shape (c,)
    c: number of clusters
    
    Returns: a single integer cluster label in {0, ..., c-1}
    """
    # 1. Compute y = W^T x_new + b => shape (c,)
    y = W.T @ x_new + b
    
    # 2. pick the index with the largest value in y.
    cluster_label = np.argmax(y)
    
    return cluster_label

class SEC:
    def __init__(self, use_local_regression=False, k=5, mu=1.0, gamma_g=1.0, gamma_l=1.0, normalized=True,sigma = None):
        """
        Initialize the SEC model.

        Parameters:
        - c: Number of clusters.
        - use_local_regression: Whether to use local regression for neighborhood graph construction.
        - k: Number of nearest neighbors AAA
        - mu, gamma_g, gamma_l: Hyperparameters for the model.
        - sigma: Gaussian width; if None, can auto-estimate.
        - normalized: Whether to normalize the data.
        """
        self.use_local_regression = use_local_regression
        self.k = k
        self.mu = mu
        self.gamma_g = gamma_g
        self.gamma_l = gamma_l
        self.normalized = normalized
        self.sigma = sigma

    def construct_knn_affinity(self, X):
        """
        Construct a k-NN based affinity matrix (A) with optional Gaussian weighting.
        X: data of shape (d, n)
        k: number of nearest neighbors
        sigma: Gaussian width; if None, can auto-estimate
        Returns: A in R^{n x n}, the affinity matrix (symmetric)
        """
        # Transpose X for easier row-wise distance computations: shape => (n, d)
        X_t = X.T
        n = X_t.shape[0]
        
        # Compute pairwise distances (squared Euclidean)
        dist2 = np.sum(X_t**2, axis=1, keepdims=True) \
                + np.sum(X_t**2, axis=1) \
                - 2 * (X_t @ X_t.T)
        
        # We will find the k nearest neighbors for each row
        # Sort each row's distances; take the indices
        A = np.zeros((n, n))
        for i in range(n):
            # sort the distances for row i
            sorted_idx = np.argsort(dist2[i, :])
            # pick k neighbors (excluding itself at sorted_idx[0])
            neighbors = sorted_idx[1:self.k+1]
            
            if self.sigma is None:
                # could pick a small positive value to avoid zero
                sigma_i = np.mean(np.sqrt(dist2[i, neighbors])) + 1e-12
            else:
                sigma_i = self.sigma
            
            # fill in the weights
            for j in neighbors:
                A[i, j] = np.exp(-dist2[i, j] / (2.0 * sigma_i**2))
        
        # Make the matrix symmetric
        A = 0.5 * (A + A.T)
        
        return A
    
    def laplacian_matrix(self, A):
        """
        Construct unnormalized (L) or normalized Laplacian (L_tilde).
        A: affinity matrix (n x n)
        normalized: bool for whether to construct normalized version
        Returns: L (n x n)
        """
        D = np.diag(A.sum(axis=1))
        if not self.normalized:
            # L = D - A
            return D - A
        else:
            # \tilde{L} = I - D^{-1/2} A D^{-1/2}
            # or equivalently D^{-1/2}(D - A)D^{-1/2}
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D) + 1e-12)))
            I = np.eye(A.shape[0])
            return I - (D_inv_sqrt @ A @ D_inv_sqrt)
    
    def global_regression_laplacian(self, X):
        """
        Compute L_g = H - X^T (X X^T + gamma_g I)^{-1} X
        where H = I_n - (1/n) 1_n 1_n^T (centering matrix).
        
        X: shape (d, n)
        gamma_g: float
        Returns: L_g in R^{n x n}
        """
        d, n = X.shape
        # Centering matrix
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))
        # (X X^T + gamma_g I) -> shape (d, d)
        reg_mat = X @ X.T + self.gamma_g * np.eye(d)
        
        # Inversion
        reg_mat_inv = np.linalg.inv(reg_mat)
        
        # Compute X^T * reg_mat_inv * X
        X_inv_X = X.T @ reg_mat_inv @ X  # shape (n, n)
        
        Lg = H - X_inv_X
        return Lg
    
    def local_regression_laplacian(self, X):
        """
        Constructs a Laplacian matrix L_l from local regression
        described in the paper's Section III-C.
        
        For each sample x_i, we gather its k neighbors Xi,
        solve local regression to get Wi, then form partial Laplacian blocks.
        This is more detailed and can be quite large for big n.
        
        X: (d, n)
        k: number of nearest neighbors
        gamma_l: local regression regularization parameter
        Returns: L_l in R^{n x n}
        """
        k = self.k
        d, n = X.shape
        # Step 1: compute distances and find kNN for each sample
        X_t = X.T
        dist2 = np.sum(X_t**2, axis=1, keepdims=True) \
                + np.sum(X_t**2, axis=1) \
                - 2 * (X_t @ X_t.T)
        
        L_l = np.zeros((n, n))
        
        for i in range(n):
            # get k nearest neighbors
            sorted_idx = np.argsort(dist2[i, :])
            neighbors = sorted_idx[1:self.k+1]
            # Xi shape (d, k)
            Xi = X[:, neighbors]
            # Instead, from the derivation, L_l is constructed so that:
            #   L_li = H_k - X_i^T (X_i X_i^T + gamma_l I)^{-1} X_i
            #
            # Here we want the block matrix approach. We'll fill it in directly:
            
            # local approach: solve (Xi Xi^T + gamma_l I_d)^{-1} Xi => shape (d, k)
            reg_local = Xi @ Xi.T + self.gamma_l * np.eye(d)
            reg_local_inv = np.linalg.inv(reg_local)
            
            # call it A_i => Xi^T * inv(...) * Xi
            A_i = Xi.T @ reg_local_inv @ Xi  # shape (k, k)
            
            # H_k = I_k - (1/k) 1_k 1_k^T
            H_k = np.eye(k) - (1.0/k)*np.ones((k, k))
            # Lli = H_k - A_i
            Lli = H_k - A_i
            
            
            # We treat the block among neighbors' indices
            for p in range(k):
                for q in range(k):
                    idx_p = neighbors[p]
                    idx_q = neighbors[q]
                    # Add because multiple local blocks might overlap
                    L_l[idx_p, idx_q] += Lli[p, q]
        return L_l
    def fit(self, X, c):
        """
        Perform Spectral Embedded Clustering (SEC) on in-sample data X.
        
        X: data matrix, shape (d, n)
        c: number of clusters
        Returns:
            labels: cluster labels in {0, 1, ..., c-1} for each of the n samples
            F: the continuous embedding (n, c)
            W, b: parameters for out-of-sample extension
        """
        d, n = X.shape
        
        # 1. Build an affinity matrix A (simple kNN + Gaussian)
        
        A = self.construct_knn_affinity(X)
        
        # 2. Build laplacian for SC
        
        L_or_Lt = self.laplacian_matrix(A)  # shape (n, n)
        
        # 3. Build global regression Laplacian L_g
        
        Lg = self.global_regression_laplacian(X)  # shape (n, n)
        
        # 4. If we use local regression Laplacian:
        if self.use_local_regression:
            L_l = self.local_regression_laplacian(X)
            # final matrix = L_l + mu * L_g
            big_L = L_l + self.mu * Lg
        else:
            # final matrix = L_or_Lt + mu * L_g
            big_L = L_or_Lt + self.mu * Lg
        
        # 5. Solve for the c smallest eigenvectors of big_L
        vals, vecs = eigh(big_L)
        
        # pick the c smallest eigenvalues/eigenvectors
        idx = np.argsort(vals)[:c]
        F = vecs[:, idx]  # shape (n, c)
        
        # 6. Solve for W, b from eqn (14)
        #    W = (X X^T + gamma_g I)^(-1) * X * F
        #    b = (1/n) * F^T * 1_n
        reg_mat = X @ X.T + self.gamma_g * np.eye(d)
        reg_mat_inv = np.linalg.inv(reg_mat + 1e-12*np.eye(d))
        W = reg_mat_inv @ X @ F
        b = (1.0 / n) * (F.T @ np.ones((n,)))
        # b shape => (c,)
        
        # 7. Discretize the rows of F with KMeans to get final cluster labels
        km = KMeans(n_clusters=c, n_init=10)
        labels = km.fit_predict(F)
        
        return labels, F, W, b

def clustering_accuracy(labels_pred, labels_true):
    """
    Compute clustering accuracy by finding the best mapping
    between predicted cluster labels and true labels via the
    Hungarian (Kuhn-Munkres) algorithm.
    """
    labels_pred = np.array(labels_pred)
    labels_true = np.array(labels_true)
    n = len(labels_true)
    
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    c_true = len(unique_true)
    c_pred = len(unique_pred)
    
    # Create a mapping from label to index
    true_to_idx = {cl: i for i, cl in enumerate(unique_true)}
    pred_to_idx = {cl: i for i, cl in enumerate(unique_pred)}
    
    # Build a contingency matrix M
    M = np.zeros((c_true, c_pred), dtype=np.int64)
    for i in range(n):
        r = true_to_idx[labels_true[i]]
        c = pred_to_idx[labels_pred[i]]
        M[r, c] += 1
    
    # Hungarian algorithm to maximize trace(M) → min(-M)
    row_ind, col_ind = linear_sum_assignment(-M)
    acc = M[row_ind, col_ind].sum() / n
    return acc

def param_tuning_sec(
    X, 
    labels_true, 
    c, 
    param_grid_mu, 
    param_grid_gammag, 
    param_grid_gammal, 
    k, 
    normalized=True, 
    use_local_regression=False
):
    """
    Example function to do parameter tuning for SEC on a single training set (X, labels_true).
    We try all combos of mu, gamma_g, gamma_l from the given param grids.
    
    Return the best parameters + the best accuracy found.
    """
    from copy import deepcopy
    
    best_acc = -1.0
    best_params = (None, None, None)
    
    for mu_val, gg_val, gl_val, k in product(param_grid_mu, param_grid_gammag, param_grid_gammal, k):
        # Run SEC
        model = SEC(
            use_local_regression=use_local_regression,
            k=k,
            mu=mu_val,
            gamma_g=gg_val,
            gamma_l=gl_val,
            normalized=normalized
        )
        
        labels_pred, F, W, b = model.fit(X, c)
        acc_val = clustering_accuracy(labels_pred, labels_true)
        
        if acc_val > best_acc:
            best_acc = acc_val
            best_params = (mu_val, gg_val, gl_val,k)
    
    return best_params, best_acc