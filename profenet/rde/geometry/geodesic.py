import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors

def construct_laplace_beltrami_operator(points, metric_tensors=None, k=15):
    """
    Construct discrete Laplace-Beltrami operator.
    
    Parameters:
    -----------
    points : np.ndarray
        Coordinates of points on the manifold (shape: [n_points, 3])
    metric_tensors : np.ndarray, optional
        Metric tensor for each point (shape: [n_points, 3, 3])
    k : int
        Number of neighbors for operator construction
        
    Returns:
    --------
    L : scipy.sparse.csr_matrix
        Laplace-Beltrami operator as sparse matrix
    """
    n_points = len(points)
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Construct sparse matrix
    rows = []
    cols = []
    values = []
    
    for i in range(n_points):
        # Get neighbors (excluding the point itself)
        nbr_idx = indices[i, 1:]
        nbr_dist = distances[i, 1:]
        
        # Compute weights
        if metric_tensors is not None:
            # Use Riemannian metric
            weights = np.zeros_like(nbr_dist)
            for j, idx in enumerate(nbr_idx):
                diff = points[idx] - points[i]
                # Average metric tensors at two points
                avg_metric = (metric_tensors[i] + metric_tensors[idx]) / 2.0
                # Compute Riemannian distance
                riemannian_dist = np.sqrt(diff @ avg_metric @ diff)
                # Weight is inverse of squared distance
                weights[j] = 1.0 / max(riemannian_dist**2, 1e-10)
        else:
            # Use Euclidean distance
            weights = 1.0 / np.maximum(nbr_dist**2, 1e-10)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Add off-diagonal entries
        for j, idx in enumerate(nbr_idx):
            rows.append(i)
            cols.append(idx)
            values.append(-weights[j])
        
        # Add diagonal entry (row sum)
        rows.append(i)
        cols.append(i)
        values.append(1.0)
    
    # Construct sparse matrix
    L = sp.csr_matrix((values, (rows, cols)), shape=(n_points, n_points))
    
    return L

def compute_heat_kernel_signature(L, n_eigenvalues=50, time_points=None):
    """
    Compute Heat Kernel Signature (HKS) for points on the manifold.
    
    Parameters:
    -----------
    L : scipy.sparse.csr_matrix
        Laplace-Beltrami operator
    n_eigenvalues : int
        Number of eigenvalues to use
    time_points : np.ndarray, optional
        Time points for HKS evaluation
        
    Returns:
    --------
    eigenvalues : np.ndarray
        Eigenvalues of Laplace-Beltrami operator
    eigenvectors : np.ndarray
        Eigenvectors of Laplace-Beltrami operator
    hks : np.ndarray
        Heat Kernel Signature (shape: [n_points, n_times])
    """
    # Default time points if not provided
    if time_points is None:
        time_points = np.logspace(-2, 2, 10)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = spla.eigsh(L, k=n_eigenvalues, which='SM')
    
    # Ensure eigenvalues are sorted
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute HKS for each time point
    n_points = L.shape[0]
    n_times = len(time_points)
    hks = np.zeros((n_points, n_times))
    
    for i, t in enumerate(time_points):
        # HKS at time t: sum_j exp(-λ_j * t) * φ_j(x)^2
        hks[:, i] = np.sum(np.exp(-eigenvalues * t) * eigenvectors**2, axis=1)
    
    return eigenvalues, eigenvectors, hks

def compute_wave_kernel_signature(L, n_eigenvalues=50, energies=None, sigma=1.0):
    """
    Compute Wave Kernel Signature (WKS) for points on the manifold.
    
    Parameters:
    -----------
    L : scipy.sparse.csr_matrix
        Laplace-Beltrami operator
    n_eigenvalues : int
        Number of eigenvalues to use
    energies : np.ndarray, optional
        Energy levels for WKS evaluation
    sigma : float
        Standard deviation for Gaussian filters
        
    Returns:
    --------
    wks : np.ndarray
        Wave Kernel Signature (shape: [n_points, n_energies])
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = spla.eigsh(L, k=n_eigenvalues, which='SM')
    
    # Ensure eigenvalues are sorted
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Default energy levels if not provided
    if energies is None:
        log_emin = np.log(max(eigenvalues[1], 1e-6))  # Skip first eigenvalue (zero)
        log_emax = np.log(eigenvalues[-1])
        energies = np.exp(np.linspace(log_emin, log_emax, 10))
    
    # Compute WKS for each energy level
    n_points = L.shape[0]
    n_energies = len(energies)
    wks = np.zeros((n_points, n_energies))
    
    for i, e in enumerate(energies):
        # Gaussian filter centered at energy e
        gaussian = np.exp(-(eigenvalues - e)**2 / (2 * sigma**2))
        
        # WKS at energy e: sum_j exp(-(λ_j - e)^2/(2σ^2)) * φ_j(x)^2
        wks_e = np.sum(gaussian * eigenvectors**2, axis=1)
        
        # Normalize
        wks[:, i] = wks_e / np.sum(gaussian)
    
    return wks