import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_curvature_tensors(points, normals=None, k=15):
    """
    Compute curvature tensors for a point cloud.
    
    Parameters:
    -----------
    points : np.ndarray
        Coordinates of points on the manifold (shape: [n_points, 3])
    normals : np.ndarray, optional
        Normal vectors at each point (shape: [n_points, 3])
    k : int
        Number of neighbors for local curvature estimation
        
    Returns:
    --------
    mean_curvature : np.ndarray
        Mean curvature at each point
    gaussian_curvature : np.ndarray
        Gaussian curvature at each point
    principal_curvatures : np.ndarray
        Principal curvatures at each point (shape: [n_points, 2])
    """
    n_points = len(points)
    
    # If normals not provided, estimate them
    if normals is None:
        normals = estimate_normals(points, k)
    
    # Initialize curvature arrays
    mean_curvature = np.zeros(n_points)
    gaussian_curvature = np.zeros(n_points)
    principal_curvatures = np.zeros((n_points, 2))
    
    # Find nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        # Get neighborhood (excluding the point itself)
        nbr_idx = indices[i, 1:]
        neighborhood = points[nbr_idx]
        
        # Center the neighborhood
        center = points[i]
        centered_nbrs = neighborhood - center
        
        # Get normal vector at the point
        normal = normals[i]
        
        # Create local coordinate system
        # First tangent direction: arbitrary perpendicular to normal
        if abs(normal[0]) < abs(normal[1]):
            tangent1 = np.array([1.0, 0.0, 0.0])
        else:
            tangent1 = np.array([0.0, 1.0, 0.0])
            
        tangent1 = tangent1 - np.dot(tangent1, normal) * normal
        tangent1 /= np.linalg.norm(tangent1)
        
        # Second tangent direction: cross product
        tangent2 = np.cross(normal, tangent1)
        tangent2 /= np.linalg.norm(tangent2)
        
        # Project neighborhood onto tangent plane
        x = np.dot(centered_nbrs, tangent1)
        y = np.dot(centered_nbrs, tangent2)
        z = np.dot(centered_nbrs, normal)
        
        # Fit quadratic surface: z = ax^2 + by^2 + cxy + dx + ey + f
        A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        
        a, b, c, d, e, f = coeffs
        
        # Compute principal curvatures
        # For a quadratic surface, these are related to the coefficients
        trace = 2*a + 2*b  # Trace of the Hessian
        det = 4*a*b - c**2  # Determinant of the Hessian
        
        # Compute mean and Gaussian curvatures
        mean_curvature[i] = trace / 2
        gaussian_curvature[i] = det
        
        # Compute principal curvatures from mean and Gaussian curvatures
        discriminant = max(0, mean_curvature[i]**2 - gaussian_curvature[i])
        principal_curvatures[i, 0] = mean_curvature[i] + np.sqrt(discriminant)
        principal_curvatures[i, 1] = mean_curvature[i] - np.sqrt(discriminant)
    
    return mean_curvature, gaussian_curvature, principal_curvatures

def estimate_normals(points, k=15):
    """
    Estimate normal vectors using PCA on local neighborhoods.
    
    Parameters:
    -----------
    points : np.ndarray
        Coordinates of points (shape: [n_points, 3])
    k : int
        Number of neighbors for local PCA
        
    Returns:
    --------
    normals : np.ndarray
        Normal vectors at each point (shape: [n_points, 3])
    """
    n_points = len(points)
    normals = np.zeros_like(points)
    
    # Find nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        # Get neighborhood (including the point itself)
        neighborhood = points[indices[i]]
        
        # Center the neighborhood
        centered = neighborhood - np.mean(neighborhood, axis=0)
        
        # Compute covariance matrix
        cov = centered.T @ centered
        
        # Perform eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # The normal is the eigenvector corresponding to the smallest eigenvalue
        normals[i] = eigvecs[:, 0]
    
    # Ensure consistent orientation
    centroid = np.mean(points, axis=0)
    for i in range(n_points):
        # Orient normal to point outward from centroid
        if np.dot(points[i] - centroid, normals[i]) < 0:
            normals[i] = -normals[i]
    
    return normals

def compute_shape_index(principal_curvatures):
    """
    Compute shape index from principal curvatures.
    
    Parameters:
    -----------
    principal_curvatures : np.ndarray
        Principal curvatures (shape: [n_points, 2])
        
    Returns:
    --------
    shape_index : np.ndarray
        Shape index at each point (shape: [n_points])
    """
    kappa1 = principal_curvatures[:, 0]
    kappa2 = principal_curvatures[:, 1]
    
    # Avoid division by zero
    denom = kappa1 - kappa2
    denom[np.abs(denom) < 1e-10] = 1e-10
    
    # Compute shape index: S = (2/π) * arctan((κ1 + κ2)/(κ1 - κ2))
    shape_index = (2/np.pi) * np.arctan((kappa1 + kappa2) / denom)
    
    return shape_index