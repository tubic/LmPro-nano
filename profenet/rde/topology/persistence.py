import numpy as np
import gudhi as gd
from gudhi.representations import Landscape
from scipy.spatial.distance import pdist, squareform

def compute_persistence_diagrams(points, max_edge_length=10.0, max_dim=2):
    """
    Compute persistence diagrams for a point cloud.
    
    Parameters:
    -----------
    points : np.ndarray
        Point cloud coordinates (shape: [n_points, 3])
    max_edge_length : float
        Maximum edge length for filtration
    max_dim : int
        Maximum homology dimension
        
    Returns:
    --------
    diagrams : list
        List of persistence diagrams for each homology dimension
    """
    # Compute distance matrix
    distance_matrix = squareform(pdist(points, metric='euclidean'))
    
    # Create Vietoris-Rips complex
    rips_complex = gd.RipsComplex(
        distance_matrix=distance_matrix,
        max_edge_length=max_edge_length
    )
    
    # Create filtration
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    
    # Compute persistence
    simplex_tree.compute_persistence()
    
    # Extract persistence diagrams
    diagrams = []
    for dim in range(max_dim + 1):
        diagram = np.array(simplex_tree.persistence_intervals_in_dimension(dim))
        if diagram.size > 0:
            # Replace infinity with max_edge_length
            diagram[np.isinf(diagram)] = max_edge_length
            diagrams.append(diagram)
        else:
            diagrams.append(np.zeros((0, 2)))
    
    return diagrams

def compute_persistence_landscapes(diagrams, num_landscapes=5, resolution=100):
    """
    Compute persistence landscapes from persistence diagrams.
    
    Parameters:
    -----------
    diagrams : list
        List of persistence diagrams for each homology dimension
    num_landscapes : int
        Number of landscape functions
    resolution : int
        Resolution of each landscape function
        
    Returns:
    --------
    landscapes : list
        List of persistence landscapes for each homology dimension
    """
    landscape_generator = Landscape(
        num_landscapes=num_landscapes,
        resolution=resolution
    )
    
    # Compute landscapes for each dimension
    landscapes = []
    for dim_diagram in diagrams:
        if dim_diagram.size > 0:
            # Convert to format expected by GUDHI
            formatted_diagram = [(0, (b, d)) for b, d in dim_diagram]
            dim_landscape = landscape_generator.fit_transform([formatted_diagram])[0]
            landscapes.append(dim_landscape)
        else:
            # Empty diagram produces zero landscape
            landscapes.append(np.zeros((num_landscapes, resolution)))
    
    return landscapes

def compute_wasserstein_distance(diagram1, diagram2, p=2):
    """
    Compute Wasserstein distance between persistence diagrams.
    
    Parameters:
    -----------
    diagram1 : list
        First persistence diagram
    diagram2 : list
        Second persistence diagram
    p : int
        Power parameter for Wasserstein distance
        
    Returns:
    --------
    distances : list
        Wasserstein distances for each homology dimension
    """
    distances = []
    
    for dim in range(min(len(diagram1), len(diagram2))):
        diag1 = diagram1[dim]
        diag2 = diagram2[dim]
        
        if diag1.size == 0 and diag2.size == 0:
            distances.append(0.0)
        elif diag1.size == 0:
            # Distance to empty diagram is sum of persistences
            distances.append(np.sum(diag2[:, 1] - diag2[:, 0])**(1/p))
        elif diag2.size == 0:
            distances.append(np.sum(diag1[:, 1] - diag1[:, 0])**(1/p))
        else:
            # Use GUDHI's implementation
            wasserstein_distance = gd.bottleneck.wasserstein_distance(diag1, diag2, p)
            distances.append(wasserstein_distance)
    
    return distances

def extract_topological_features(diagrams, max_features_per_dim=5):
    """
    Extract fixed-length feature vector from persistence diagrams.
    
    Parameters:
    -----------
    diagrams : list
        List of persistence diagrams for each homology dimension
    max_features_per_dim : int
        Maximum number of topological features per dimension
        
    Returns:
    --------
    features : np.ndarray
        Feature vector
    """
    features = []
    
    for dim, diagram in enumerate(diagrams):
        if diagram.size > 0:
            # Compute persistence for each pair
            persistence = diagram[:, 1] - diagram[:, 0]
            
            # Sort by persistence
            sorted_idx = np.argsort(persistence)[::-1]
            
            # Take top features
            top_persistence = persistence[sorted_idx[:max_features_per_dim]]
            
            # Pad if necessary
            if len(top_persistence) < max_features_per_dim:
                top_persistence = np.pad(
                    top_persistence, 
                    (0, max_features_per_dim - len(top_persistence))
                )
            
            # Add to features
            features.extend(top_persistence)
        else:
            # Empty diagram
            features.extend(np.zeros(max_features_per_dim))
    
    return np.array(features)