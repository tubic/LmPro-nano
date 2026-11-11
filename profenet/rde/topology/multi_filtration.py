import numpy as np
import gudhi as gd
from sklearn.neighbors import NearestNeighbors

def compute_multiparameter_persistence(points, param1, param2, max_edge_length=10.0, 
                                      num_directions=8, max_dim=2):
    """
    Compute multi-parameter persistence using the fibered barcode approach.
    
    Parameters:
    -----------
    points : np.ndarray
        Point cloud coordinates (shape: [n_points, 3])
    param1 : np.ndarray
        First parameter values at each point (e.g., electrostatic potential)
    param2 : np.ndarray
        Second parameter values at each point (e.g., hydrophobicity)
    max_edge_length : float
        Maximum edge length for filtration
    num_directions : int
        Number of linear combinations to compute
    max_dim : int
        Maximum homology dimension
        
    Returns:
    --------
    fibered_barcodes : dict
        Dictionary mapping directions to persistence diagrams
    """
    n_points = len(points)
    
    # Normalize parameters to [0, 1] range
    param1_norm = (param1 - np.min(param1)) / (np.max(param1) - np.min(param1) + 1e-10)
    param2_norm = (param2 - np.min(param2)) / (np.max(param2) - np.min(param2) + 1e-10)
    
    # Find nearest neighbors for constructing the base simplicial complex
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Generate directions for linear combinations
    angles = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    fibered_barcodes = {}
    
    for i, direction in enumerate(directions):
        # Compute linear combination of parameters
        combined_param = direction[0] * param1_norm + direction[1] * param2_norm
        
        # Create simplex tree for this direction
        simplex_tree = gd.SimplexTree()
        
        # Add vertices
        for j in range(n_points):
            simplex_tree.insert([j], filtration=combined_param[j])
        
        # Add edges based on neighborhood and parameter values
        for j in range(n_points):
            for k in indices[j, 1:]:  # Skip self
                if k > j:  # Avoid duplicate edges
                    edge_length = distances[j, np.where(indices[j] == k)[0][0]]
                    if edge_length <= max_edge_length:
                        # Use maximum of parameter values as filtration value
                        filt_val = max(combined_param[j], combined_param[k])
                        simplex_tree.insert([j, k], filtration=filt_val)
        
        # Add triangles
        simplex_tree.expansion(max_dim)
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract persistence diagrams
        diagrams = []
        for dim in range(max_dim + 1):
            diagram = np.array(simplex_tree.persistence_intervals_in_dimension(dim))
            if diagram.size > 0:
                # Replace infinity with max value
                diagram[np.isinf(diagram)] = 1.0
                diagrams.append(diagram)
            else:
                diagrams.append(np.zeros((0, 2)))
        
        # Store diagrams for this direction
        direction_name = f"direction_{i}_{direction[0]:.2f}_{direction[1]:.2f}"
        fibered_barcodes[direction_name] = diagrams
    
    return fibered_barcodes

def extract_multiparameter_features(fibered_barcodes, max_features_per_dim=3):
    """
    Extract fixed-length feature vector from multi-parameter persistence.
    
    Parameters:
    -----------
    fibered_barcodes : dict
        Dictionary mapping directions to persistence diagrams
    max_features_per_dim : int
        Maximum number of topological features per dimension
        
    Returns:
    --------
    features : np.ndarray
        Feature vector
    """
    features = []
    
    # For each direction
    for direction, diagrams in fibered_barcodes.items():
        direction_features = []
        
        # For each homology dimension
        for dim, diagram in enumerate(diagrams):
            if diagram.size > 0:
                # Compute persistence
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
                
                direction_features.extend(top_persistence)
            else:
                # Empty diagram
                direction_features.extend(np.zeros(max_features_per_dim))
        
        features.extend(direction_features)
    
    return np.array(features)