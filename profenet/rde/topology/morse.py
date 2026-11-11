import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

def compute_discrete_morse_complex(points, function_values, max_edge_length=10.0):
    """
    Compute discrete Morse complex.
    
    Parameters:
    -----------
    points : np.ndarray
        Point cloud coordinates (shape: [n_points, 3])
    function_values : np.ndarray
        Function values at each point (e.g., mean curvature)
    max_edge_length : float
        Maximum edge length for connections
        
    Returns:
    --------
    critical_points : list
        List of (index, type) pairs for critical points
    morse_complex : dict
        Dictionary with information about the Morse complex
    """
    n_points = len(points)
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with function values
    for i in range(n_points):
        G.add_node(i, value=function_values[i])
    
    # Add edges based on neighborhood
    for i in range(n_points):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):  # Skip self
            if dist <= max_edge_length:
                G.add_edge(i, j, weight=dist)
    
    # Identify critical points
    critical_points = []
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        node_value = G.nodes[node]['value']
        
        # Count higher and lower neighbors
        higher_neighbors = sum(1 for n in neighbors if G.nodes[n]['value'] > node_value)
        lower_neighbors = sum(1 for n in neighbors if G.nodes[n]['value'] < node_value)
        
        # Classify critical points
        if lower_neighbors == 0:
            critical_points.append((node, 'minimum'))
        elif higher_neighbors == 0:
            critical_points.append((node, 'maximum'))
        elif lower_neighbors == 1 and higher_neighbors == 1:
            # Regular point
            pass
        else:
            critical_points.append((node, 'saddle'))
    
    # Compute Morse-Smale complex (simplified)
    morse_complex = {
        'minima': [idx for idx, type_ in critical_points if type_ == 'minimum'],
        'saddles': [idx for idx, type_ in critical_points if type_ == 'saddle'],
        'maxima': [idx for idx, type_ in critical_points if type_ == 'maximum']
    }
    
    # Trace gradient flows between critical points
    morse_complex['ascending_manifolds'] = trace_gradient_flows(
        G, morse_complex['minima'], ascending=True
    )
    
    morse_complex['descending_manifolds'] = trace_gradient_flows(
        G, morse_complex['maxima'], ascending=False
    )
    
    return critical_points, morse_complex

def trace_gradient_flows(G, start_points, ascending=True):
    """
    Trace gradient flows from critical points.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph representation of the point cloud
    start_points : list
        Indices of starting critical points
    ascending : bool
        If True, follow ascending flow; if False, follow descending flow
        
    Returns:
    --------
    flows : dict
        Dictionary mapping start points to lists of flow paths
    """
    flows = {start: [] for start in start_points}
    
    for start in start_points:
        # Use breadth-first search to trace gradient flows
        queue = [(start, [start])]
        visited = set([start])
        
        while queue:
            node, path = queue.pop(0)
            node_value = G.nodes[node]['value']
            
            # Find neighbors in the direction of flow
            neighbors = []
            for neighbor in G.neighbors(node):
                if ascending:
                    if G.nodes[neighbor]['value'] > node_value:
                        neighbors.append((neighbor, G.nodes[neighbor]['value']))
                else:
                    if G.nodes[neighbor]['value'] < node_value:
                        neighbors.append((neighbor, G.nodes[neighbor]['value']))
            
            # Sort by function value
            neighbors.sort(key=lambda x: x[1], reverse=ascending)
            
            # Follow steepest path
            for neighbor, _ in neighbors[:2]:  # Consider up to 2 steepest paths
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)
                    
                    # If we've reached another critical point, record the path
                    if not neighbors:  # No valid neighbors = critical point
                        flows[start].append(path)
        
    return flows