import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

class RiemannianManifold:
    """Riemannian manifold representation of protein interfaces."""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.25):
        """
        Parameters:
        -----------
        alpha : float
            Weight for Euclidean metric component
        beta : float
            Weight for chemical property metric component
        gamma : float
            Weight for hydrophobicity metric component
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def compute_metric_tensor(self, points, features=None, k=15):
        """
        Compute the Riemannian metric tensor at each point.
        
        Parameters:
        -----------
        points : np.ndarray
            Coordinates of points on the manifold (shape: [n_points, 3])
        features : np.ndarray, optional
            Feature vectors at each point (shape: [n_points, n_features])
        k : int
            Number of neighbors for local metric estimation
            
        Returns:
        --------
        metric_tensors : np.ndarray
            Metric tensor for each point (shape: [n_points, 3, 3])
        """
        n_points = len(points)
        metric_tensors = np.zeros((n_points, 3, 3))
        
        # Find nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        for i in range(n_points):
            # Euclidean metric component
            E_i = np.eye(3)  # Start with identity matrix
            
            # If features are provided, compute chemical similarity
            if features is not None:
                # Get features for neighborhood
                nbr_features = features[indices[i]]
                
                # Compute feature differences
                feature_diffs = nbr_features - features[i]
                feature_sim = np.exp(-np.sum(feature_diffs**2, axis=1) / (2.0 * 0.5**2))
                
                # Chemical property metric component
                C_i = np.eye(3) * np.mean(feature_sim)
                
                # Hydrophobicity component (simplified)
                hydro = nbr_features[:, 0]  # Assume first feature is hydrophobicity
                H_i = (1.0 + self.gamma * np.mean(hydro)) * np.eye(3)
                
                # Combine metric components
                metric_tensors[i] = self.alpha * E_i + self.beta * C_i + self.gamma * H_i
            else:
                # Just use Euclidean metric if no features
                metric_tensors[i] = E_i
                
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(metric_tensors[i])
            if np.min(eigvals) < 1e-5:
                metric_tensors[i] = eigvecs @ np.diag(np.maximum(eigvals, 1e-5)) @ eigvecs.T
                
        return metric_tensors
    
    def compute_geodesic_distance(self, points, metric_tensors, i, j):
        """
        Compute geodesic distance between two points on the manifold.
        
        Parameters:
        -----------
        points : np.ndarray
            Coordinates of points on the manifold
        metric_tensors : np.ndarray
            Metric tensor for each point
        i, j : int
            Indices of the two points
            
        Returns:
        --------
        distance : float
            Geodesic distance between points i and j
        """
        # Simple approximation using Euclidean distance and average metric tensor
        diff = points[j] - points[i]
        avg_metric = (metric_tensors[i] + metric_tensors[j]) / 2.0
        
        # Compute Riemannian distance: sqrt(v^T g v)
        distance = np.sqrt(diff @ avg_metric @ diff)
        
        return distance

class RiemannianLayer(nn.Module):
    """Neural network layer that respects Riemannian geometry of protein interfaces."""
    
    def __init__(self, in_features, out_features):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Metric tensor parameters
        self.metric_diag = nn.Parameter(torch.ones(in_features))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, metric_tensor=None):
        """
        Forward pass respecting Riemannian geometry.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features (shape: [batch_size, n_points, in_features])
        metric_tensor : torch.Tensor, optional
            External metric tensor (shape: [batch_size, n_points, in_features, in_features])
            
        Returns:
        --------
        y : torch.Tensor
            Output features (shape: [batch_size, n_points, out_features])
        """
        batch_size, n_points, _ = x.shape
        
        # Use external metric tensor if provided, otherwise use learned diagonal metric
        if metric_tensor is None:
            # Create diagonal metric tensor
            g = torch.diag_embed(torch.abs(self.metric_diag))
            g = g.unsqueeze(0).unsqueeze(0).expand(batch_size, n_points, -1, -1)
        else:
            g = metric_tensor
        
        # Apply metric tensor to input
        x_flat = x.reshape(-1, self.in_features)
        g_flat = g.reshape(-1, self.in_features, self.in_features)
        
        # Compute x^T g x for each point
        x_metric = torch.bmm(x_flat.unsqueeze(1), g_flat)
        x_metric = torch.bmm(x_metric, x_flat.unsqueeze(2)).squeeze(2)
        
        # Reshape back
        x_metric = x_metric.reshape(batch_size, n_points, self.in_features)
        
        # Apply regular linear transformation
        y = torch.matmul(x_metric, self.weight.t()) + self.bias
        
        return y