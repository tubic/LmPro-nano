import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

class FreeEnergyDecomposition:
    """Decomposition of binding free energy into geometric and topological components."""
    
    def __init__(self, temperature=298.15, beta=None):
        """
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        beta : float, optional
            Inverse temperature (1/kT) in mol/kcal
        """
        self.temperature = temperature
        
        # Boltzmann constant in kcal/(mol·K)
        self.kB = 0.0019872041  # kcal/(mol·K)
        
        # Compute beta = 1/kT if not provided
        if beta is None:
            self.beta = 1.0 / (self.kB * self.temperature)
        else:
            self.beta = beta
    
    def compute_energy_decomposition(self, interface_points, metric_tensors, 
                                   mean_curvature, gaussian_curvature,
                                   persistence_diagrams):
        """
        Compute decomposition of binding free energy.
        
        Parameters:
        -----------
        interface_points : np.ndarray
            Coordinates of interface points
        metric_tensors : np.ndarray
            Metric tensor for each point
        mean_curvature : np.ndarray
            Mean curvature for each point
        gaussian_curvature : np.ndarray
            Gaussian curvature for each point
        persistence_diagrams : list
            Persistence diagrams for each homology dimension
            
        Returns:
        --------
        energy_components : dict
            Dictionary of free energy components
        """
        # Compute geometric contribution
        dG_geo = self._compute_geometric_contribution(
            interface_points, metric_tensors, mean_curvature, gaussian_curvature
        )
        
        # Compute topological contribution
        dG_topo = self._compute_topological_contribution(persistence_diagrams)
        
        # Compute coupling contribution
        dG_coupling = self._compute_coupling_contribution(
            interface_points, mean_curvature, gaussian_curvature, persistence_diagrams
        )
        
        # Total binding free energy
        dG_total = dG_geo + dG_topo + dG_coupling
        
        # Collect energy components
        energy_components = {
            'dG_geometric': dG_geo,
            'dG_topological': dG_topo,
            'dG_coupling': dG_coupling,
            'dG_total': dG_total
        }
        
        return energy_components
    
    def _compute_geometric_contribution(self, interface_points, metric_tensors, 
                                      mean_curvature, gaussian_curvature):
        """
        Compute geometric contribution to binding free energy.
        
        Parameters:
        -----------
        interface_points : np.ndarray
            Coordinates of interface points
        metric_tensors : np.ndarray
            Metric tensor for each point
        mean_curvature : np.ndarray
            Mean curvature for each point
        gaussian_curvature : np.ndarray
            Gaussian curvature for each point
            
        Returns:
        --------
        dG_geo : float
            Geometric contribution to free energy
        """
        n_points = len(interface_points)
        
        # Compute area element for each point
        area_elements = np.zeros(n_points)
        
        try:
            # Try to compute triangulation
            tri = Delaunay(interface_points)
            
            # Compute areas of triangles
            for simplex in tri.simplices:
                p1, p2, p3 = interface_points[simplex]
                # Compute area using cross product
                area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
                
                # Distribute area to vertices
                for idx in simplex:
                    area_elements[idx] += area / 3.0
        except:
            # Fallback: estimate area using nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(interface_points)
            distances, indices = nbrs.kneighbors(interface_points)
            
            for i in range(n_points):
                # Estimate area as circle with radius = average distance to neighbors
                avg_dist = np.mean(distances[i][1:])  # Skip self
                area_elements[i] = np.pi * avg_dist**2
        
        # Compute geometric energy for each point
        geometric_energy = np.zeros(n_points)
        
        for i in range(n_points):
            # Energy depends on mean curvature, Gaussian curvature, and metric determinant
            H = mean_curvature[i]
            K = gaussian_curvature[i]
            g = metric_tensors[i]
            
            # Determinant of metric tensor
            det_g = np.linalg.det(g)
            
            # Geometric energy model: (2H^2 - K) * sqrt(det(g))
            geometric_energy[i] = (2.0 * H**2 - K) * np.sqrt(det_g)
        
        # Compute partition function
        Z_geo = np.sum(np.exp(-self.beta * geometric_energy) * area_elements)
        
        # Compute free energy
        dG_geo = -1.0 / self.beta * np.log(max(Z_geo, 1e-10))
        
        return dG_geo
    
    def _compute_topological_contribution(self, persistence_diagrams):
        """
        Compute topological contribution to binding free energy.
        
        Parameters:
        -----------
        persistence_diagrams : list
            Persistence diagrams for each homology dimension
            
        Returns:
        --------
        dG_topo : float
            Topological contribution to free energy
        """
        # Energy for each persistence pair
        topo_energy = []
        
        # Process each homology dimension
        for dim, diagram in enumerate(persistence_diagrams):
            for birth, death in diagram:
                # Persistence = death - birth
                persistence = death - birth
                
                # Energy model: higher persistence = lower energy
                # Scale by dimension
                energy = -np.log(1.0 + persistence) * (dim + 1.0)
                topo_energy.append(energy)
        
        # Compute partition function
        Z_topo = np.sum(np.exp(-self.beta * np.array(topo_energy)))
        
        # Compute free energy
        dG_topo = -1.0 / self.beta * np.log(max(Z_topo, 1e-10))
        
        return dG_topo
    
    def _compute_coupling_contribution(self, interface_points, mean_curvature, 
                                     gaussian_curvature, persistence_diagrams):
        """
        Compute coupling contribution to binding free energy.
        
        Parameters:
        -----------
        interface_points : np.ndarray
            Coordinates of interface points
        mean_curvature : np.ndarray
            Mean curvature for each point
        gaussian_curvature : np.ndarray
            Gaussian curvature for each point
        persistence_diagrams : list
            Persistence diagrams for each homology dimension
            
        Returns:
        --------
        dG_coupling : float
            Coupling contribution to free energy
        """
        n_points = len(interface_points)
        
        # Extract topological features
        topo_features = []
        for dim, diagram in enumerate(persistence_diagrams):
            for birth, death in diagram:
                persistence = death - birth
                topo_features.append((dim, birth, death, persistence))
        
        # Sort by persistence
        topo_features.sort(key=lambda x: x[3], reverse=True)
        
        # Take top 5 features
        top_features = topo_features[:5]
        
        # Compute coupling energy for each point
        coupling_energy = np.zeros(n_points)
        
        for i in range(n_points):
            H = mean_curvature[i]
            K = gaussian_curvature[i]
            
            # Coupling model
            for dim, birth, death, persistence in top_features:
                # Different coupling depending on homology dimension
                if dim == 0:  # Connected components
                    coupling_energy[i] += 0.5 * persistence * H**2
                elif dim == 1:  # Loops
                    coupling_energy[i] += 0.5 * persistence * (K - H**2)
                elif dim == 2:  # Voids
                    coupling_energy[i] += 0.5 * persistence * K
        
        # Compute partition function
        Z_coupling = np.sum(np.exp(-self.beta * coupling_energy))
        
        # Compute free energy
        dG_coupling = -1.0 / self.beta * np.log(max(Z_coupling, 1e-10))
        
        return dG_coupling