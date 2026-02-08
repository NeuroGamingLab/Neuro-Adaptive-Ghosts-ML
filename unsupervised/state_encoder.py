"""
Unsupervised State Encoder
Uses clustering and dimensionality reduction to encode game states
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import pickle
import os

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


class StateEncoder:
    """
    Encodes game states using unsupervised learning.
    Uses PCA for dimensionality reduction and K-Means for clustering.
    """
    
    def __init__(
        self,
        n_components: int = 32,
        n_clusters: int = 20,
        use_minibatch: bool = True
    ):
        """
        Initialize the state encoder.
        
        Args:
            n_components: Number of PCA components
            n_clusters: Number of K-Means clusters
            use_minibatch: Use MiniBatch K-Means for faster training
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.use_minibatch = use_minibatch
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        
        if use_minibatch:
            self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        else:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        self.is_fitted = False
        self.state_buffer: List[np.ndarray] = []
        self.cluster_centers_original: Optional[np.ndarray] = None
        
    def collect_state(self, state: np.ndarray):
        """
        Collect a state for later training.
        
        Args:
            state: Game state observation
        """
        self.state_buffer.append(state.flatten())
    
    def fit(self, states: Optional[np.ndarray] = None, verbose: bool = True):
        """
        Fit the encoder on collected states.
        
        Args:
            states: Optional array of states (uses buffer if not provided)
            verbose: Print progress information
        """
        if states is None:
            if len(self.state_buffer) == 0:
                raise ValueError("No states collected. Call collect_state() first or provide states.")
            states = np.array(self.state_buffer)
        
        if verbose:
            print(f"Fitting encoder on {len(states)} states...")
        
        # Standardize
        states_scaled = self.scaler.fit_transform(states)
        
        if verbose:
            print("  - Fitted scaler")
        
        # PCA
        states_reduced = self.pca.fit_transform(states_scaled)
        
        if verbose:
            explained_var = sum(self.pca.explained_variance_ratio_) * 100
            print(f"  - Fitted PCA ({self.n_components} components, {explained_var:.1f}% variance explained)")
        
        # K-Means
        self.kmeans.fit(states_reduced)
        
        if verbose:
            print(f"  - Fitted K-Means ({self.n_clusters} clusters)")
        
        # Store cluster centers in original space for visualization
        self.cluster_centers_original = self.pca.inverse_transform(self.kmeans.cluster_centers_)
        self.cluster_centers_original = self.scaler.inverse_transform(self.cluster_centers_original)
        
        self.is_fitted = True
        
        if verbose:
            print("Encoder fitting complete!")
        
        return self
    
    def encode(self, state: np.ndarray) -> int:
        """
        Encode a state to its cluster ID.
        
        Args:
            state: Game state observation
            
        Returns:
            Cluster ID
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        state_flat = state.flatten().reshape(1, -1)
        state_scaled = self.scaler.transform(state_flat)
        state_reduced = self.pca.transform(state_scaled)
        cluster_id = self.kmeans.predict(state_reduced)[0]
        
        return int(cluster_id)
    
    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Encode multiple states to cluster IDs.
        
        Args:
            states: Array of game states
            
        Returns:
            Array of cluster IDs
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        states_flat = states.reshape(len(states), -1)
        states_scaled = self.scaler.transform(states_flat)
        states_reduced = self.pca.transform(states_scaled)
        cluster_ids = self.kmeans.predict(states_reduced)
        
        return cluster_ids
    
    def get_reduced_state(self, state: np.ndarray) -> np.ndarray:
        """
        Get PCA-reduced representation of a state.
        
        Args:
            state: Game state observation
            
        Returns:
            Reduced state vector
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        state_flat = state.flatten().reshape(1, -1)
        state_scaled = self.scaler.transform(state_flat)
        state_reduced = self.pca.transform(state_scaled)
        
        return state_reduced[0]
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about clusters.
        
        Returns:
            Dictionary with cluster information
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        return {
            'n_clusters': self.n_clusters,
            'n_components': self.n_components,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': sum(self.pca.explained_variance_ratio_),
            'inertia': self.kmeans.inertia_
        }
    
    def get_state_cluster_distance(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Get cluster ID and distance to cluster center.
        
        Args:
            state: Game state observation
            
        Returns:
            Tuple of (cluster_id, distance)
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        state_flat = state.flatten().reshape(1, -1)
        state_scaled = self.scaler.transform(state_flat)
        state_reduced = self.pca.transform(state_scaled)
        
        cluster_id = self.kmeans.predict(state_reduced)[0]
        center = self.kmeans.cluster_centers_[cluster_id]
        distance = np.linalg.norm(state_reduced[0] - center)
        
        return int(cluster_id), float(distance)
    
    def save(self, path: str):
        """
        Save encoder to file.
        
        Args:
            path: File path
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        data = {
            'n_components': self.n_components,
            'n_clusters': self.n_clusters,
            'scaler': self.scaler,
            'pca': self.pca,
            'kmeans': self.kmeans,
            'is_fitted': self.is_fitted,
            'cluster_centers_original': self.cluster_centers_original
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Encoder saved to: {path}")
    
    def load(self, path: str):
        """
        Load encoder from file.
        
        Args:
            path: File path
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.n_components = data['n_components']
        self.n_clusters = data['n_clusters']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.kmeans = data['kmeans']
        self.is_fitted = data['is_fitted']
        self.cluster_centers_original = data['cluster_centers_original']
        
        print(f"Encoder loaded from: {path}")
    
    def clear_buffer(self):
        """Clear the state collection buffer."""
        self.state_buffer = []
