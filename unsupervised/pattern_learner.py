"""
Pattern Learner
Discovers patterns in player/ghost behavior using unsupervised learning
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pickle
import os

try:
    from sklearn.cluster import DBSCAN, HDBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import LocalOutlierFactor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class PatternLearner:
    """
    Learns behavioral patterns from game data using unsupervised methods.
    Uses DBSCAN/HDBSCAN for trajectory clustering and LOF for anomaly detection.
    """
    
    def __init__(
        self,
        trajectory_length: int = 10,
        min_cluster_size: int = 5,
        use_hdbscan: bool = False
    ):
        """
        Initialize the pattern learner.
        
        Args:
            trajectory_length: Number of steps to consider as a trajectory
            min_cluster_size: Minimum cluster size for DBSCAN/HDBSCAN
            use_hdbscan: Use HDBSCAN instead of DBSCAN
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.trajectory_length = trajectory_length
        self.min_cluster_size = min_cluster_size
        self.use_hdbscan = use_hdbscan and HAS_HDBSCAN
        
        self.scaler = StandardScaler()
        
        if self.use_hdbscan:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3
            )
        else:
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=min_cluster_size
            )
        
        self.anomaly_detector = LocalOutlierFactor(
            n_neighbors=20,
            novelty=True
        )
        
        self.trajectory_buffer: List[List[Tuple[int, int]]] = []
        self.current_trajectory: List[Tuple[int, int]] = []
        
        self.is_fitted = False
        self.pattern_stats: Dict[int, Dict[str, Any]] = {}
        
    def add_position(self, position: Tuple[int, int]):
        """
        Add a position to the current trajectory.
        
        Args:
            position: (x, y) position
        """
        self.current_trajectory.append(position)
        
        if len(self.current_trajectory) >= self.trajectory_length:
            self.trajectory_buffer.append(self.current_trajectory.copy())
            self.current_trajectory = self.current_trajectory[1:]  # Sliding window
    
    def add_trajectory(self, trajectory: List[Tuple[int, int]]):
        """
        Add a complete trajectory.
        
        Args:
            trajectory: List of (x, y) positions
        """
        if len(trajectory) >= self.trajectory_length:
            # Split into fixed-length chunks
            for i in range(0, len(trajectory) - self.trajectory_length + 1):
                self.trajectory_buffer.append(trajectory[i:i + self.trajectory_length])
    
    def _trajectory_to_features(self, trajectory: List[Tuple[int, int]]) -> np.ndarray:
        """
        Convert trajectory to feature vector.
        
        Features include:
        - Flattened positions
        - Velocity (differences between consecutive positions)
        - Total distance traveled
        - Direction changes
        """
        positions = np.array(trajectory)
        
        # Flattened positions
        flat_positions = positions.flatten()
        
        # Velocities
        velocities = np.diff(positions, axis=0)
        flat_velocities = velocities.flatten()
        
        # Total distance
        total_distance = np.sum(np.abs(velocities))
        
        # Direction changes (number of times direction changed)
        if len(velocities) > 1:
            direction_changes = np.sum(
                (velocities[1:, 0] != velocities[:-1, 0]) | 
                (velocities[1:, 1] != velocities[:-1, 1])
            )
        else:
            direction_changes = 0
        
        # Combine features
        features = np.concatenate([
            flat_positions,
            flat_velocities,
            [total_distance, direction_changes]
        ])
        
        return features
    
    def fit(self, trajectories: Optional[List[List[Tuple[int, int]]]] = None, verbose: bool = True):
        """
        Fit the pattern learner on trajectories.
        
        Args:
            trajectories: Optional list of trajectories (uses buffer if not provided)
            verbose: Print progress information
        """
        if trajectories is None:
            if len(self.trajectory_buffer) == 0:
                raise ValueError("No trajectories collected. Add trajectories first.")
            trajectories = self.trajectory_buffer
        
        if verbose:
            print(f"Fitting pattern learner on {len(trajectories)} trajectories...")
        
        # Convert to features
        features = np.array([self._trajectory_to_features(t) for t in trajectories])
        
        # Standardize
        features_scaled = self.scaler.fit_transform(features)
        
        if verbose:
            print("  - Computed features and scaled")
        
        # Cluster
        labels = self.clusterer.fit_predict(features_scaled)
        
        if verbose:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"  - Found {n_clusters} patterns ({n_noise} noise points)")
        
        # Fit anomaly detector
        self.anomaly_detector.fit(features_scaled)
        
        if verbose:
            print("  - Fitted anomaly detector")
        
        # Compute pattern statistics
        self._compute_pattern_stats(trajectories, labels)
        
        self.is_fitted = True
        
        if verbose:
            print("Pattern learning complete!")
        
        return labels
    
    def _compute_pattern_stats(
        self,
        trajectories: List[List[Tuple[int, int]]],
        labels: np.ndarray
    ):
        """Compute statistics for each pattern cluster."""
        self.pattern_stats = {}
        
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_trajectories = [trajectories[i] for i in cluster_indices]
            
            # Compute mean trajectory
            mean_trajectory = np.mean([np.array(t) for t in cluster_trajectories], axis=0)
            
            # Compute average speed
            avg_speeds = []
            for t in cluster_trajectories:
                positions = np.array(t)
                velocities = np.diff(positions, axis=0)
                speed = np.mean(np.sqrt(np.sum(velocities**2, axis=1)))
                avg_speeds.append(speed)
            
            self.pattern_stats[label] = {
                'count': len(cluster_indices),
                'mean_trajectory': mean_trajectory.tolist(),
                'avg_speed': np.mean(avg_speeds),
                'speed_std': np.std(avg_speeds)
            }
    
    def predict_pattern(self, trajectory: List[Tuple[int, int]]) -> int:
        """
        Predict which pattern a trajectory belongs to.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Pattern ID (-1 for noise/unknown)
        """
        if not self.is_fitted:
            raise ValueError("Pattern learner not fitted. Call fit() first.")
        
        if len(trajectory) < self.trajectory_length:
            return -1
        
        # Use first trajectory_length positions
        trajectory = trajectory[:self.trajectory_length]
        features = self._trajectory_to_features(trajectory).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Find nearest cluster (for DBSCAN, we need to check distance to cluster centers)
        if hasattr(self.clusterer, 'core_sample_indices_'):
            # Use core samples for DBSCAN
            core_samples = self.clusterer.components_
            if len(core_samples) > 0:
                distances = np.linalg.norm(core_samples - features_scaled, axis=1)
                nearest_idx = np.argmin(distances)
                if distances[nearest_idx] < self.clusterer.eps:
                    return int(self.clusterer.labels_[self.clusterer.core_sample_indices_[nearest_idx]])
        
        return -1
    
    def is_anomaly(self, trajectory: List[Tuple[int, int]]) -> bool:
        """
        Check if a trajectory is anomalous.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            True if trajectory is an anomaly
        """
        if not self.is_fitted:
            raise ValueError("Pattern learner not fitted. Call fit() first.")
        
        if len(trajectory) < self.trajectory_length:
            return True  # Too short trajectories are considered anomalous
        
        trajectory = trajectory[:self.trajectory_length]
        features = self._trajectory_to_features(trajectory).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.anomaly_detector.predict(features_scaled)
        return prediction[0] == -1
    
    def get_anomaly_score(self, trajectory: List[Tuple[int, int]]) -> float:
        """
        Get anomaly score for a trajectory.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Pattern learner not fitted. Call fit() first.")
        
        if len(trajectory) < self.trajectory_length:
            return float('inf')
        
        trajectory = trajectory[:self.trajectory_length]
        features = self._trajectory_to_features(trajectory).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Negative score (LOF returns negative for outliers)
        score = -self.anomaly_detector.score_samples(features_scaled)[0]
        return float(score)
    
    def get_pattern_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about discovered patterns."""
        return self.pattern_stats.copy()
    
    def save(self, path: str):
        """Save pattern learner to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        data = {
            'trajectory_length': self.trajectory_length,
            'min_cluster_size': self.min_cluster_size,
            'scaler': self.scaler,
            'clusterer': self.clusterer,
            'anomaly_detector': self.anomaly_detector,
            'is_fitted': self.is_fitted,
            'pattern_stats': self.pattern_stats
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Pattern learner saved to: {path}")
    
    def load(self, path: str):
        """Load pattern learner from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.trajectory_length = data['trajectory_length']
        self.min_cluster_size = data['min_cluster_size']
        self.scaler = data['scaler']
        self.clusterer = data['clusterer']
        self.anomaly_detector = data['anomaly_detector']
        self.is_fitted = data['is_fitted']
        self.pattern_stats = data['pattern_stats']
        
        print(f"Pattern learner loaded from: {path}")
    
    def clear_buffer(self):
        """Clear trajectory buffers."""
        self.trajectory_buffer = []
        self.current_trajectory = []
