"""
Neural Pathfinding Module

Replaces traditional A* pathfinding with a learned neural network that can:
- Find creative/unexpected paths
- Adapt to player tendencies
- Learn from both A* demonstrations and gameplay experience

Architecture:
- Input: Local view of maze (walls, dots, ghosts, pacman) + position encodings
- Output: Direction probabilities (UP, DOWN, LEFT, RIGHT)
"""

import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Neural pathfinding disabled.")


# Direction mappings
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
DIR_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']


class PathfindingNet(nn.Module):
    """
    Neural network for pathfinding decisions.
    
    Takes a local view of the maze and outputs direction probabilities.
    Uses a CNN for spatial features + MLP for decision making.
    """
    
    def __init__(self, view_size=11, hidden_dim=128):
        super().__init__()
        self.view_size = view_size
        
        # CNN for processing local maze view
        # Input channels: walls, dots, power_pellets, ghost_positions, pacman, self_position
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = 64 * view_size * view_size
        
        # MLP for decision making
        self.fc1 = nn.Linear(conv_output_size + 4, hidden_dim)  # +4 for relative position
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4 directions
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, local_view, relative_pos):
        """
        Forward pass.
        
        Args:
            local_view: (batch, 6, view_size, view_size) - local maze view
            relative_pos: (batch, 4) - relative position to target (dx, dy, dist, angle)
            
        Returns:
            direction_probs: (batch, 4) - probability for each direction
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(local_view)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Concatenate with relative position
        x = torch.cat([x, relative_pos], dim=1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x  # Raw logits (use softmax for probabilities)


class NeuralPathfinder:
    """
    Neural pathfinding system that learns to navigate mazes.
    
    Features:
    - Imitation learning from A* paths
    - Online learning from gameplay
    - Fallback to A* when uncertain
    """
    
    def __init__(self, view_size=11, model_path=None):
        self.view_size = view_size
        self.enabled = HAS_TORCH
        self.model = None
        self.optimizer = None
        self.training_data = []
        self.confidence_threshold = 0.6  # Fallback to A* below this
        
        if self.enabled:
            self.model = PathfindingNet(view_size=view_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Load pre-trained model if exists
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
                print(f"Loaded neural pathfinder from {model_path}")
            else:
                print("Neural Pathfinder initialized (untrained)")
                
            self.model.eval()  # Start in eval mode
    
    def get_local_view(self, maze, ghost_x, ghost_y, pacman_x, pacman_y, all_ghosts):
        """
        Extract local view around the ghost for neural network input.
        
        Returns 6-channel tensor:
        - Channel 0: Walls (1 = wall, 0 = empty)
        - Channel 1: Dots
        - Channel 2: Power pellets
        - Channel 3: Other ghost positions
        - Channel 4: Pacman position
        - Channel 5: Self position (center)
        """
        half_view = self.view_size // 2
        view = np.zeros((6, self.view_size, self.view_size), dtype=np.float32)
        
        grid_height = len(maze.grid)
        grid_width = len(maze.grid[0]) if grid_height > 0 else 0
        
        for dy in range(-half_view, half_view + 1):
            for dx in range(-half_view, half_view + 1):
                mx = ghost_x + dx
                my = ghost_y + dy
                vx = dx + half_view
                vy = dy + half_view
                
                # Check bounds
                if 0 <= mx < grid_width and 0 <= my < grid_height:
                    cell = maze.grid[my][mx]
                    
                    # Channel 0: Walls
                    if cell == 1:
                        view[0, vy, vx] = 1.0
                    # Channel 1: Dots
                    elif cell == 2:
                        view[1, vy, vx] = 1.0
                    # Channel 2: Power pellets
                    elif cell == 3:
                        view[2, vy, vx] = 1.0
                else:
                    # Out of bounds = wall
                    view[0, vy, vx] = 1.0
        
        # Channel 3: Other ghost positions
        for gx, gy in all_ghosts:
            dx = gx - ghost_x + half_view
            dy = gy - ghost_y + half_view
            if 0 <= dx < self.view_size and 0 <= dy < self.view_size:
                view[3, dy, dx] = 1.0
        
        # Channel 4: Pacman position
        px = pacman_x - ghost_x + half_view
        py = pacman_y - ghost_y + half_view
        if 0 <= px < self.view_size and 0 <= py < self.view_size:
            view[4, py, px] = 1.0
        
        # Channel 5: Self position (always center)
        view[5, half_view, half_view] = 1.0
        
        return view
    
    def get_relative_position(self, ghost_x, ghost_y, target_x, target_y):
        """
        Get relative position features to target.
        
        Returns: [dx_norm, dy_norm, distance_norm, angle_norm]
        """
        dx = target_x - ghost_x
        dy = target_y - ghost_y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        
        # Normalize
        dx_norm = dx / 40.0  # Assume max grid ~40
        dy_norm = dy / 30.0
        dist_norm = dist / 50.0
        angle = np.arctan2(dy, dx) / np.pi  # Normalize to [-1, 1]
        
        return np.array([dx_norm, dy_norm, dist_norm, angle], dtype=np.float32)
    
    def get_direction(self, maze, ghost_x, ghost_y, target_x, target_y, all_ghosts=None):
        """
        Get the best direction to move using the neural network.
        
        Args:
            maze: The game maze
            ghost_x, ghost_y: Current ghost position
            target_x, target_y: Target position (usually Pacman)
            all_ghosts: List of (x, y) for all ghosts (for collision avoidance)
            
        Returns:
            (dx, dy): Direction to move, or None to use fallback
        """
        if not self.enabled or self.model is None:
            return None  # Fallback to A*
        
        all_ghosts = all_ghosts or []
        
        # Prepare input
        local_view = self.get_local_view(
            maze, ghost_x, ghost_y, target_x, target_y, all_ghosts
        )
        relative_pos = self.get_relative_position(
            ghost_x, ghost_y, target_x, target_y
        )
        
        # Convert to tensors
        view_tensor = torch.FloatTensor(local_view).unsqueeze(0)
        pos_tensor = torch.FloatTensor(relative_pos).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(view_tensor, pos_tensor)
            probs = F.softmax(logits, dim=1).squeeze().numpy()
        
        # Filter valid moves (not into walls)
        valid_probs = []
        valid_dirs = []
        
        for i, (dx, dy) in enumerate(DIRECTIONS):
            new_x = ghost_x + dx
            new_y = ghost_y + dy
            
            # Check if valid move
            if self._is_valid_move(maze, new_x, new_y):
                valid_probs.append(probs[i])
                valid_dirs.append((dx, dy))
        
        if not valid_dirs:
            return None  # No valid moves
        
        # Normalize valid probabilities
        valid_probs = np.array(valid_probs)
        valid_probs = valid_probs / (valid_probs.sum() + 1e-6)
        
        # Check confidence
        max_prob = np.max(valid_probs)
        if max_prob < self.confidence_threshold:
            return None  # Low confidence, fallback to A*
        
        # Select best direction
        best_idx = np.argmax(valid_probs)
        return valid_dirs[best_idx]
    
    def _is_valid_move(self, maze, x, y):
        """Check if position is valid (not a wall)."""
        if x < 0 or y < 0:
            return False
        try:
            return maze.grid[y][x] != 1
        except IndexError:
            return False
    
    def collect_training_sample(self, maze, ghost_x, ghost_y, target_x, target_y, 
                                 all_ghosts, correct_direction):
        """
        Collect a training sample from A* or optimal play.
        
        Args:
            correct_direction: The correct direction from A* (dx, dy)
        """
        if not self.enabled:
            return
        
        local_view = self.get_local_view(
            maze, ghost_x, ghost_y, target_x, target_y, all_ghosts
        )
        relative_pos = self.get_relative_position(
            ghost_x, ghost_y, target_x, target_y
        )
        
        # Convert direction to index
        try:
            dir_idx = DIRECTIONS.index(correct_direction)
        except ValueError:
            return  # Invalid direction
        
        self.training_data.append({
            'view': local_view,
            'pos': relative_pos,
            'label': dir_idx
        })
        
        # Train periodically
        if len(self.training_data) >= 64:
            self.train_batch()
    
    def train_batch(self):
        """Train on collected data."""
        if not self.enabled or len(self.training_data) < 32:
            return
        
        self.model.train()
        
        # Sample batch
        batch_size = min(64, len(self.training_data))
        indices = np.random.choice(len(self.training_data), batch_size, replace=False)
        
        views = torch.FloatTensor(np.array([self.training_data[i]['view'] for i in indices]))
        positions = torch.FloatTensor(np.array([self.training_data[i]['pos'] for i in indices]))
        labels = torch.LongTensor([self.training_data[i]['label'] for i in indices])
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(views, positions)
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()
        
        # Clear old data (keep some for diversity)
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-500:]
        
        return loss.item()
    
    def save_model(self, path):
        """Save model weights."""
        if self.enabled and self.model:
            torch.save(self.model.state_dict(), path)
            print(f"Neural pathfinder saved to {path}")
    
    def load_model(self, path):
        """Load model weights."""
        if self.enabled and self.model and os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()


class HybridPathfinder:
    """
    Combines neural pathfinding with A* for robust navigation.
    
    - Uses neural network for fast, adaptive decisions
    - Falls back to A* when neural net is uncertain
    - Learns from A* demonstrations to improve over time
    """
    
    def __init__(self, maze_width=40, maze_height=30, model_path=None):
        self.neural = NeuralPathfinder(view_size=11, model_path=model_path)
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.use_neural = True
        self.neural_success_count = 0
        self.fallback_count = 0
        
        print(f"Hybrid Pathfinder initialized (Neural: {self.neural.enabled})")
    
    def get_direction(self, maze, ghost_x, ghost_y, target_x, target_y, 
                      all_ghosts=None, ghost_mode='chase'):
        """
        Get best direction using hybrid approach.
        
        Returns: (dx, dy) direction tuple
        """
        all_ghosts = all_ghosts or []
        
        # Try neural pathfinder first
        if self.use_neural and self.neural.enabled:
            neural_dir = self.neural.get_direction(
                maze, ghost_x, ghost_y, target_x, target_y, all_ghosts
            )
            
            if neural_dir is not None:
                self.neural_success_count += 1
                return neural_dir
        
        # Fallback to A*
        self.fallback_count += 1
        astar_dir = self._astar_direction(maze, ghost_x, ghost_y, target_x, target_y)
        
        # Collect training sample from A*
        if astar_dir and self.neural.enabled:
            self.neural.collect_training_sample(
                maze, ghost_x, ghost_y, target_x, target_y, all_ghosts, astar_dir
            )
        
        return astar_dir or (0, 0)
    
    def _astar_direction(self, maze, start_x, start_y, goal_x, goal_y):
        """Simple A* to get next direction."""
        import heapq
        
        def heuristic(x, y):
            return abs(x - goal_x) + abs(y - goal_y)
        
        def is_valid(x, y):
            if x < 0 or y < 0:
                return False
            try:
                return maze.grid[y][x] != 1
            except IndexError:
                return False
        
        # A* search
        open_set = [(heuristic(start_x, start_y), 0, start_x, start_y, None)]
        closed_set = set()
        came_from = {}
        
        while open_set:
            _, cost, x, y, first_dir = heapq.heappop(open_set)
            
            if (x, y) in closed_set:
                continue
            closed_set.add((x, y))
            
            # Goal reached
            if x == goal_x and y == goal_y:
                return first_dir
            
            # Explore neighbors
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) not in closed_set and is_valid(nx, ny):
                    new_cost = cost + 1
                    # Track first direction from start
                    new_first_dir = first_dir if first_dir else (dx, dy)
                    heapq.heappush(open_set, (
                        new_cost + heuristic(nx, ny),
                        new_cost, nx, ny, new_first_dir
                    ))
        
        # No path found, move towards target directly
        dx = np.sign(goal_x - start_x)
        dy = np.sign(goal_y - start_y)
        
        if dx != 0 and is_valid(start_x + dx, start_y):
            return (dx, 0)
        if dy != 0 and is_valid(start_x, start_y + dy):
            return (0, dy)
        
        return None
    
    def get_stats(self):
        """Get pathfinding statistics."""
        total = self.neural_success_count + self.fallback_count
        if total == 0:
            return "No pathfinding yet"
        
        neural_pct = (self.neural_success_count / total) * 100
        return f"Neural: {neural_pct:.1f}% ({self.neural_success_count}/{total})"
    
    def toggle_neural(self):
        """Toggle neural pathfinding on/off."""
        self.use_neural = not self.use_neural
        return self.use_neural
    
    def save(self, path):
        """Save the neural model."""
        self.neural.save_model(path)


# Convenience function for creating pathfinder
def create_pathfinder(model_path=None):
    """Create a hybrid pathfinder instance."""
    return HybridPathfinder(model_path=model_path)
