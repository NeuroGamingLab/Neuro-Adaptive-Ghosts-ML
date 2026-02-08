"""
Multi-Agent Ghost System with Coordinated Attacks + Alpha Pack Hunting
Implements pincer movements, role-based strategies, escape route blocking,
and dynamic alpha-follower pack behavior
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from collections import deque

from .ghost_agent import GhostAgent


class CoordinatedGhostSystem:
    """
    Advanced multi-agent system that coordinates ghost attacks.
    
    Strategies:
    - Pincer Movement: Ghosts approach from multiple directions
    - Escape Blocking: Some ghosts cut off escape routes
    - Role Assignment: Chaser, Ambusher, Patrol, Interceptor
    - Predictive Targeting: Anticipate Pacman's movement
    - ALPHA PACK HUNTING: Closest ghost becomes alpha, others follow!
    """
    
    # Ghost roles
    ROLE_CHASER = 'chaser'        # Directly pursues Pacman
    ROLE_AMBUSHER = 'ambusher'    # Predicts and intercepts
    ROLE_BLOCKER = 'blocker'      # Blocks escape routes
    ROLE_PATROL = 'patrol'        # Controls territory
    
    # Alpha pack hunting constants
    ALPHA_ACTIVATION_DISTANCE = 8  # Distance to activate alpha mode
    ALPHA_AGGRESSION_BOOST = 1.5   # Speed multiplier for alpha (conceptual)
    
    def __init__(self, num_ghosts: int = 4, coordination_level: float = 0.8):
        """
        Initialize coordinated ghost system.
        
        Args:
            num_ghosts: Number of ghost agents
            coordination_level: How coordinated the ghosts are (0.0-1.0)
        """
        self.num_ghosts = num_ghosts
        self.coordination_level = coordination_level
        
        # Agent storage
        self.agents: List[Optional[GhostAgent]] = [None] * num_ghosts
        self.agent_types: List[str] = ['coordinated'] * num_ghosts
        
        # Role assignments (default roles for 4 ghosts)
        self.roles = [
            self.ROLE_CHASER,     # Blinky - direct chase
            self.ROLE_AMBUSHER,   # Pinky - predict and intercept
            self.ROLE_BLOCKER,    # Inky - block escape routes
            self.ROLE_PATROL      # Clyde - patrol and cut off
        ]
        
        # Tracking for coordination
        self.pacman_history: deque = deque(maxlen=10)  # Track Pacman's recent positions
        self.ghost_targets: List[Tuple[int, int]] = [(0, 0)] * num_ghosts
        self.last_pacman_pos: Optional[Tuple[int, int]] = None
        self.pacman_velocity: Tuple[int, int] = (0, 0)
        
        # ALPHA PACK HUNTING STATE
        self.alpha_ghost_idx: Optional[int] = None  # Current alpha ghost
        self.alpha_mode_active: bool = False  # Is pack hunting active?
        self.pack_formation: str = 'spread'  # 'spread', 'converge', 'surround'
        
    def update_pacman_tracking(self, pacman_pos: Tuple[int, int]):
        """Update Pacman position tracking for prediction."""
        if self.last_pacman_pos is not None:
            # Calculate velocity
            self.pacman_velocity = (
                pacman_pos[0] - self.last_pacman_pos[0],
                pacman_pos[1] - self.last_pacman_pos[1]
            )
        
        self.pacman_history.append(pacman_pos)
        self.last_pacman_pos = pacman_pos
    
    def detect_alpha(self, pacman_pos: Tuple[int, int], ghost_positions: List[Tuple[int, int]], 
                     frightened: List[bool]) -> Optional[int]:
        """
        Detect which ghost should be the alpha (pack leader).
        Alpha is the closest non-frightened ghost within activation distance.
        """
        min_dist = float('inf')
        alpha_idx = None
        
        for i, (gx, gy) in enumerate(ghost_positions):
            # Skip frightened ghosts - they can't be alpha
            if frightened[i]:
                continue
            
            dist = abs(gx - pacman_pos[0]) + abs(gy - pacman_pos[1])
            
            if dist < min_dist:
                min_dist = dist
                alpha_idx = i
        
        # Only activate alpha mode if within threshold
        if min_dist <= self.ALPHA_ACTIVATION_DISTANCE:
            self.alpha_mode_active = True
            self.alpha_ghost_idx = alpha_idx
            return alpha_idx
        else:
            self.alpha_mode_active = False
            self.alpha_ghost_idx = None
            return None
    
    def get_pack_formation_targets(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        alpha_idx: int,
        maze: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Calculate pack formation targets when alpha mode is active.
        Alpha chases directly, others form a coordinated pincer.
        """
        targets = [pacman_pos] * self.num_ghosts
        
        alpha_pos = ghost_positions[alpha_idx]
        px, py = pacman_pos
        ax, ay = alpha_pos
        
        # Calculate direction from alpha to pacman
        dx = px - ax
        dy = py - ay
        
        # Normalize direction
        length = max(1, abs(dx) + abs(dy))
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Perpendicular direction for flanking
        perp_x = -dy_norm
        perp_y = dx_norm
        
        # Predicted escape direction (opposite of alpha approach)
        escape_x = int(px + dx_norm * 5)
        escape_y = int(py + dy_norm * 5)
        escape_x = max(0, min(39, escape_x))
        escape_y = max(0, min(30, escape_y))
        
        for i in range(self.num_ghosts):
            if i == alpha_idx:
                # ALPHA: Direct aggressive pursuit
                targets[i] = pacman_pos
            else:
                # Determine follower role based on index relative to alpha
                follower_role = (i - alpha_idx) % self.num_ghosts
                
                if follower_role == 1:
                    # FLANKER LEFT: Approach from left side
                    flank_x = int(px + perp_x * 4)
                    flank_y = int(py + perp_y * 4)
                    flank_x = max(0, min(39, flank_x))
                    flank_y = max(0, min(30, flank_y))
                    targets[i] = (flank_x, flank_y)
                    
                elif follower_role == 2:
                    # FLANKER RIGHT: Approach from right side
                    flank_x = int(px - perp_x * 4)
                    flank_y = int(py - perp_y * 4)
                    flank_x = max(0, min(39, flank_x))
                    flank_y = max(0, min(30, flank_y))
                    targets[i] = (flank_x, flank_y)
                    
                else:
                    # CUT-OFF: Block predicted escape route
                    targets[i] = (escape_x, escape_y)
        
        return targets
    
    def predict_pacman_position(self, steps_ahead: int = 4) -> Tuple[int, int]:
        """Predict where Pacman will be in N steps."""
        if self.last_pacman_pos is None:
            return (20, 23)  # Default position
        
        # Simple linear prediction
        predicted_x = self.last_pacman_pos[0] + self.pacman_velocity[0] * steps_ahead
        predicted_y = self.last_pacman_pos[1] + self.pacman_velocity[1] * steps_ahead
        
        # Clamp to grid
        predicted_x = max(0, min(39, predicted_x))
        predicted_y = max(0, min(30, predicted_y))
        
        return (int(predicted_x), int(predicted_y))
    
    def find_escape_routes(
        self, 
        pacman_pos: Tuple[int, int], 
        ghost_positions: List[Tuple[int, int]],
        maze: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find Pacman's likely escape routes."""
        escape_routes = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        px, py = pacman_pos
        
        # Check each direction from Pacman
        for dx, dy in directions:
            # Look several steps ahead
            for dist in range(1, 6):
                nx = px + dx * dist
                ny = py + dy * dist
                
                # Handle wrap-around
                if nx < 0: nx = 39
                elif nx > 39: nx = 0
                
                # Check if valid path
                if 0 <= ny < len(maze) and maze[ny, nx] != 1:
                    # Check if no ghost is blocking this route
                    ghost_blocking = False
                    for gx, gy in ghost_positions:
                        if abs(gx - nx) + abs(gy - ny) < 3:
                            ghost_blocking = True
                            break
                    
                    if not ghost_blocking:
                        escape_routes.append((nx, ny))
                else:
                    break
        
        return escape_routes
    
    def assign_targets(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        maze: np.ndarray
    ):
        """Assign coordinated targets to each ghost based on their role."""
        self.update_pacman_tracking(pacman_pos)
        
        predicted_pos = self.predict_pacman_position(4)
        escape_routes = self.find_escape_routes(pacman_pos, ghost_positions, maze)
        
        for i in range(self.num_ghosts):
            role = self.roles[i]
            gx, gy = ghost_positions[i]
            
            if role == self.ROLE_CHASER:
                # Direct pursuit of Pacman
                self.ghost_targets[i] = pacman_pos
                
            elif role == self.ROLE_AMBUSHER:
                # Target predicted position (intercept)
                self.ghost_targets[i] = predicted_pos
                
            elif role == self.ROLE_BLOCKER:
                # Target the best escape route to block
                if escape_routes:
                    # Find escape route furthest from other ghosts
                    best_route = escape_routes[0]
                    max_ghost_dist = 0
                    
                    # Get other ghost positions (exclude self)
                    other_ghosts = [(gx2, gy2) for gx2, gy2 in ghost_positions if (gx2, gy2) != (gx, gy)]
                    
                    for route in escape_routes:
                        if other_ghosts:
                            min_dist = min(
                                abs(route[0] - gx2) + abs(route[1] - gy2)
                                for gx2, gy2 in other_ghosts
                            )
                        else:
                            # No other ghosts, use distance from Pacman
                            min_dist = abs(route[0] - pacman_pos[0]) + abs(route[1] - pacman_pos[1])
                        
                        if min_dist > max_ghost_dist:
                            max_ghost_dist = min_dist
                            best_route = route
                    
                    self.ghost_targets[i] = best_route
                else:
                    self.ghost_targets[i] = pacman_pos
                    
            elif role == self.ROLE_PATROL:
                # Patrol opposite side of Pacman from other ghosts
                # Create a flanking position
                avg_ghost_x = sum(g[0] for g in ghost_positions) / len(ghost_positions)
                avg_ghost_y = sum(g[1] for g in ghost_positions) / len(ghost_positions)
                
                # Go to opposite side
                flank_x = pacman_pos[0] + (pacman_pos[0] - avg_ghost_x) * 0.5
                flank_y = pacman_pos[1] + (pacman_pos[1] - avg_ghost_y) * 0.5
                
                # Clamp to grid
                flank_x = max(0, min(39, int(flank_x)))
                flank_y = max(0, min(30, int(flank_y)))
                
                self.ghost_targets[i] = (flank_x, flank_y)
    
    def get_coordinated_actions(
        self,
        observations: List[np.ndarray],
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        maze: np.ndarray,
        frightened: List[bool],
        deterministic: bool = True
    ) -> List[int]:
        """
        Get coordinated actions for all ghosts.
        
        Args:
            observations: List of observations for each ghost
            pacman_pos: Pacman position (x, y)
            ghost_positions: List of ghost positions
            maze: Current maze state
            frightened: List of frightened states for each ghost
            deterministic: Use deterministic actions
            
        Returns:
            List of actions for each ghost
        """
        # Detect alpha ghost for pack hunting
        alpha_idx = self.detect_alpha(pacman_pos, ghost_positions, frightened)
        
        # Choose targeting strategy
        if self.alpha_mode_active and alpha_idx is not None:
            # ALPHA PACK HUNTING MODE - coordinated pack attack!
            self.ghost_targets = self.get_pack_formation_targets(
                pacman_pos, ghost_positions, alpha_idx, maze
            )
        else:
            # Standard role-based coordination
            self.assign_targets(pacman_pos, ghost_positions, maze)
        
        actions = []
        
        for i in range(self.num_ghosts):
            gx, gy = ghost_positions[i]
            
            # If frightened, run away
            if frightened[i]:
                action = self._flee_action(gx, gy, pacman_pos, maze)
            # Check if we should use trained agent
            elif self.agents[i] is not None and self.agent_types[i] == 'trained':
                action = self.agents[i].predict(observations[i], deterministic=deterministic)
            # ALPHA gets higher coordination (more aggressive)
            elif self.alpha_mode_active and i == alpha_idx:
                # Alpha always moves toward target (no randomness)
                action = self._move_toward_target(gx, gy, self.ghost_targets[i], maze)
            # Use coordinated targeting
            elif np.random.random() < self.coordination_level:
                action = self._move_toward_target(gx, gy, self.ghost_targets[i], maze)
            # Random fallback
            else:
                action = self._random_action(gx, gy, maze)
            
            actions.append(action)
        
        return actions
    
    def _move_toward_target(
        self, 
        gx: int, 
        gy: int, 
        target: Tuple[int, int], 
        maze: np.ndarray
    ) -> int:
        """Get action to move toward target position."""
        valid_actions = self._get_valid_actions((gx, gy), maze)
        
        if not valid_actions:
            return 0
        
        tx, ty = target
        
        # Find action that minimizes distance to target
        best_action = valid_actions[0]
        min_dist = float('inf')
        
        directions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        
        for action in valid_actions:
            dx, dy = directions[action]
            nx = gx + dx
            ny = gy + dy
            
            # Handle wrap-around
            if nx < 0: nx = 39
            elif nx > 39: nx = 0
            
            dist = abs(nx - tx) + abs(ny - ty)
            if dist < min_dist:
                min_dist = dist
                best_action = action
        
        return best_action
    
    def _flee_action(
        self, 
        gx: int, 
        gy: int, 
        pacman_pos: Tuple[int, int], 
        maze: np.ndarray
    ) -> int:
        """Get action to flee from Pacman."""
        valid_actions = self._get_valid_actions((gx, gy), maze)
        
        if not valid_actions:
            return 0
        
        px, py = pacman_pos
        
        # Find action that maximizes distance from Pacman
        best_action = valid_actions[0]
        max_dist = -1
        
        directions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        
        for action in valid_actions:
            dx, dy = directions[action]
            nx = gx + dx
            ny = gy + dy
            
            if nx < 0: nx = 39
            elif nx > 39: nx = 0
            
            dist = abs(nx - px) + abs(ny - py)
            if dist > max_dist:
                max_dist = dist
                best_action = action
        
        return best_action
    
    def _random_action(self, gx: int, gy: int, maze: np.ndarray) -> int:
        """Get random valid action."""
        valid_actions = self._get_valid_actions((gx, gy), maze)
        if valid_actions:
            return np.random.choice(valid_actions)
        return 0
    
    def _get_valid_actions(self, pos: Tuple[int, int], maze: np.ndarray) -> List[int]:
        """Get list of valid actions from position."""
        directions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        
        valid = []
        for action, (dx, dy) in directions.items():
            nx = pos[0] + dx
            ny = pos[1] + dy
            
            if nx < 0: nx = 39
            elif nx > 39: nx = 0
            
            if 0 <= ny < len(maze) and maze[ny, nx] != 1:
                valid.append(action)
        
        return valid
    
    def load_agent(self, ghost_idx: int, model_path: str, algorithm: str = 'ppo'):
        """Load a trained agent for a specific ghost."""
        if ghost_idx < 0 or ghost_idx >= self.num_ghosts:
            raise ValueError(f"Invalid ghost index: {ghost_idx}")
        
        self.agents[ghost_idx] = GhostAgent(algorithm=algorithm, model_path=model_path)
        self.agent_types[ghost_idx] = 'trained'
        print(f"Loaded trained agent for ghost {ghost_idx}")
    
    def set_role(self, ghost_idx: int, role: str):
        """Set the role for a ghost."""
        valid_roles = [self.ROLE_CHASER, self.ROLE_AMBUSHER, self.ROLE_BLOCKER, self.ROLE_PATROL]
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Choose from: {valid_roles}")
        
        self.roles[ghost_idx] = role
        print(f"Ghost {ghost_idx} role set to: {role}")
    
    def set_coordination_level(self, level: float):
        """Set coordination level (0.0-1.0)."""
        self.coordination_level = max(0.0, min(1.0, level))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'num_ghosts': self.num_ghosts,
            'roles': self.roles.copy(),
            'coordination_level': self.coordination_level,
            'current_targets': [tuple(t) for t in self.ghost_targets],
            'pacman_velocity': self.pacman_velocity,
            'agent_types': self.agent_types.copy(),
            'alpha_mode_active': self.alpha_mode_active,
            'alpha_ghost_idx': self.alpha_ghost_idx,
            'pack_formation': self.pack_formation
        }
    
    def get_alpha_status(self) -> Tuple[bool, Optional[int]]:
        """Get current alpha hunting status."""
        return self.alpha_mode_active, self.alpha_ghost_idx
    
    def is_alpha(self, ghost_idx: int) -> bool:
        """Check if a ghost is currently the alpha."""
        return self.alpha_mode_active and self.alpha_ghost_idx == ghost_idx


# Backward compatibility alias
class MultiAgentGhostSystem(CoordinatedGhostSystem):
    """Alias for backward compatibility."""
    
    def get_actions(
        self,
        observations: List[np.ndarray],
        pacman_pos: tuple,
        ghost_positions: List[tuple],
        maze: np.ndarray,
        deterministic: bool = True
    ) -> List[int]:
        """Backward compatible method."""
        frightened = [False] * self.num_ghosts
        return self.get_coordinated_actions(
            observations, pacman_pos, ghost_positions, maze, frightened, deterministic
        )