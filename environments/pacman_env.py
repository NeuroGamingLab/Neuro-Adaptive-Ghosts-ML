"""
Pacman Gymnasium Environment for Reinforcement Learning
Supports both Pacman agent training and Ghost agent training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any

# Game Constants
CELL_SIZE = 20
GRID_WIDTH = 40
GRID_HEIGHT = 31

# Cell Types
WALL = 1
PATH = 0
DOT = 2
POWER_PELLET = 3

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTION_TO_DIR = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}


class PacmanEnv(gym.Env):
    """
    Pacman environment for training RL agents.
    Can be configured to train either Pacman or Ghost agents.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, agent_type: str = 'ghost', render_mode: Optional[str] = None):
        """
        Initialize the Pacman environment.
        
        Args:
            agent_type: 'pacman' or 'ghost' - which agent to train
            render_mode: 'human' for pygame display, 'rgb_array' for pixel output
        """
        super().__init__()
        
        self.agent_type = agent_type
        self.render_mode = render_mode
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: grid state + positions
        # Flattened grid (40x31) + pacman pos (2) + ghost positions (4*2) + ghost states (4)
        obs_size = GRID_WIDTH * GRID_HEIGHT + 2 + 8 + 4
        self.observation_space = spaces.Box(
            low=-1, high=3, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize game state
        self.maze = None
        self.pacman_pos = None
        self.ghost_positions = None
        self.ghost_directions = None
        self.ghost_frightened = None
        self.dots_remaining = 0
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self.lives = 3
        
        # Pygame for rendering
        self.screen = None
        self.clock = None
        
    def _create_maze(self) -> np.ndarray:
        """Create the maze grid."""
        maze = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
            [1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,2,1,1],
            [1,3,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,3,1,1],
            [1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,2,1,1],
            [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
            [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1],
            [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1],
            [1,2,2,2,2,2,2,1,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,1],
            [1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,1,2,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,2,2,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,2,1,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1],
            [1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
            [1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1],
            [1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1],
            [1,3,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,3,1,1,1],
            [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1,1,1],
            [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1,1,1],
            [1,2,2,2,2,2,2,1,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,1],
            [1,2,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1],
            [1,2,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1],
            [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ], dtype=np.int32)
        return maze
    
    def _is_wall(self, x: int, y: int) -> bool:
        """Check if position is a wall."""
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        return self.maze[y, x] == WALL
    
    def _get_valid_moves(self, x: int, y: int) -> list:
        """Get list of valid moves from a position."""
        valid = []
        for action, (dx, dy) in ACTION_TO_DIR.items():
            new_x, new_y = x + dx, y + dy
            # Handle tunnel wrap-around
            if new_x < 0:
                new_x = GRID_WIDTH - 1
            elif new_x >= GRID_WIDTH:
                new_x = 0
            if not self._is_wall(new_x, new_y):
                valid.append(action)
        return valid
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Flatten maze
        maze_flat = self.maze.flatten().astype(np.float32)
        
        # Normalize pacman position
        pacman_norm = np.array([
            self.pacman_pos[0] / GRID_WIDTH,
            self.pacman_pos[1] / GRID_HEIGHT
        ], dtype=np.float32)
        
        # Normalize ghost positions
        ghost_pos_norm = []
        for gx, gy in self.ghost_positions:
            ghost_pos_norm.extend([gx / GRID_WIDTH, gy / GRID_HEIGHT])
        ghost_pos_norm = np.array(ghost_pos_norm, dtype=np.float32)
        
        # Ghost frightened states
        ghost_states = np.array(self.ghost_frightened, dtype=np.float32)
        
        # Combine all observations
        obs = np.concatenate([maze_flat, pacman_norm, ghost_pos_norm, ghost_states])
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize maze
        self.maze = self._create_maze()
        
        # Count dots
        self.dots_remaining = np.sum((self.maze == DOT) | (self.maze == POWER_PELLET))
        
        # Initialize Pacman position
        self.pacman_pos = [20, 23]
        
        # Initialize ghost positions (in ghost house)
        self.ghost_positions = [
            [18, 14],
            [20, 14],
            [22, 14],
            [20, 12]
        ]
        
        # Initialize ghost directions
        self.ghost_directions = [
            random.choice(list(ACTION_TO_DIR.keys())) for _ in range(4)
        ]
        
        # Ghost frightened states
        self.ghost_frightened = [0, 0, 0, 0]
        
        # Reset counters
        self.score = 0
        self.steps = 0
        self.lives = 3
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        For ghost training: action controls the first ghost
        For pacman training: action controls pacman
        """
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.agent_type == 'ghost':
            # Update ghost 0 with the action
            reward = self._step_ghost(action)
        else:
            # Update pacman with the action
            reward = self._step_pacman(action)
        
        # Check termination conditions
        if self.dots_remaining == 0:
            terminated = True
            reward += 100 if self.agent_type == 'pacman' else -100
        
        if self.lives <= 0:
            terminated = True
            reward += -100 if self.agent_type == 'pacman' else 100
        
        if self.steps >= self.max_steps:
            truncated = True
        
        obs = self._get_observation()
        info = {
            'score': self.score,
            'dots_remaining': self.dots_remaining,
            'lives': self.lives
        }
        
        return obs, reward, terminated, truncated, info
    
    def _step_ghost(self, action: int) -> float:
        """Execute ghost step and return reward."""
        reward = 0.0
        
        # Move ghost 0 with the given action
        gx, gy = self.ghost_positions[0]
        dx, dy = ACTION_TO_DIR.get(action, (0, 0))
        new_x, new_y = gx + dx, gy + dy
        
        # Handle tunnel
        if new_x < 0:
            new_x = GRID_WIDTH - 1
        elif new_x >= GRID_WIDTH:
            new_x = 0
        
        # Only move if not a wall
        if not self._is_wall(new_x, new_y):
            self.ghost_positions[0] = [new_x, new_y]
            
            # Reward for getting closer to Pacman
            old_dist = self._manhattan_distance((gx, gy), tuple(self.pacman_pos))
            new_dist = self._manhattan_distance((new_x, new_y), tuple(self.pacman_pos))
            
            if not self.ghost_frightened[0]:
                reward += (old_dist - new_dist) * 0.1  # Reward for getting closer
        else:
            reward -= 0.1  # Penalty for hitting wall
        
        # Move other ghosts randomly
        for i in range(1, 4):
            self._move_ghost_random(i)
        
        # Move Pacman (simple AI)
        self._move_pacman_ai()
        
        # Check collisions
        collision_reward = self._check_ghost_collision()
        reward += collision_reward
        
        return reward
    
    def _step_pacman(self, action: int) -> float:
        """Execute Pacman step and return reward."""
        reward = -0.01  # Small penalty for each step to encourage efficiency
        
        px, py = self.pacman_pos
        dx, dy = ACTION_TO_DIR.get(action, (0, 0))
        new_x, new_y = px + dx, py + dy
        
        # Handle tunnel
        if new_x < 0:
            new_x = GRID_WIDTH - 1
        elif new_x >= GRID_WIDTH:
            new_x = 0
        
        # Only move if not a wall
        if not self._is_wall(new_x, new_y):
            self.pacman_pos = [new_x, new_y]
            
            # Check for dot collection
            cell = self.maze[new_y, new_x]
            if cell == DOT:
                self.maze[new_y, new_x] = PATH
                self.score += 10
                self.dots_remaining -= 1
                reward += 1.0
            elif cell == POWER_PELLET:
                self.maze[new_y, new_x] = PATH
                self.score += 50
                self.dots_remaining -= 1
                reward += 5.0
                # Make ghosts frightened
                for i in range(4):
                    self.ghost_frightened[i] = 30  # 30 steps of frightened
        
        # Move ghosts (simple AI for now)
        for i in range(4):
            self._move_ghost_ai(i)
        
        # Update frightened timers
        for i in range(4):
            if self.ghost_frightened[i] > 0:
                self.ghost_frightened[i] -= 1
        
        # Check collisions
        collision_reward = self._check_pacman_collision()
        reward += collision_reward
        
        return reward
    
    def _move_ghost_random(self, ghost_idx: int):
        """Move ghost randomly."""
        gx, gy = self.ghost_positions[ghost_idx]
        valid_moves = self._get_valid_moves(gx, gy)
        
        if valid_moves:
            action = random.choice(valid_moves)
            dx, dy = ACTION_TO_DIR[action]
            new_x, new_y = gx + dx, gy + dy
            
            if new_x < 0:
                new_x = GRID_WIDTH - 1
            elif new_x >= GRID_WIDTH:
                new_x = 0
                
            self.ghost_positions[ghost_idx] = [new_x, new_y]
    
    def _move_ghost_ai(self, ghost_idx: int):
        """Move ghost with simple chase AI."""
        gx, gy = self.ghost_positions[ghost_idx]
        px, py = self.pacman_pos
        
        valid_moves = self._get_valid_moves(gx, gy)
        
        if not valid_moves:
            return
        
        if self.ghost_frightened[ghost_idx] > 0:
            # Run away from Pacman
            best_action = max(valid_moves, key=lambda a: 
                self._manhattan_distance(
                    (gx + ACTION_TO_DIR[a][0], gy + ACTION_TO_DIR[a][1]),
                    (px, py)
                )
            )
        else:
            # Chase Pacman (with some randomness)
            if random.random() < 0.2:
                best_action = random.choice(valid_moves)
            else:
                best_action = min(valid_moves, key=lambda a:
                    self._manhattan_distance(
                        (gx + ACTION_TO_DIR[a][0], gy + ACTION_TO_DIR[a][1]),
                        (px, py)
                    )
                )
        
        dx, dy = ACTION_TO_DIR[best_action]
        new_x, new_y = gx + dx, gy + dy
        
        if new_x < 0:
            new_x = GRID_WIDTH - 1
        elif new_x >= GRID_WIDTH:
            new_x = 0
            
        self.ghost_positions[ghost_idx] = [new_x, new_y]
    
    def _move_pacman_ai(self):
        """Simple Pacman AI for ghost training."""
        px, py = self.pacman_pos
        valid_moves = self._get_valid_moves(px, py)
        
        if not valid_moves:
            return
        
        # Find nearest ghost
        min_ghost_dist = float('inf')
        for gx, gy in self.ghost_positions:
            dist = self._manhattan_distance((px, py), (gx, gy))
            min_ghost_dist = min(min_ghost_dist, dist)
        
        # If ghost is close, run away
        if min_ghost_dist < 5:
            best_action = max(valid_moves, key=lambda a:
                min(self._manhattan_distance(
                    (px + ACTION_TO_DIR[a][0], py + ACTION_TO_DIR[a][1]),
                    (gx, gy)
                ) for gx, gy in self.ghost_positions)
            )
        else:
            # Otherwise, go towards nearest dot
            best_action = random.choice(valid_moves)
            min_dot_dist = float('inf')
            
            for action in valid_moves:
                dx, dy = ACTION_TO_DIR[action]
                new_x, new_y = px + dx, py + dy
                
                # Check if there's a dot nearby
                for y in range(max(0, new_y - 3), min(GRID_HEIGHT, new_y + 4)):
                    for x in range(max(0, new_x - 3), min(GRID_WIDTH, new_x + 4)):
                        if self.maze[y, x] in [DOT, POWER_PELLET]:
                            dist = self._manhattan_distance((new_x, new_y), (x, y))
                            if dist < min_dot_dist:
                                min_dot_dist = dist
                                best_action = action
        
        dx, dy = ACTION_TO_DIR[best_action]
        new_x, new_y = px + dx, py + dy
        
        if new_x < 0:
            new_x = GRID_WIDTH - 1
        elif new_x >= GRID_WIDTH:
            new_x = 0
        
        if not self._is_wall(new_x, new_y):
            self.pacman_pos = [new_x, new_y]
            
            # Collect dots
            cell = self.maze[new_y, new_x]
            if cell == DOT:
                self.maze[new_y, new_x] = PATH
                self.dots_remaining -= 1
            elif cell == POWER_PELLET:
                self.maze[new_y, new_x] = PATH
                self.dots_remaining -= 1
                for i in range(4):
                    self.ghost_frightened[i] = 30
    
    def _check_ghost_collision(self) -> float:
        """Check collision for ghost training (ghost catches Pacman)."""
        reward = 0.0
        px, py = self.pacman_pos
        
        for i, (gx, gy) in enumerate(self.ghost_positions):
            if gx == px and gy == py:
                if self.ghost_frightened[i] > 0:
                    # Ghost was caught
                    self.ghost_positions[i] = [18 + i * 2, 14]
                    self.ghost_frightened[i] = 0
                    reward -= 10.0 if i == 0 else 0  # Only penalize trained ghost
                else:
                    # Ghost caught Pacman
                    self.lives -= 1
                    reward += 50.0 if i == 0 else 0  # Only reward trained ghost
                    # Reset positions
                    self.pacman_pos = [20, 23]
                    for j in range(4):
                        self.ghost_positions[j] = [18 + j * 2, 14]
        
        return reward
    
    def _check_pacman_collision(self) -> float:
        """Check collision for Pacman training."""
        reward = 0.0
        px, py = self.pacman_pos
        
        for i, (gx, gy) in enumerate(self.ghost_positions):
            if gx == px and gy == py:
                if self.ghost_frightened[i] > 0:
                    # Ate ghost
                    self.ghost_positions[i] = [18 + i * 2, 14]
                    self.ghost_frightened[i] = 0
                    self.score += 200
                    reward += 20.0
                else:
                    # Caught by ghost
                    self.lives -= 1
                    reward -= 50.0
                    # Reset positions
                    self.pacman_pos = [20, 23]
                    for j in range(4):
                        self.ghost_positions[j] = [18 + j * 2, 14]
        
        return reward
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self._render_pygame()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_pygame(self):
        """Render using pygame."""
        try:
            import pygame
        except ImportError:
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
            pygame.display.set_caption('Pacman RL Environment')
            self.clock = pygame.time.Clock()
        
        # Colors
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        PINK = (255, 192, 203)
        CYAN = (0, 255, 255)
        ORANGE = (255, 165, 0)
        GHOST_COLORS = [RED, PINK, CYAN, ORANGE]
        
        self.screen.fill(BLACK)
        
        # Draw maze
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = self.maze[y, x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if cell == WALL:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif cell == DOT:
                    center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(self.screen, YELLOW, center, 2)
                elif cell == POWER_PELLET:
                    center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(self.screen, YELLOW, center, 6)
        
        # Draw Pacman
        px, py = self.pacman_pos
        center = (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(self.screen, YELLOW, center, CELL_SIZE // 2 - 2)
        
        # Draw ghosts
        for i, (gx, gy) in enumerate(self.ghost_positions):
            center = (gx * CELL_SIZE + CELL_SIZE // 2, gy * CELL_SIZE + CELL_SIZE // 2)
            color = (0, 0, 255) if self.ghost_frightened[i] > 0 else GHOST_COLORS[i]
            pygame.draw.circle(self.screen, color, center, CELL_SIZE // 2 - 2)
            # Eyes
            pygame.draw.circle(self.screen, WHITE, (center[0] - 3, center[1] - 2), 3)
            pygame.draw.circle(self.screen, WHITE, (center[0] + 3, center[1] - 2), 3)
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array."""
        img = np.zeros((GRID_HEIGHT * CELL_SIZE, GRID_WIDTH * CELL_SIZE, 3), dtype=np.uint8)
        
        # Draw maze
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = self.maze[y, x]
                if cell == WALL:
                    img[y*CELL_SIZE:(y+1)*CELL_SIZE, x*CELL_SIZE:(x+1)*CELL_SIZE] = [0, 0, 255]
        
        return img
    
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
