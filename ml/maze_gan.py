"""
GAN-based Procedural Maze Generator for Pacman
Uses a Generative Adversarial Network to create new, playable maze layouts
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Optional
import random

# Maze dimensions
MAZE_HEIGHT = 31
MAZE_WIDTH = 40

# Cell types
EMPTY = 0
WALL = 1
DOT = 2
POWER_PELLET = 3


class MazeGenerator(nn.Module):
    """
    Generator network that creates maze layouts from random noise.
    Uses transposed convolutions to upsample from latent space to maze.
    """
    
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Calculate initial size for upsampling
        self.init_height = MAZE_HEIGHT // 4
        self.init_width = MAZE_WIDTH // 4
        
        # Project and reshape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 4 * self.init_height * self.init_width),
            nn.BatchNorm1d(feature_maps * 4 * self.init_height * self.init_width),
            nn.ReLU(True)
        )
        
        # Transposed conv layers for upsampling
        self.conv_blocks = nn.Sequential(
            # Upsample to (H/2, W/2)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Upsample to (H, W)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Final conv to get maze channels (4 cell types)
            nn.Conv2d(feature_maps, 4, 3, 1, 1, bias=False),
        )
        
        self.feature_maps = feature_maps
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate maze from latent vector z."""
        batch_size = z.size(0)
        
        # Project to initial feature map
        x = self.fc(z)
        x = x.view(batch_size, self.feature_maps * 4, self.init_height, self.init_width)
        
        # Upsample
        x = self.conv_blocks(x)
        
        # Adjust to exact maze size
        x = nn.functional.interpolate(x, size=(MAZE_HEIGHT, MAZE_WIDTH), mode='bilinear', align_corners=False)
        
        # Softmax over cell types
        x = torch.softmax(x, dim=1)
        
        return x


class MazeDiscriminator(nn.Module):
    """
    Discriminator network that distinguishes real mazes from generated ones.
    Uses convolutions to analyze maze structure.
    """
    
    def __init__(self, feature_maps: int = 64):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # Input: 4 channels (one-hot encoded cell types)
            nn.Conv2d(4, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate flattened size
        self.flat_size = feature_maps * 4 * (MAZE_HEIGHT // 8) * (MAZE_WIDTH // 8)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, maze: torch.Tensor) -> torch.Tensor:
        """Classify maze as real or fake."""
        x = self.conv_blocks(maze)
        x = nn.functional.adaptive_avg_pool2d(x, (MAZE_HEIGHT // 8, MAZE_WIDTH // 8))
        x = self.fc(x)
        return x


class MazeGAN:
    """
    Complete GAN system for maze generation with training and generation methods.
    """
    
    def __init__(self, latent_dim: int = 100, device: str = 'cpu'):
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        
        # Initialize networks
        self.generator = MazeGenerator(latent_dim).to(self.device)
        self.discriminator = MazeDiscriminator().to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
    
    def maze_to_onehot(self, maze: np.ndarray) -> torch.Tensor:
        """Convert maze grid to one-hot encoded tensor."""
        onehot = np.zeros((4, MAZE_HEIGHT, MAZE_WIDTH), dtype=np.float32)
        for i in range(4):
            onehot[i] = (maze == i).astype(np.float32)
        return torch.from_numpy(onehot)
    
    def onehot_to_maze(self, onehot: torch.Tensor) -> np.ndarray:
        """Convert one-hot tensor back to maze grid."""
        return onehot.argmax(dim=0).cpu().numpy().astype(np.int32)
    
    def generate_maze(self, validate: bool = True) -> np.ndarray:
        """Generate a new maze using the GAN."""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate from random noise
            z = torch.randn(1, self.latent_dim, device=self.device)
            generated = self.generator(z)
            
            # Convert to maze
            maze = self.onehot_to_maze(generated[0])
        
        if validate:
            maze = self.validate_and_fix_maze(maze)
        
        return maze
    
    def validate_and_fix_maze(self, maze: np.ndarray) -> np.ndarray:
        """Ensure maze is playable with proper structure."""
        maze = maze.copy()
        
        # 1. Ensure border walls
        maze[0, :] = WALL
        maze[-1, :] = WALL
        maze[:, 0] = WALL
        maze[:, -1] = WALL
        
        # 2. Create ghost house in center
        center_y, center_x = MAZE_HEIGHT // 2, MAZE_WIDTH // 2
        ghost_house_y = center_y - 2
        ghost_house_x = center_x - 4
        
        # Ghost house walls
        for y in range(ghost_house_y, ghost_house_y + 5):
            for x in range(ghost_house_x, ghost_house_x + 9):
                if 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH:
                    if y == ghost_house_y or y == ghost_house_y + 4:
                        maze[y, x] = WALL
                    elif x == ghost_house_x or x == ghost_house_x + 8:
                        maze[y, x] = WALL
                    else:
                        maze[y, x] = EMPTY
        
        # Ghost house entrance
        maze[ghost_house_y, center_x - 1:center_x + 2] = EMPTY
        
        # 3. Create tunnels on sides
        tunnel_y = MAZE_HEIGHT // 2
        maze[tunnel_y, 0:3] = EMPTY
        maze[tunnel_y, -3:] = EMPTY
        
        # 4. Ensure Pacman start area is clear
        pacman_y = MAZE_HEIGHT - 8
        pacman_x = MAZE_WIDTH // 2
        for dy in range(-1, 2):
            for dx in range(-2, 3):
                y, x = pacman_y + dy, pacman_x + dx
                if 0 < y < MAZE_HEIGHT - 1 and 0 < x < MAZE_WIDTH - 1:
                    if maze[y, x] == WALL:
                        maze[y, x] = DOT
        
        # 5. Add power pellets in corners
        corners = [(3, 1), (3, MAZE_WIDTH - 2), (MAZE_HEIGHT - 4, 1), (MAZE_HEIGHT - 4, MAZE_WIDTH - 2)]
        for y, x in corners:
            if maze[y, x] != WALL:
                maze[y, x] = POWER_PELLET
        
        # 6. Convert remaining empty spaces to dots (except ghost house)
        for y in range(1, MAZE_HEIGHT - 1):
            for x in range(1, MAZE_WIDTH - 1):
                # Skip ghost house area
                if ghost_house_y <= y <= ghost_house_y + 4 and ghost_house_x <= x <= ghost_house_x + 8:
                    continue
                if maze[y, x] == EMPTY:
                    maze[y, x] = DOT
        
        # 7. Ensure connectivity using flood fill
        maze = self.ensure_connectivity(maze)
        
        return maze
    
    def ensure_connectivity(self, maze: np.ndarray) -> np.ndarray:
        """Ensure all non-wall areas are connected."""
        maze = maze.copy()
        
        # Find Pacman start position
        start_y = MAZE_HEIGHT - 8
        start_x = MAZE_WIDTH // 2
        
        # Flood fill from start
        visited = np.zeros_like(maze, dtype=bool)
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= MAZE_HEIGHT or x < 0 or x >= MAZE_WIDTH:
                continue
            if visited[y, x] or maze[y, x] == WALL:
                continue
            
            visited[y, x] = True
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        
        # Find unreachable dots and create paths to them
        for y in range(1, MAZE_HEIGHT - 1):
            for x in range(1, MAZE_WIDTH - 1):
                if maze[y, x] in [DOT, POWER_PELLET, EMPTY] and not visited[y, x]:
                    # Create a path to nearest visited cell
                    self._create_path_to_visited(maze, visited, y, x)
        
        return maze
    
    def _create_path_to_visited(self, maze: np.ndarray, visited: np.ndarray, 
                                 start_y: int, start_x: int):
        """Create a path from an unreachable cell to the visited area."""
        # Simple approach: carve towards center
        y, x = start_y, start_x
        center_y, center_x = MAZE_HEIGHT // 2, MAZE_WIDTH // 2
        
        max_steps = 50
        for _ in range(max_steps):
            if visited[y, x]:
                break
            
            # Move towards a visited cell
            dy = 1 if y < center_y else -1 if y > center_y else 0
            dx = 1 if x < center_x else -1 if x > center_x else 0
            
            # Randomly choose direction
            if random.random() < 0.5 and dy != 0:
                y += dy
            elif dx != 0:
                x += dx
            else:
                y += dy
            
            # Clamp to bounds
            y = max(1, min(MAZE_HEIGHT - 2, y))
            x = max(1, min(MAZE_WIDTH - 2, x))
            
            # Carve path
            if maze[y, x] == WALL:
                maze[y, x] = DOT
                visited[y, x] = True
    
    def save(self, path: str):
        """Save GAN models."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }, path)
        print(f"Saved GAN models to {path}")
    
    def load(self, path: str):
        """Load GAN models."""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            print(f"Loaded GAN models from {path}")
            return True
        return False


class HybridMazeGenerator:
    """
    Hybrid approach: Uses GAN for structure + rule-based post-processing.
    Can also generate mazes purely procedurally without trained GAN.
    """
    
    def __init__(self, use_gan: bool = True, model_path: Optional[str] = None):
        self.use_gan = use_gan
        self.gan = None
        
        if use_gan:
            try:
                self.gan = MazeGAN()
                if model_path and Path(model_path).exists():
                    self.gan.load(model_path)
            except Exception as e:
                print(f"Could not initialize GAN: {e}")
                self.use_gan = False
    
    def generate(self) -> np.ndarray:
        """Generate a new playable maze."""
        if self.use_gan and self.gan:
            return self.gan.generate_maze(validate=True)
        else:
            return self._generate_procedural()
    
    def _generate_procedural(self) -> np.ndarray:
        """Generate maze using procedural algorithms (cellular automata + rules)."""
        maze = np.ones((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.int32)
        
        # Initialize with random noise
        for y in range(1, MAZE_HEIGHT - 1):
            for x in range(1, MAZE_WIDTH - 1):
                if random.random() < 0.55:
                    maze[y, x] = DOT
        
        # Apply cellular automata smoothing
        for _ in range(4):
            new_maze = maze.copy()
            for y in range(1, MAZE_HEIGHT - 1):
                for x in range(1, MAZE_WIDTH - 1):
                    # Count wall neighbors
                    walls = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if maze[y + dy, x + dx] == WALL:
                                walls += 1
                    
                    # Apply rules
                    if walls >= 5:
                        new_maze[y, x] = WALL
                    elif walls <= 2:
                        new_maze[y, x] = DOT
            maze = new_maze
        
        # Create symmetric maze (mirror left to right)
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH // 2):
                maze[y, MAZE_WIDTH - 1 - x] = maze[y, x]
        
        # Apply structural fixes
        maze = self._apply_structure(maze)
        
        return maze
    
    def _apply_structure(self, maze: np.ndarray) -> np.ndarray:
        """Apply structural rules to make maze playable."""
        # Border walls
        maze[0, :] = WALL
        maze[-1, :] = WALL
        maze[:, 0] = WALL
        maze[:, -1] = WALL
        
        # Ghost house
        center_y, center_x = MAZE_HEIGHT // 2, MAZE_WIDTH // 2
        gh_y, gh_x = center_y - 2, center_x - 4
        
        for y in range(gh_y, gh_y + 5):
            for x in range(gh_x, gh_x + 9):
                if 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH:
                    if y == gh_y or y == gh_y + 4 or x == gh_x or x == gh_x + 8:
                        maze[y, x] = WALL
                    else:
                        maze[y, x] = EMPTY
        
        # Ghost house entrance
        maze[gh_y, center_x - 1:center_x + 2] = EMPTY
        
        # Side tunnels
        tunnel_y = MAZE_HEIGHT // 2
        maze[tunnel_y, 0:4] = EMPTY
        maze[tunnel_y, -4:] = EMPTY
        
        # Pacman start area
        pac_y, pac_x = MAZE_HEIGHT - 8, MAZE_WIDTH // 2
        for dy in range(-1, 2):
            for dx in range(-3, 4):
                y, x = pac_y + dy, pac_x + dx
                if 0 < y < MAZE_HEIGHT - 1 and 0 < x < MAZE_WIDTH - 1:
                    maze[y, x] = DOT
        
        # Power pellets in corners
        corners = [
            (3, 2), (3, MAZE_WIDTH - 3),
            (MAZE_HEIGHT - 4, 2), (MAZE_HEIGHT - 4, MAZE_WIDTH - 3)
        ]
        for y, x in corners:
            maze[y, x] = POWER_PELLET
            # Ensure accessible
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 < ny < MAZE_HEIGHT - 1 and 0 < nx < MAZE_WIDTH - 1:
                    if maze[ny, nx] == WALL:
                        maze[ny, nx] = DOT
        
        # Ensure connectivity
        self._ensure_connected(maze)
        
        return maze
    
    def _ensure_connected(self, maze: np.ndarray):
        """Ensure all dots are reachable."""
        # Flood fill from pacman start
        start_y, start_x = MAZE_HEIGHT - 8, MAZE_WIDTH // 2
        visited = np.zeros_like(maze, dtype=bool)
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack.pop()
            if not (0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH):
                continue
            if visited[y, x] or maze[y, x] == WALL:
                continue
            
            visited[y, x] = True
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        
        # Connect unreachable areas
        for y in range(1, MAZE_HEIGHT - 1):
            for x in range(1, MAZE_WIDTH - 1):
                if maze[y, x] != WALL and not visited[y, x]:
                    # Carve path to nearest visited
                    cy, cx = y, x
                    for _ in range(20):
                        if visited[cy, cx]:
                            break
                        
                        # Move towards center
                        if cy < MAZE_HEIGHT // 2:
                            cy += 1
                        elif cy > MAZE_HEIGHT // 2:
                            cy -= 1
                        elif cx < MAZE_WIDTH // 2:
                            cx += 1
                        else:
                            cx -= 1
                        
                        cy = max(1, min(MAZE_HEIGHT - 2, cy))
                        cx = max(1, min(MAZE_WIDTH - 2, cx))
                        
                        if maze[cy, cx] == WALL:
                            maze[cy, cx] = DOT


# Convenience function
def generate_new_maze(use_gan: bool = False) -> np.ndarray:
    """Generate a new playable maze."""
    generator = HybridMazeGenerator(use_gan=use_gan)
    return generator.generate()


if __name__ == "__main__":
    # Test maze generation
    print("Testing Procedural Maze Generator...")
    
    maze = generate_new_maze(use_gan=False)
    
    # Print maze
    symbols = {0: ' ', 1: '█', 2: '·', 3: '●'}
    print("\nGenerated Maze:")
    for row in maze:
        print(''.join(symbols.get(cell, '?') for cell in row))
    
    # Count elements
    print(f"\nMaze Statistics:")
    print(f"  Walls: {np.sum(maze == WALL)}")
    print(f"  Dots: {np.sum(maze == DOT)}")
    print(f"  Power Pellets: {np.sum(maze == POWER_PELLET)}")
    print(f"  Empty: {np.sum(maze == EMPTY)}")
