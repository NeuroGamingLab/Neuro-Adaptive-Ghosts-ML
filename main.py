#!/usr/bin/env python3
"""
Pacman Game with ML-Powered Agents
Main entry point for running the game with trained or untrained agents
"""

import argparse
import os
import sys
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Warning: pygame not installed. Install with: pip install pygame")

# Maze generator
try:
    from ml.maze_gan import generate_new_maze, HybridMazeGenerator
    HAS_MAZE_GAN = True
except ImportError:
    HAS_MAZE_GAN = False
    print("Note: Maze GAN not available. Using default maze.")

# Neural Pathfinding
try:
    from ml.neural_pathfinder import HybridPathfinder
    HAS_NEURAL_PATH = True
except ImportError:
    HAS_NEURAL_PATH = False
    print("Note: Neural pathfinding not available.")

# Ghost Evolution
try:
    from ml.ghost_evolution import GhostEvolution, EvolvingGhostBehavior
    HAS_EVOLUTION = True
except ImportError:
    HAS_EVOLUTION = False
    print("Note: Ghost evolution not available.")


class SoundManager:
    """Manages game sound effects using programmatically generated sounds."""
    
    def __init__(self):
        self.enabled = True
        self.sounds = {}
        
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.sample_rate = 44100  # Store for sound generation
            print(f"Mixer settings: {pygame.mixer.get_init()}")
            self._generate_sounds()
            print(f"Sound system initialized! Sounds: {list(self.sounds.keys())}")
        except Exception as e:
            print(f"Sound init failed: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False
    
    def _generate_tone(self, frequency, duration, volume=0.3, wave_type='square'):
        """Generate a tone as a pygame Sound object."""
        sample_rate = self.sample_rate  # Use mixer's sample rate
        n_samples = int(sample_rate * duration)
        
        # Generate time array
        t = np.linspace(0, duration, n_samples, False)
        
        # Generate waveform
        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        else:
            wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply volume and convert to 16-bit
        wave = (wave * volume * 32767).astype(np.int16)
        
        # Make stereo
        stereo_wave = np.column_stack((wave, wave))
        
        # Create pygame sound
        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound
    
    def _generate_sounds(self):
        """Generate all game sounds - LOUD volumes."""
        # Chomp sound (eating dots) - quick blip
        self.sounds['chomp'] = self._generate_tone(600, 0.05, 0.5, 'square')
        
        # Power pellet - rising tone (LOUD)
        self.sounds['power'] = self._combine_tones([
            (400, 0.1), (500, 0.1), (600, 0.1), (800, 0.15)
        ], 0.7)
        
        # Fart sound - EXPLOSIVE!
        self.sounds['fart'] = self._generate_fart_sound()
        
        # Teleport - sci-fi whoosh
        self.sounds['teleport'] = self._combine_tones([
            (1000, 0.05), (800, 0.05), (600, 0.05), (400, 0.05), (300, 0.1)
        ], 0.6)
        
        # Speed boost - acceleration sound
        self.sounds['speed'] = self._combine_tones([
            (300, 0.08), (400, 0.08), (500, 0.08), (700, 0.1)
        ], 0.6)
        
        # Ghost eaten - victory blip
        self.sounds['ghost_eaten'] = self._combine_tones([
            (800, 0.1), (1000, 0.1), (1200, 0.15)
        ], 0.6)
        
        # Death sound - descending
        self.sounds['death'] = self._combine_tones([
            (500, 0.15), (400, 0.15), (300, 0.15), (200, 0.2)
        ], 0.7)
        
        # Game over - sad tones
        self.sounds['game_over'] = self._combine_tones([
            (400, 0.2), (350, 0.2), (300, 0.3)
        ], 0.7)
        
        # Win sound - happy fanfare
        self.sounds['win'] = self._combine_tones([
            (523, 0.15), (659, 0.15), (784, 0.15), (1047, 0.3)
        ], 0.7)
        
        # Alpha hunt activated
        self.sounds['alpha'] = self._combine_tones([
            (200, 0.1), (250, 0.1), (200, 0.1)
        ], 0.6)
    
    def _combine_tones(self, tone_specs, volume):
        """Combine multiple tones into one sound."""
        sample_rate = self.sample_rate  # Use mixer's sample rate
        all_samples = []
        
        for freq, dur in tone_specs:
            n_samples = int(sample_rate * dur)
            t = np.linspace(0, dur, n_samples, False)
            wave = np.sign(np.sin(2 * np.pi * freq * t))
            all_samples.extend(wave)
        
        wave = np.array(all_samples)
        wave = (wave * volume * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _generate_fart_sound(self):
        """Generate an EXPLOSIVE fart sound effect - SIMPLE BUT LOUD!"""
        sample_rate = self.sample_rate  # Use mixer's sample rate
        duration = 0.8
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        
        # Simple but effective: Low boom + noise
        # Bass explosion
        boom = np.sin(2 * np.pi * 80 * t)
        
        # Wobble for fart effect
        wobble = np.sin(2 * np.pi * 10 * t)
        fart = np.sin(2 * np.pi * (60 + 30 * wobble) * t)
        
        # Noise burst
        noise = np.random.uniform(-0.5, 0.5, n_samples)
        
        # Combine with envelope
        envelope = np.exp(-t * 4)
        wave = (boom * 0.5 + fart * 0.3 + noise * 0.2) * envelope
        
        # Normalize to max volume
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        # Convert to 16-bit stereo at FULL volume
        wave = (wave * 32000).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        print(f"Fart sound created: {sound.get_length()}s")
        return sound
    
    def play(self, sound_name):
        """Play a sound by name at full volume."""
        if self.enabled and sound_name in self.sounds:
            self.sounds[sound_name].set_volume(1.0)  # Max volume
            channel = self.sounds[sound_name].play()
            if sound_name == 'fart':
                print(f">>> FART SOUND TRIGGERED! Channel: {channel}")
        elif sound_name == 'fart':
            print(f">>> FART NOT PLAYED - enabled:{self.enabled}, in_sounds:{sound_name in self.sounds}")
    
    def toggle(self):
        """Toggle sound on/off."""
        self.enabled = not self.enabled
        return self.enabled


# Game Constants
CELL_SIZE = 20
GRID_WIDTH = 40
GRID_HEIGHT = 31

# UI Layout
TITLE_BAR_HEIGHT = 40
BOTTOM_BAR_HEIGHT = 35
SIDE_PANEL_WIDTH = 140

# Window dimensions (game + UI borders)
GAME_WIDTH = GRID_WIDTH * CELL_SIZE  # 800
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE  # 620
WINDOW_WIDTH = GAME_WIDTH + SIDE_PANEL_WIDTH  # 940
WINDOW_HEIGHT = GAME_HEIGHT + TITLE_BAR_HEIGHT + BOTTOM_BAR_HEIGHT  # 695

# Offsets for game area
GAME_OFFSET_X = 0
GAME_OFFSET_Y = TITLE_BAR_HEIGHT

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
GHOST_COLORS = [RED, PINK, CYAN, ORANGE]

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
STOP = (0, 0)

ACTION_TO_DIR = {
    0: UP,
    1: DOWN,
    2: LEFT,
    3: RIGHT
}


def wrap_position(x, y):
    """Wrap (x, y) to torus: both horizontal and vertical wrap (360° play area)."""
    wx = (x % GRID_WIDTH + GRID_WIDTH) % GRID_WIDTH
    wy = (y % GRID_HEIGHT + GRID_HEIGHT) % GRID_HEIGHT
    return wx, wy


def torus_manhattan(ax, ay, bx, by):
    """Manhattan distance on a torus (shortest path with wrap)."""
    dx = (ax - bx + GRID_WIDTH) % GRID_WIDTH
    if dx > GRID_WIDTH // 2:
        dx -= GRID_WIDTH
    dy = (ay - by + GRID_HEIGHT) % GRID_HEIGHT
    if dy > GRID_HEIGHT // 2:
        dy -= GRID_HEIGHT
    return abs(dx) + abs(dy)


class Maze:
    """Game maze with walls, dots, and power pellets."""
    
    def __init__(self):
        self.grid = np.array([
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
        self._ensure_vertical_tunnel()
        self.original_grid = self.grid.copy()
        self.dots_count = np.sum((self.grid == 2) | (self.grid == 3))
    
    def _ensure_vertical_tunnel(self):
        """Open entire top and bottom rows so player can move North/South (torus wrap) from any column."""
        for col in range(GRID_WIDTH):
            self.grid[0, col] = 0
            self.grid[GRID_HEIGHT - 1, col] = 0
    
    def reset(self):
        self.grid = self.original_grid.copy()
        self.dots_count = np.sum((self.grid == 2) | (self.grid == 3))
    
    def load_grid(self, new_grid: np.ndarray):
        """Load a new maze grid (for procedural generation)."""
        self.grid = new_grid.copy()
        self._ensure_vertical_tunnel()
        self.original_grid = self.grid.copy()
        self.dots_count = np.sum((self.grid == 2) | (self.grid == 3))
    
    def is_wall(self, x, y):
        wx, wy = wrap_position(x, y)
        return self.grid[wy, wx] == 1
    
    def get_cell(self, x, y):
        wx, wy = wrap_position(x, y)
        return self.grid[wy, wx]
    
    def set_cell(self, x, y, value):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            self.grid[y, x] = value
    
    def draw(self, screen):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = self.grid[y, x]
                rect = pygame.Rect(
                    GAME_OFFSET_X + x * CELL_SIZE, 
                    GAME_OFFSET_Y + y * CELL_SIZE, 
                    CELL_SIZE, CELL_SIZE
                )
                
                if cell == 1:
                    pygame.draw.rect(screen, BLUE, rect)
                elif cell == 2:
                    center = (GAME_OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2, 
                             GAME_OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(screen, YELLOW, center, 2)
                elif cell == 3:
                    center = (GAME_OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2, 
                             GAME_OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(screen, YELLOW, center, 6)


class Pacman:
    """Pacman player character with teleport and super speed abilities."""
    
    # Teleport constants
    MAX_TELEPORT_CHARGES = 3
    TELEPORT_COOLDOWN = 30  # frames (~3 seconds at 10 FPS)
    
    # Super Speed constants
    MAX_SPEED_CHARGES = 3
    SPEED_DURATION = 50  # frames (~5 seconds at 10 FPS)
    SPEED_COOLDOWN = 20  # frames (~2 seconds at 10 FPS)
    SPEED_MULTIPLIER = 2  # 2x speed
    
    def __init__(self, x=20, y=23):
        self.start_x = x
        self.start_y = y
        self.reset()
    
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.direction = STOP
        self.next_direction = STOP
        self.mouth_angle = 0
        # Teleport ability
        self.teleport_charges = self.MAX_TELEPORT_CHARGES
        self.teleport_cooldown = 0
        self.teleport_effect = 0  # Visual effect timer
        self.last_teleport_pos = None  # For visual trail
        # Super Speed ability
        self.speed_charges = self.MAX_SPEED_CHARGES
        self.speed_active = 0  # Remaining frames of speed boost
        self.speed_cooldown = 0
        self.speed_trail = []  # Trail positions for visual effect
    
    def can_teleport(self):
        """Check if Pacman can teleport."""
        return self.teleport_charges > 0 and self.teleport_cooldown <= 0
    
    def can_speed_boost(self):
        """Check if Pacman can activate speed boost."""
        return self.speed_charges > 0 and self.speed_cooldown <= 0 and self.speed_active <= 0
    
    def is_speed_active(self):
        """Check if speed boost is currently active."""
        return self.speed_active > 0
    
    def activate_speed(self):
        """Activate super speed."""
        if not self.can_speed_boost():
            return False
        
        self.speed_charges -= 1
        self.speed_active = self.SPEED_DURATION
        self.speed_trail = []
        return True
    
    def find_safe_teleport_location(self, maze, ghost_positions):
        """Find a safe location far from all ghosts."""
        best_pos = None
        best_min_dist = -1
        
        # Calculate center of ghost pack
        if ghost_positions:
            ghost_center_x = sum(g[0] for g in ghost_positions) / len(ghost_positions)
            ghost_center_y = sum(g[1] for g in ghost_positions) / len(ghost_positions)
        else:
            ghost_center_x, ghost_center_y = self.x, self.y
        
        # Search all valid positions
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Must be a valid position (not wall)
                if maze.is_wall(x, y):
                    continue
                
                # Skip current position and nearby positions (torus distance)
                if torus_manhattan(x, y, self.x, self.y) < 5:
                    continue
                
                # Calculate minimum distance to any ghost (torus)
                min_ghost_dist = float('inf')
                for gx, gy in ghost_positions:
                    dist = torus_manhattan(x, y, gx, gy)
                    min_ghost_dist = min(min_ghost_dist, dist)
                
                # Also consider distance from ghost pack center (torus)
                pack_dist = torus_manhattan(x, y, int(ghost_center_x), int(ghost_center_y))
                
                # Combined score: farther from nearest ghost + farther from pack center
                score = min_ghost_dist + pack_dist * 0.5
                
                if score > best_min_dist:
                    best_min_dist = score
                    best_pos = (x, y)
        
        return best_pos
    
    def teleport(self, maze, ghost_positions):
        """Execute teleport to a safe location."""
        if not self.can_teleport():
            return False
        
        safe_pos = self.find_safe_teleport_location(maze, ghost_positions)
        
        if safe_pos:
            self.last_teleport_pos = (self.x, self.y)
            self.x, self.y = safe_pos
            self.teleport_charges -= 1
            self.teleport_cooldown = self.TELEPORT_COOLDOWN
            self.teleport_effect = 15  # Visual effect frames
            self.direction = STOP
            return True
        return False
    
    def update(self, maze):
        # Update teleport cooldown
        if self.teleport_cooldown > 0:
            self.teleport_cooldown -= 1
        
        # Update teleport effect
        if self.teleport_effect > 0:
            self.teleport_effect -= 1
        
        # Update speed boost timers
        if self.speed_active > 0:
            self.speed_active -= 1
            if self.speed_active <= 0:
                self.speed_cooldown = self.SPEED_COOLDOWN
        
        if self.speed_cooldown > 0:
            self.speed_cooldown -= 1
        
        # Try to change direction
        if self.next_direction != STOP:
            nx = self.x + self.next_direction[0]
            ny = self.y + self.next_direction[1]
            if not maze.is_wall(nx, ny):
                self.direction = self.next_direction
                self.next_direction = STOP
        
        # Determine move count (2x when speed active)
        moves = self.SPEED_MULTIPLIER if self.is_speed_active() else 1
        
        # Move (multiple times if speed boost active)
        for _ in range(moves):
            if self.direction != STOP:
                # Store trail position for speed effect
                if self.is_speed_active():
                    self.speed_trail.append((self.x, self.y))
                    if len(self.speed_trail) > 8:
                        self.speed_trail.pop(0)
                
                nx = self.x + self.direction[0]
                ny = self.y + self.direction[1]
                # Torus wrap (360°): both horizontal and vertical
                nx, ny = wrap_position(nx, ny)
                
                if not maze.is_wall(nx, ny):
                    self.x = nx
                    self.y = ny
                else:
                    break  # Stop if we hit a wall
        
        # Clear trail when speed ends
        if not self.is_speed_active() and self.speed_trail:
            self.speed_trail = []
        
        self.mouth_angle = (self.mouth_angle + 0.5 if self.is_speed_active() else 0.3) % (2 * 3.14159)
    
    def draw(self, screen):
        cx = GAME_OFFSET_X + self.x * CELL_SIZE + CELL_SIZE // 2
        cy = GAME_OFFSET_Y + self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        # Draw speed trail effect (orange/red motion blur)
        if self.speed_trail:
            for i, (tx, ty) in enumerate(self.speed_trail):
                trail_cx = GAME_OFFSET_X + tx * CELL_SIZE + CELL_SIZE // 2
                trail_cy = GAME_OFFSET_Y + ty * CELL_SIZE + CELL_SIZE // 2
                alpha = int(150 * (i + 1) / len(self.speed_trail))
                trail_radius = int(radius * (0.4 + 0.4 * (i + 1) / len(self.speed_trail)))
                
                trail_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, (255, 165, 0, alpha), 
                                 (CELL_SIZE // 2, CELL_SIZE // 2), trail_radius)
                screen.blit(trail_surface, (trail_cx - CELL_SIZE // 2, trail_cy - CELL_SIZE // 2))
        
        # Draw teleport trail effect
        if self.teleport_effect > 0 and self.last_teleport_pos:
            old_cx = GAME_OFFSET_X + self.last_teleport_pos[0] * CELL_SIZE + CELL_SIZE // 2
            old_cy = GAME_OFFSET_Y + self.last_teleport_pos[1] * CELL_SIZE + CELL_SIZE // 2
            alpha = int(255 * self.teleport_effect / 15)
            
            # Draw fading ghost at old position
            trail_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (255, 255, 0, alpha // 2), (CELL_SIZE // 2, CELL_SIZE // 2), radius)
            screen.blit(trail_surface, (old_cx - CELL_SIZE // 2, old_cy - CELL_SIZE // 2))
            
            # Draw teleport line
            pygame.draw.line(screen, (0, 255, 255, alpha), (old_cx, old_cy), (cx, cy), 2)
        
        # Teleport glow effect at destination
        if self.teleport_effect > 0:
            glow_radius = radius + 5 + int(5 * np.sin(self.teleport_effect * 0.5))
            glow_surface = pygame.Surface((glow_radius * 2 + 10, glow_radius * 2 + 10), pygame.SRCALPHA)
            glow_alpha = int(150 * self.teleport_effect / 15)
            pygame.draw.circle(glow_surface, (0, 255, 255, glow_alpha), 
                             (glow_radius + 5, glow_radius + 5), glow_radius)
            screen.blit(glow_surface, (cx - glow_radius - 5, cy - glow_radius - 5))
        
        # Speed boost glow effect (orange pulsing glow)
        if self.is_speed_active():
            glow_radius = radius + 3 + int(3 * np.sin(self.speed_active * 0.3))
            glow_surface = pygame.Surface((glow_radius * 2 + 10, glow_radius * 2 + 10), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (255, 165, 0, 100), 
                             (glow_radius + 5, glow_radius + 5), glow_radius)
            screen.blit(glow_surface, (cx - glow_radius - 5, cy - glow_radius - 5))
        
        # Draw Pacman (orange tint when speed active)
        pacman_color = (255, 200, 0) if self.is_speed_active() else YELLOW
        pygame.draw.circle(screen, pacman_color, (cx, cy), radius)
        
        # Draw mouth
        mouth_open = abs(np.sin(self.mouth_angle)) > 0.5
        if mouth_open and self.direction != STOP:
            if self.direction == RIGHT:
                points = [(cx, cy), (cx + radius, cy - radius//2), (cx + radius, cy + radius//2)]
            elif self.direction == LEFT:
                points = [(cx, cy), (cx - radius, cy - radius//2), (cx - radius, cy + radius//2)]
            elif self.direction == UP:
                points = [(cx, cy), (cx - radius//2, cy - radius), (cx + radius//2, cy - radius)]
            else:
                points = [(cx, cy), (cx - radius//2, cy + radius), (cx + radius//2, cy + radius)]
            pygame.draw.polygon(screen, BLACK, points)


class Ghost:
    """Ghost enemy with optional ML-powered behavior and alpha pack status."""
    
    # Role names for display
    ROLE_NAMES = {
        'chaser': 'CHASER',
        'ambusher': 'AMBUSH',
        'blocker': 'BLOCKER',
        'patrol': 'PATROL'
    }
    
    def __init__(self, x, y, color, ghost_id=0, role='chaser'):
        self.start_x = x
        self.start_y = y
        self.color = color
        self.ghost_id = ghost_id
        self.role = role
        self.target = None  # Coordinated target position
        self.is_alpha = False  # Is this ghost the pack leader?
        self.reset()
    
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.direction = UP
        self.frightened = False
        self.frightened_timer = 0
        self.target = None
        self.is_alpha = False
    
    def get_valid_moves(self, maze):
        valid = []
        for action, (dx, dy) in ACTION_TO_DIR.items():
            nx, ny = wrap_position(self.x + dx, self.y + dy)
            if not maze.is_wall(nx, ny):
                valid.append(action)
        return valid
    
    def update(self, maze, pacman, ml_action=None):
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False
        
        valid_moves = self.get_valid_moves(maze)
        
        if not valid_moves:
            return
        
        if ml_action is not None and ml_action in valid_moves:
            # Use ML/coordinated action
            action = ml_action
        elif self.frightened:
            # Run away from Pacman (torus distance)
            best_dist = -1
            action = valid_moves[0]
            for a in valid_moves:
                dx, dy = ACTION_TO_DIR[a]
                nx, ny = wrap_position(self.x + dx, self.y + dy)
                dist = torus_manhattan(nx, ny, pacman.x, pacman.y)
                if dist > best_dist:
                    best_dist = dist
                    action = a
        else:
            # Chase Pacman with some randomness (torus distance)
            if np.random.random() < 0.2:
                action = np.random.choice(valid_moves)
            else:
                best_dist = float('inf')
                action = valid_moves[0]
                for a in valid_moves:
                    dx, dy = ACTION_TO_DIR[a]
                    nx, ny = wrap_position(self.x + dx, self.y + dy)
                    dist = torus_manhattan(nx, ny, pacman.x, pacman.y)
                    if dist < best_dist:
                        best_dist = dist
                        action = a
        
        # Move (torus wrap: 360° horizontal and vertical)
        dx, dy = ACTION_TO_DIR[action]
        nx, ny = wrap_position(self.x + dx, self.y + dy)
        
        if not maze.is_wall(nx, ny):
            self.x = nx
            self.y = ny
            self.direction = ACTION_TO_DIR[action]
    
    def set_frightened(self, duration=30):
        self.frightened = True
        self.frightened_timer = duration
    
    def draw(self, screen, pacman=None):
        cx = GAME_OFFSET_X + self.x * CELL_SIZE + CELL_SIZE // 2
        cy = GAME_OFFSET_Y + self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        color = BLUE if self.frightened else self.color
        
        # Hunting line: ghost → target (coordinated) or ghost → Pacman
        if not self.frightened:
            tx, ty = None, None
            if self.target is not None:
                tx, ty = self.target[0], self.target[1]
            elif pacman is not None:
                tx, ty = pacman.x, pacman.y
            if tx is not None and ty is not None:
                end_x = GAME_OFFSET_X + tx * CELL_SIZE + CELL_SIZE // 2
                end_y = GAME_OFFSET_Y + ty * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.line(screen, color, (cx, cy), (end_x, end_y), 2)
        
        # ALPHA GHOST: Draw multi-color blinking circle!
        if self.is_alpha and not self.frightened:
            # Multi-color cycling effect
            time_ms = pygame.time.get_ticks()
            color_cycle = (time_ms // 100) % 6  # Cycle through 6 colors
            alpha_colors = [
                (255, 0, 0),      # Red
                (255, 165, 0),    # Orange
                (255, 255, 0),    # Yellow
                (0, 255, 0),      # Green
                (0, 255, 255),    # Cyan
                (255, 0, 255),    # Magenta
            ]
            ring_color = alpha_colors[color_cycle]
            
            # Pulsing ring radius
            pulse = int(3 * np.sin(time_ms * 0.015))
            ring_radius = radius + 6 + pulse
            
            # Draw outer ring (thicker, colored)
            pygame.draw.circle(screen, ring_color, (cx, cy), ring_radius, 3)
            
            # Draw inner glow
            glow_surface = pygame.Surface((ring_radius * 2 + 10, ring_radius * 2 + 10), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*ring_color, 60), 
                             (ring_radius + 5, ring_radius + 5), ring_radius)
            screen.blit(glow_surface, (cx - ring_radius - 5, cy - ring_radius - 5))
        
        # Body
        pygame.draw.circle(screen, color, (cx, cy - 2), radius)
        pygame.draw.rect(screen, color, (cx - radius, cy - 2, radius * 2, radius + 2))
        
        # Eyes
        pygame.draw.circle(screen, WHITE, (cx - 3, cy - 4), 3)
        pygame.draw.circle(screen, WHITE, (cx + 3, cy - 4), 3)
        pygame.draw.circle(screen, BLACK, (cx - 3, cy - 4), 1)
        pygame.draw.circle(screen, BLACK, (cx + 3, cy - 4), 1)
        
        
        # (Hunting line drawn above, before alpha ring)


class Game:
    """Main game class with ML agent support and coordinated ghost attacks."""
    
    def __init__(self, use_ml_ghosts=False, ghost_model_path=None, coordinated=True, sphere_view=False):
        if not HAS_PYGAME:
            raise ImportError("pygame is required. Install with: pip install pygame")
        
        pygame.init()
        self.use_sphere_view = False
        self.sphere_view = None
        if sphere_view:
            try:
                from sphere_view import SphereView, has_opengl
                if not has_opengl():
                    print("Warning: PyOpenGL not available. Install with: pip install PyOpenGL PyOpenGL-accelerate")
                else:
                    self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
                    self.sphere_view = SphereView(WINDOW_WIDTH, WINDOW_HEIGHT)
                    self.sphere_view.init_gl()
                    self.use_sphere_view = True
                    print("Sphere view: ON (play area on 3D sphere, camera follows you)")
            except Exception as e:
                print(f"Sphere view failed: {e}. Falling back to 2D.")
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        if not self.use_sphere_view:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("OBSERVE: Agentic Ghosts Intelligence")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)
        
        # Sound system
        self.sound = SoundManager()
        
        self.maze = Maze()
        self.pacman = Pacman()
        
        # Create ghosts with roles
        self.ghosts = [
            Ghost(18, 14, RED, 0, role='chaser'),      # Blinky - direct chase
            Ghost(20, 14, PINK, 1, role='ambusher'),   # Pinky - intercept
            Ghost(22, 14, CYAN, 2, role='blocker'),    # Inky - block escape
            Ghost(20, 12, ORANGE, 3, role='patrol')    # Clyde - patrol/flank
        ]
        
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.won = False
        
        # Power Fart shockwave effect
        self.fart_effect = 0  # Effect timer
        self.fart_center = None  # Center of shockwave
        self.fart_radius = GRID_WIDTH // 2  # Kill radius = half the board width
        self.fart_kills = 0  # Ghosts killed by fart
        
        # Alpha hunt tracking for sound
        self.was_alpha_active = False
        
        # Coordination system
        self.coordinated = coordinated
        self.coordination_system = None
        
        # Neural pathfinding
        self.neural_pathfinder = None
        if HAS_NEURAL_PATH:
            self.neural_pathfinder = HybridPathfinder()
            print("Neural Pathfinding enabled!")
        
        # Ghost evolution system
        self.evolution = None
        self.ghost_behaviors = []
        if HAS_EVOLUTION:
            self.evolution = GhostEvolution(save_path="ghost_evolution.json")
            self._apply_evolved_behaviors()
            print(f"Ghost Evolution enabled! {self.evolution.get_stats()}")
        
        # ML agents
        self.use_ml_ghosts = use_ml_ghosts
        self.ghost_agents = None
        
        if coordinated:
            self._setup_coordination()
        
        if use_ml_ghosts and ghost_model_path:
            self._load_ml_agents(ghost_model_path)
    
    def _setup_coordination(self):
        """Setup the coordinated ghost attack system."""
        try:
            from agents.multi_agent import CoordinatedGhostSystem
            self.coordination_system = CoordinatedGhostSystem(
                num_ghosts=4,
                coordination_level=0.85  # 85% coordination
            )
            print("Coordinated ghost attacks enabled!")
            print("  - Blinky (Red): CHASER - Direct pursuit")
            print("  - Pinky (Pink): AMBUSHER - Predicts & intercepts")
            print("  - Inky (Cyan): BLOCKER - Blocks escape routes")
            print("  - Clyde (Orange): PATROL - Flanking maneuvers")
        except ImportError as e:
            print(f"Could not setup coordination: {e}")
            self.coordinated = False
    
    def _apply_evolved_behaviors(self):
        """Apply evolved genomes to ghost behaviors."""
        if not self.evolution:
            return
        
        genomes = self.evolution.get_active_genomes()
        self.ghost_behaviors = []
        
        for i, genome in enumerate(genomes):
            behavior = EvolvingGhostBehavior(genome)
            self.ghost_behaviors.append(behavior)
            
            # Apply genome traits to ghost
            if i < len(self.ghosts):
                ghost = self.ghosts[i]
                # Store evolved traits on ghost for use in movement
                ghost.evolved_aggression = genome.aggression
                ghost.evolved_prediction = genome.prediction
                ghost.evolved_ambush = genome.ambush
                ghost.evolved_randomness = genome.randomness
        
        print(f"Evolved behaviors applied to {len(self.ghost_behaviors)} ghosts")
    
    def _load_ml_agents(self, model_path):
        """Load trained ML agents for ghosts."""
        try:
            from agents.multi_agent import MultiAgentGhostSystem
            from agents.ghost_agent import GhostAgent
            
            self.ghost_agents = MultiAgentGhostSystem(num_ghosts=4)
            
            # Try to load trained model for first ghost
            if os.path.exists(model_path):
                self.ghost_agents.load_agent(0, model_path, algorithm='ppo')
                print(f"Loaded ML ghost agent from: {model_path}")
            else:
                print(f"Model not found: {model_path}")
                print("Using default chase behavior for all ghosts")
                self.use_ml_ghosts = False
        except ImportError as e:
            print(f"Could not load ML agents: {e}")
            self.use_ml_ghosts = False
    
    def _get_observation(self, ghost_idx):
        """Get observation for ghost agent."""
        # Flatten maze + positions
        maze_flat = self.maze.grid.flatten().astype(np.float32)
        pacman_pos = np.array([self.pacman.x / GRID_WIDTH, self.pacman.y / GRID_HEIGHT], dtype=np.float32)
        ghost_pos = []
        for g in self.ghosts:
            ghost_pos.extend([g.x / GRID_WIDTH, g.y / GRID_HEIGHT])
        ghost_pos = np.array(ghost_pos, dtype=np.float32)
        ghost_states = np.array([1 if g.frightened else 0 for g in self.ghosts], dtype=np.float32)
        
        return np.concatenate([maze_flat, pacman_pos, ghost_pos, ghost_states])
    
    def reset(self):
        """Reset the game."""
        self.maze.reset()
        self.pacman.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.won = False
        # Reset fart effect
        self.fart_effect = 0
        self.fart_center = None
        self.fart_kills = 0
        # Re-apply evolved behaviors for new game
        if self.evolution:
            self._apply_evolved_behaviors()
    
    def generate_new_maze(self):
        """Generate a new procedural maze using GAN/hybrid generator."""
        if HAS_MAZE_GAN:
            try:
                print("🎲 Generating new maze...")
                new_grid = generate_new_maze(use_gan=False)  # Use procedural for now
                self.maze.load_grid(new_grid)
                
                # Reset game with new maze
                self.pacman.reset()
                for ghost in self.ghosts:
                    ghost.reset()
                self.score = 0
                self.lives = 3
                self.game_over = False
                self.won = False
                self.fart_effect = 0
                self.fart_center = None
                self.fart_kills = 0
                
                print(f"✓ New maze generated! Dots: {self.maze.dots_count}")
                return True
            except Exception as e:
                print(f"✗ Maze generation failed: {e}")
                return False
        else:
            print("✗ Maze generator not available")
            return False
    
    def handle_input(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.pacman.next_direction = UP
                elif event.key == pygame.K_DOWN:
                    self.pacman.next_direction = DOWN
                elif event.key == pygame.K_LEFT:
                    self.pacman.next_direction = LEFT
                elif event.key == pygame.K_RIGHT:
                    self.pacman.next_direction = RIGHT
                elif event.key == pygame.K_SPACE:
                    # Teleport to escape ghost pack attack!
                    ghost_positions = [(g.x, g.y) for g in self.ghosts]
                    if self.pacman.teleport(self.maze, ghost_positions):
                        self.sound.play('teleport')  # Teleport sound!
                        print("TELEPORTED! Escaped the ghost pack!")
                elif event.key == pygame.K_s or event.key == pygame.K_LSHIFT:
                    # Activate super speed!
                    if self.pacman.activate_speed():
                        self.sound.play('speed')  # Speed boost sound!
                        print("🏃 SUPER SPEED ACTIVATED! 2x faster!")
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_g:
                    # Generate new maze using GAN/procedural generation
                    self.generate_new_maze()
                elif event.key == pygame.K_m:
                    # Toggle sound mute
                    enabled = self.sound.toggle()
                    print(f"Sound {'ON' if enabled else 'OFF'}")
                elif event.key == pygame.K_n:
                    # Toggle neural pathfinding
                    if self.neural_pathfinder:
                        enabled = self.neural_pathfinder.toggle_neural()
                        print(f"Neural Pathfinding {'ON' if enabled else 'OFF'}")
                        print(f"Stats: {self.neural_pathfinder.get_stats()}")
                elif event.key == pygame.K_e:
                    # Force evolution / show stats
                    if self.evolution:
                        print(f"\n=== GHOST EVOLUTION ===")
                        print(f"Stats: {self.evolution.get_stats()}")
                        for i, genome in enumerate(self.evolution.active_genomes):
                            print(f"Ghost {i}: {genome.get_behavior_description()}")
                        print("Press E again to force evolution...")
                        # Track double-press to force evolve
                        if hasattr(self, '_last_e_press') and pygame.time.get_ticks() - self._last_e_press < 1000:
                            print("Forcing evolution!")
                            self.evolution.force_evolve()
                            self._apply_evolved_behaviors()
                        self._last_e_press = pygame.time.get_ticks()
        return True
    
    def update(self):
        """Update game state."""
        if self.game_over:
            return
        
        # Update Pacman
        self.pacman.update(self.maze)
        
        # Check dot collection
        cell = self.maze.get_cell(self.pacman.x, self.pacman.y)
        if cell == 2:
            self.maze.set_cell(self.pacman.x, self.pacman.y, 0)
            self.score += 10
            self.maze.dots_count -= 1
            self.sound.play('chomp')  # Chomp sound!
        elif cell == 3:
            self.maze.set_cell(self.pacman.x, self.pacman.y, 0)
            self.score += 50
            self.maze.dots_count -= 1
            self.sound.play('power')  # Power pellet sound!
            
            # POWER FART ATTACK!
            self.fart_center = (self.pacman.x, self.pacman.y)
            self.fart_effect = 30  # Visual effect duration
            self.fart_kills = 0
            self.sound.play('fart')  # FART SOUND!
            
            # Kill all ghosts within radius (1/2 board width); use torus distance
            for i, ghost in enumerate(self.ghosts):
                dist = torus_manhattan(ghost.x, ghost.y, self.pacman.x, self.pacman.y)
                if dist <= self.fart_radius:
                    # Ghost is in the fart zone - ELIMINATED!
                    ghost.reset()
                    self.score += 400  # Bonus for fart kill
                    self.fart_kills += 1
                    self.sound.play('ghost_eaten')  # Ghost eliminated sound!
                    # Evolution: record ghost death from fart
                    if self.evolution:
                        self.evolution.record_death(i)
                    print(f"FART KILL! Ghost eliminated! +400 pts")
                else:
                    # Ghost survived but is still frightened
                    ghost.set_frightened(30)
            
            if self.fart_kills > 0:
                print(f"POWER FART! {self.fart_kills} ghost(s) eliminated!")
        
        # Get ghost actions (coordinated or individual)
        if self.coordinated and self.coordination_system:
            # Use coordinated attack system
            pacman_pos = (self.pacman.x, self.pacman.y)
            ghost_positions = [(g.x, g.y) for g in self.ghosts]
            frightened = [g.frightened for g in self.ghosts]
            
            # Get coordinated actions (includes alpha detection)
            actions = self.coordination_system.get_coordinated_actions(
                observations=[self._get_observation(i) for i in range(4)],
                pacman_pos=pacman_pos,
                ghost_positions=ghost_positions,
                maze=self.maze.grid,
                frightened=frightened
            )
            
            # Update alpha status for visual display
            alpha_active, alpha_idx = self.coordination_system.get_alpha_status()
            
            # Play sound when alpha hunt activates
            if alpha_active and not self.was_alpha_active:
                self.sound.play('alpha')  # Alpha hunt activated!
            self.was_alpha_active = alpha_active
            
            # Update each ghost with coordinated action and alpha status
            for i, ghost in enumerate(self.ghosts):
                ghost.target = self.coordination_system.ghost_targets[i]
                ghost.is_alpha = (alpha_active and i == alpha_idx)
                ghost.update(self.maze, self.pacman, actions[i])
        
        elif self.use_ml_ghosts and self.ghost_agents:
            # Use ML agents
            for i, ghost in enumerate(self.ghosts):
                obs = self._get_observation(i)
                actions = self.ghost_agents.get_actions(
                    [obs] * 4,
                    (self.pacman.x, self.pacman.y),
                    [(g.x, g.y) for g in self.ghosts],
                    self.maze.grid
                )
                ghost.update(self.maze, self.pacman, actions[i])
        else:
            # Simple AI with Neural Pathfinding
            all_ghost_pos = [(g.x, g.y) for g in self.ghosts]
            for ghost in self.ghosts:
                neural_action = None
                
                # Use neural pathfinding if available
                if self.neural_pathfinder and not ghost.frightened:
                    target_x, target_y = self.pacman.x, self.pacman.y
                    direction = self.neural_pathfinder.get_direction(
                        self.maze, ghost.x, ghost.y, target_x, target_y, all_ghost_pos
                    )
                    if direction:
                        # Convert direction (dx, dy) to action
                        for action, (dx, dy) in ACTION_TO_DIR.items():
                            if (dx, dy) == direction:
                                neural_action = action
                                break
                
                ghost.update(self.maze, self.pacman, neural_action)
        
        # Check collisions
        for i, ghost in enumerate(self.ghosts):
            if ghost.x == self.pacman.x and ghost.y == self.pacman.y:
                if ghost.frightened:
                    ghost.reset()
                    self.score += 200
                    self.sound.play('ghost_eaten')  # Ate ghost sound!
                    # Evolution: record ghost death
                    if self.evolution:
                        self.evolution.record_death(i)
                else:
                    self.lives -= 1
                    self.sound.play('death')  # Death sound!
                    # Evolution: record ghost catch!
                    if self.evolution:
                        self.evolution.record_catch(i)
                    if self.lives <= 0:
                        self.game_over = True
                        self.sound.play('game_over')  # Game over sound!
                        # Evolution: end game (Pacman lost)
                        if self.evolution:
                            self.evolution.end_game(pacman_won=False, final_score=self.score)
                    else:
                        self.pacman.reset()
                        for g in self.ghosts:
                            g.reset()
        
        # Check win
        if self.maze.dots_count == 0:
            self.won = True
            self.game_over = True
            self.sound.play('win')  # Win fanfare!
            # Evolution: end game (Pacman won)
            if self.evolution:
                self.evolution.end_game(pacman_won=True, final_score=self.score)
    
    def draw(self):
        """Draw the game."""
        # 3D sphere view (camera follows player)
        if self.use_sphere_view and self.sphere_view:
            self.sphere_view.draw_frame(self.maze, self.pacman, self.ghosts)
            pygame.display.set_caption(
                f"OBSERVE: Agentic Ghosts | Score: {self.score} | Lives: {self.lives} | [2D: run without --sphere-view]"
            )
            pygame.display.flip()
            return
        
        self.screen.fill(BLACK)
        
        # === TITLE BAR (top) ===
        pygame.draw.rect(self.screen, (20, 20, 40), (0, 0, WINDOW_WIDTH, TITLE_BAR_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 100), (0, TITLE_BAR_HEIGHT - 1), (WINDOW_WIDTH, TITLE_BAR_HEIGHT - 1), 2)
        
        # Title text
        title_font = pygame.font.Font(None, 36)
        title_text = title_font.render("OBSERVE: Agentic Ghosts Intelligence", True, YELLOW)
        self.screen.blit(title_text, (10, 8))
        
        # === SIDE PANEL (right) ===
        panel_x = GAME_WIDTH
        pygame.draw.rect(self.screen, (20, 20, 40), (panel_x, TITLE_BAR_HEIGHT, SIDE_PANEL_WIDTH, GAME_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 100), (panel_x, TITLE_BAR_HEIGHT), (panel_x, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT), 2)
        
        # Controls in side panel
        panel_y = TITLE_BAR_HEIGHT + 10
        controls_title = self.font.render("CONTROLS", True, WHITE)
        self.screen.blit(controls_title, (panel_x + 10, panel_y))
        panel_y += 30
        
        controls = [
            ("Arrows", "Move", (180, 180, 180)),
            ("SPACE", "Teleport", CYAN),
            ("S", "Speed 2x", ORANGE),
            ("G", "New Maze", (255, 100, 255)),
            ("R", "Restart", (180, 180, 180)),
            ("M", "Mute", (150, 150, 150)),
            ("N", "Neural AI", (100, 255, 100)),
            ("E", "Evolution", (255, 200, 100)),
        ]
        for key, action, color in controls:
            key_text = self.small_font.render(key, True, WHITE)
            action_text = self.small_font.render(action, True, color)
            self.screen.blit(key_text, (panel_x + 10, panel_y))
            self.screen.blit(action_text, (panel_x + 60, panel_y))
            panel_y += 18
        
        # Separator
        panel_y += 5
        pygame.draw.line(self.screen, (60, 60, 100), (panel_x + 10, panel_y), (panel_x + SIDE_PANEL_WIDTH - 10, panel_y), 1)
        panel_y += 10
        
        # Power Pill info
        pill_text = self.small_font.render("Power Pill", True, (50, 200, 50))
        fart_text = self.small_font.render("= FART!", True, (50, 200, 50))
        self.screen.blit(pill_text, (panel_x + 10, panel_y))
        panel_y += 16
        self.screen.blit(fart_text, (panel_x + 10, panel_y))
        panel_y += 25
        
        # Ghost roles section
        pygame.draw.line(self.screen, (60, 60, 100), (panel_x + 10, panel_y), (panel_x + SIDE_PANEL_WIDTH - 10, panel_y), 1)
        panel_y += 10
        
        ghosts_title = self.font.render("GHOSTS", True, WHITE)
        self.screen.blit(ghosts_title, (panel_x + 10, panel_y))
        panel_y += 25
        
        # Check alpha status
        alpha_active = False
        alpha_idx = None
        if self.coordination_system:
            alpha_active, alpha_idx = self.coordination_system.get_alpha_status()
        
        # Show ALPHA HUNT mode
        if alpha_active:
            hunt_text = self.small_font.render(">> ALPHA HUNT!", True, (255, 215, 0))
            self.screen.blit(hunt_text, (panel_x + 10, panel_y))
            panel_y += 20
        
        ghost_names = ["BLINKY", "PINKY", "INKY", "CLYDE"]
        ghost_colors = [RED, PINK, CYAN, ORANGE]
        
        for i, (name, color) in enumerate(zip(ghost_names, ghost_colors)):
            if alpha_active and i == alpha_idx:
                role_text = f"* {name}"
                text_color = (255, 215, 0)
            elif alpha_active:
                role_text = f"  {name}"
                text_color = (120, 120, 120)
            else:
                role_text = f"  {name}"
                text_color = color
            
            text = self.small_font.render(role_text, True, text_color)
            self.screen.blit(text, (panel_x + 10, panel_y))
            
            # Show evolved personality if available
            if self.evolution and i < len(self.evolution.active_genomes):
                personality = self.evolution.get_ghost_personality(i)
                if personality and personality != "Balanced":
                    # Truncate if too long
                    if len(personality) > 12:
                        personality = personality[:10] + ".."
                    pers_text = self.small_font.render(personality, True, (150, 150, 150))
                    self.screen.blit(pers_text, (panel_x + 70, panel_y))
            
            panel_y += 16
        
        # === BOTTOM BAR ===
        bottom_y = TITLE_BAR_HEIGHT + GAME_HEIGHT
        pygame.draw.rect(self.screen, (20, 20, 40), (0, bottom_y, WINDOW_WIDTH, BOTTOM_BAR_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 100), (0, bottom_y), (WINDOW_WIDTH, bottom_y), 2)
        
        # Draw the maze
        self.maze.draw(self.screen)
        
        # Draw POWER FART shockwave effect 💨
        if self.fart_effect > 0 and self.fart_center:
            # Calculate expanding ring
            progress = 1.0 - (self.fart_effect / 30.0)
            current_radius = int(self.fart_radius * CELL_SIZE * progress)
            max_radius = self.fart_radius * CELL_SIZE
            
            center_x = GAME_OFFSET_X + self.fart_center[0] * CELL_SIZE + CELL_SIZE // 2
            center_y = GAME_OFFSET_Y + self.fart_center[1] * CELL_SIZE + CELL_SIZE // 2
            
            # Draw multiple expanding rings (green toxic gas effect)
            for ring_offset in range(3):
                ring_radius = current_radius - ring_offset * 15
                if ring_radius > 0:
                    alpha = int(180 * (1.0 - progress) * (1.0 - ring_offset * 0.3))
                    ring_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    
                    # Green toxic gas color
                    gas_color = (50, 200, 50, alpha)
                    pygame.draw.circle(ring_surface, gas_color, (center_x, center_y), ring_radius, 8)
                    
                    # Inner yellowish cloud
                    if ring_offset == 0:
                        inner_color = (150, 200, 50, alpha // 2)
                        pygame.draw.circle(ring_surface, inner_color, (center_x, center_y), ring_radius // 2)
                    
                    self.screen.blit(ring_surface, (0, 0))
            
            # Draw kill zone indicator (dashed circle at max radius)
            if self.fart_effect > 15:
                zone_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                zone_alpha = int(100 * (self.fart_effect - 15) / 15)
                pygame.draw.circle(zone_surface, (255, 50, 50, zone_alpha), 
                                 (center_x, center_y), max_radius, 2)
                self.screen.blit(zone_surface, (0, 0))
            
            # Show kill count
            if self.fart_kills > 0 and self.fart_effect > 10:
                kill_text = self.font.render(f"💨 x{self.fart_kills} ELIMINATED!", True, (50, 255, 50))
                text_rect = kill_text.get_rect(center=(center_x, center_y - 30))
                self.screen.blit(kill_text, text_rect)
            
            self.fart_effect -= 1
        
        if not self.game_over:
            self.pacman.draw(self.screen)
            for ghost in self.ghosts:
                ghost.draw(self.screen, self.pacman)
        
        # UI - Bottom bar (updated Y position)
        bar_y = TITLE_BAR_HEIGHT + GAME_HEIGHT + 5
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.lives}", True, WHITE)
        self.screen.blit(score_text, (10, bar_y))
        self.screen.blit(lives_text, (180, bar_y))
        
        # Speed boost indicator
        if self.pacman.is_speed_active():
            speed_color = ORANGE
            speed_text = f"SPD {self.pacman.speed_active // 10 + 1}s"
        elif self.pacman.can_speed_boost():
            speed_color = ORANGE
            speed_text = f"SPD x{self.pacman.speed_charges}"
        else:
            speed_color = (100, 100, 100)
            speed_text = f"SPD x{self.pacman.speed_charges}"
            if self.pacman.speed_cooldown > 0:
                speed_text += f" ({self.pacman.speed_cooldown // 10 + 1}s)"
        speed_render = self.font.render(speed_text, True, speed_color)
        self.screen.blit(speed_render, (320, bar_y))
        
        # Teleport charges indicator
        teleport_color = CYAN if self.pacman.can_teleport() else (100, 100, 100)
        teleport_text = f"TELE x{self.pacman.teleport_charges}"
        if self.pacman.teleport_cooldown > 0:
            teleport_text += f" ({self.pacman.teleport_cooldown // 10 + 1}s)"
        teleport_render = self.font.render(teleport_text, True, teleport_color)
        self.screen.blit(teleport_render, (450, bar_y))
        
        # Coordination status
        if self.coordinated:
            coord_text = self.small_font.render("COORDINATED ATTACK", True, CYAN)
            self.screen.blit(coord_text, (580, bar_y + 5))
        elif self.use_ml_ghosts:
            ml_text = self.small_font.render("ML Ghosts: ON", True, CYAN)
            self.screen.blit(ml_text, (580, bar_y + 5))
        
        # Neural pathfinding status
        if self.neural_pathfinder:
            np_color = (100, 255, 100) if self.neural_pathfinder.use_neural else (100, 100, 100)
            np_text = f"Neural: {self.neural_pathfinder.get_stats()}"
            np_render = self.small_font.render(np_text, True, np_color)
            self.screen.blit(np_render, (580, bar_y + 20))
        
        # Evolution status
        if self.evolution:
            evo_text = f"Evo: {self.evolution.get_stats()}"
            evo_render = self.small_font.render(evo_text, True, (255, 200, 100))
            self.screen.blit(evo_render, (580, bar_y + 35))
        
        if self.game_over:
            # Darken background (game area only)
            overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (GAME_OFFSET_X, GAME_OFFSET_Y))
            
            # Center text on game area
            center_x = GAME_OFFSET_X + GAME_WIDTH // 2
            center_y = GAME_OFFSET_Y + GAME_HEIGHT // 2
            
            if self.won:
                msg = self.font.render("YOU WIN! Press R to restart", True, YELLOW)
            else:
                msg = self.font.render("GAME OVER! Press R to restart", True, RED)
            rect = msg.get_rect(center=(center_x, center_y))
            self.screen.blit(msg, rect)
            
            # Show final score
            final_score = self.font.render(f"Final Score: {self.score}", True, WHITE)
            score_rect = final_score.get_rect(center=(center_x, center_y + 40))
            self.screen.blit(final_score, score_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(10)  # 10 FPS for visible movement
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Pacman Game with ML Agents')
    parser.add_argument('--ml', action='store_true', help='Use ML-powered ghost agents')
    parser.add_argument('--model', type=str, default='models/ghost_agent/best_model.zip',
                       help='Path to trained ghost model')
    parser.add_argument('--no-coord', action='store_true', help='Disable coordinated ghost attacks')
    parser.add_argument('--sphere-view', action='store_true', help='Render play area on a 3D sphere (camera follows player); requires PyOpenGL')
    args = parser.parse_args()
    
    print("=" * 60)
    print("👻🧠 OBSERVE: Agentic Ghosts Intelligence")
    print("=" * 60)
    print("\n📋 Controls:")
    print("   Arrow Keys - Move Pacman")
    print("   SPACEBAR   - ⚡ TELEPORT (escape ghost pack!)")
    print("   S / SHIFT  - 🏃 SUPER SPEED (2x faster!)")
    print("   G          - 🎲 GENERATE NEW MAZE (GAN-powered!)")
    print("   R          - Restart game")
    print()
    print("💡 Powers: 3 charges each, cooldown between uses")
    print("   ⚡ Teleport - Instant escape to safe location")
    print("   🏃 Speed   - Move 2x faster for 5 seconds")
    print()
    print("🌐 360° WRAP: Play area wraps like a sphere (torus)—exit")
    print("   top to appear at bottom, left to right; no hard edges!")
    print()
    print("💨 POWER FART: Eat a power pellet to release a deadly")
    print("   shockwave that eliminates ALL ghosts within half")
    print("   the board width! (+400 pts per ghost killed)")
    print()
    print("🎲 PROCEDURAL MAZES: Press G to generate a new maze!")
    print("   Uses GAN-inspired algorithms for unique layouts")
    print()
    
    if not args.no_coord:
        print("👻 Ghost Coordination: ENABLED")
        print("   🔴 Blinky: CHASER - Direct pursuit")
        print("   🩷 Pinky:  AMBUSHER - Predicts your moves")
        print("   🩵 Inky:   BLOCKER - Blocks escape routes")
        print("   🟠 Clyde:  PATROL - Flanking maneuvers")
        print()
        print("🐺 ALPHA PACK HUNTING: When a ghost gets close,")
        print("   it becomes the ALPHA (👑) and others FOLLOW!")
        print("   Watch for the golden crown - that's the leader!")
    else:
        print("👻 Ghost Coordination: Disabled")
    
    if args.ml:
        print(f"\n🤖 ML Ghosts: Enabled (model: {args.model})")
    if args.sphere_view:
        print("\n🌐 Sphere view: Play on a 3D sphere (install PyOpenGL if prompted)")
    
    print("\n" + "=" * 60)
    print()
    
    game = Game(
        use_ml_ghosts=args.ml, 
        ghost_model_path=args.model,
        coordinated=not args.no_coord,
        sphere_view=args.sphere_view,
    )
    game.run()


if __name__ == '__main__':
    main()
