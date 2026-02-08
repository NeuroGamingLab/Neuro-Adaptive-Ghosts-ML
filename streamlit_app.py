#!/usr/bin/env python3
"""
Pacman Game - Streamlit Version
Simple button-based controls
"""

import streamlit as st
import numpy as np
import time
from PIL import Image, ImageDraw

# Page config
st.set_page_config(
    page_title="Pacman",
    page_icon="🎮",
    layout="wide"
)

# Game Constants
CELL_SIZE = 18
GRID_WIDTH = 40
GRID_HEIGHT = 31

# Colors (RGB)
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

ACTION_TO_DIR = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}


def create_maze():
    """Create the game maze."""
    return np.array([
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


class GameState:
    """Game state container."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.maze = create_maze()
        self.pacman_pos = [20, 23]
        self.ghost_positions = [[18, 14], [20, 14], [22, 14], [20, 12]]
        self.ghost_frightened = [False] * 4
        self.frightened_timer = [0] * 4
        self.score = 0
        self.lives = 3
        self.dots_remaining = np.sum((self.maze == 2) | (self.maze == 3))
        self.game_over = False
        self.won = False
        self.steps = 0
    
    def is_wall(self, x, y):
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        return self.maze[y, x] == 1
    
    def get_valid_moves(self, x, y):
        valid = []
        for action, (dx, dy) in ACTION_TO_DIR.items():
            nx, ny = x + dx, y + dy
            if nx < 0: nx = GRID_WIDTH - 1
            elif nx >= GRID_WIDTH: nx = 0
            if not self.is_wall(nx, ny):
                valid.append(action)
        return valid
    
    def move_pacman(self, direction):
        if self.game_over:
            return
        
        dx, dy = direction
        nx = self.pacman_pos[0] + dx
        ny = self.pacman_pos[1] + dy
        
        if nx < 0: nx = GRID_WIDTH - 1
        elif nx >= GRID_WIDTH: nx = 0
        
        if not self.is_wall(nx, ny):
            self.pacman_pos = [nx, ny]
            
            cell = self.maze[ny, nx]
            if cell == 2:
                self.maze[ny, nx] = 0
                self.score += 10
                self.dots_remaining -= 1
            elif cell == 3:
                self.maze[ny, nx] = 0
                self.score += 50
                self.dots_remaining -= 1
                for i in range(4):
                    self.ghost_frightened[i] = True
                    self.frightened_timer[i] = 20
        
        self.steps += 1
        self._move_ghosts()
        self._check_collisions()
        self._update_frightened()
        
        if self.dots_remaining == 0:
            self.won = True
            self.game_over = True
    
    def _move_ghosts(self):
        for i, (gx, gy) in enumerate(self.ghost_positions):
            valid = self.get_valid_moves(gx, gy)
            if not valid: continue
            
            px, py = self.pacman_pos
            
            if self.ghost_frightened[i]:
                best = max(valid, key=lambda a: abs(gx + ACTION_TO_DIR[a][0] - px) + abs(gy + ACTION_TO_DIR[a][1] - py))
            else:
                if np.random.random() < 0.3:
                    best = np.random.choice(valid)
                else:
                    best = min(valid, key=lambda a: abs(gx + ACTION_TO_DIR[a][0] - px) + abs(gy + ACTION_TO_DIR[a][1] - py))
            
            dx, dy = ACTION_TO_DIR[best]
            nx, ny = gx + dx, gy + dy
            if nx < 0: nx = GRID_WIDTH - 1
            elif nx >= GRID_WIDTH: nx = 0
            if not self.is_wall(nx, ny):
                self.ghost_positions[i] = [nx, ny]
    
    def _check_collisions(self):
        px, py = self.pacman_pos
        for i, (gx, gy) in enumerate(self.ghost_positions):
            if gx == px and gy == py:
                if self.ghost_frightened[i]:
                    self.ghost_positions[i] = [18 + i * 2, 14]
                    self.ghost_frightened[i] = False
                    self.score += 200
                else:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        self.pacman_pos = [20, 23]
                        self.ghost_positions = [[18, 14], [20, 14], [22, 14], [20, 12]]
    
    def _update_frightened(self):
        for i in range(4):
            if self.frightened_timer[i] > 0:
                self.frightened_timer[i] -= 1
                if self.frightened_timer[i] == 0:
                    self.ghost_frightened[i] = False


def render_game(state):
    """Render game to PIL Image."""
    width = GRID_WIDTH * CELL_SIZE
    height = GRID_HEIGHT * CELL_SIZE
    
    img = Image.new('RGB', (width, height), BLACK)
    draw = ImageDraw.Draw(img)
    
    # Draw maze
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell = state.maze[y, x]
            px, py = x * CELL_SIZE, y * CELL_SIZE
            
            if cell == 1:
                draw.rectangle([px, py, px + CELL_SIZE - 1, py + CELL_SIZE - 1], fill=BLUE)
            elif cell == 2:
                cx, cy = px + CELL_SIZE // 2, py + CELL_SIZE // 2
                draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=YELLOW)
            elif cell == 3:
                cx, cy = px + CELL_SIZE // 2, py + CELL_SIZE // 2
                draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=YELLOW)
    
    # Draw Pacman
    px, py = state.pacman_pos
    cx = px * CELL_SIZE + CELL_SIZE // 2
    cy = py * CELL_SIZE + CELL_SIZE // 2
    r = CELL_SIZE // 2 - 1
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=YELLOW)
    
    # Draw ghosts
    for i, (gx, gy) in enumerate(state.ghost_positions):
        cx = gx * CELL_SIZE + CELL_SIZE // 2
        cy = gy * CELL_SIZE + CELL_SIZE // 2
        r = CELL_SIZE // 2 - 1
        color = BLUE if state.ghost_frightened[i] else GHOST_COLORS[i]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        draw.ellipse([cx - 3, cy - 2, cx, cy + 1], fill=WHITE)
        draw.ellipse([cx, cy - 2, cx + 3, cy + 1], fill=WHITE)
    
    return img


def main():
    # Initialize
    if 'game' not in st.session_state:
        st.session_state.game = GameState()
        st.session_state.auto_play = False
    
    game = st.session_state.game
    
    # Header
    st.markdown("# 🎮 PACMAN")
    
    # Game stats in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Score", game.score)
    col2.metric("Lives", "❤️" * game.lives if game.lives > 0 else "💀")
    col3.metric("Dots Left", game.dots_remaining)
    col4.metric("Steps", game.steps)
    
    with col5:
        if st.button("🔄 NEW GAME", use_container_width=True):
            st.session_state.game = GameState()
            st.session_state.auto_play = False
            st.rerun()
    
    st.divider()
    
    # Layout: Controls | Game | Info
    left_col, game_col, right_col = st.columns([1, 3, 1])
    
    # Left: Movement Controls
    with left_col:
        st.markdown("### ⬆️ MOVE")
        
        # Up button
        if st.button("⬆️ UP", use_container_width=True, disabled=game.game_over, key="up"):
            game.move_pacman(UP)
            st.rerun()
        
        # Left and Right buttons
        l, r = st.columns(2)
        with l:
            if st.button("⬅️", use_container_width=True, disabled=game.game_over, key="left"):
                game.move_pacman(LEFT)
                st.rerun()
        with r:
            if st.button("➡️", use_container_width=True, disabled=game.game_over, key="right"):
                game.move_pacman(RIGHT)
                st.rerun()
        
        # Down button
        if st.button("⬇️ DOWN", use_container_width=True, disabled=game.game_over, key="down"):
            game.move_pacman(DOWN)
            st.rerun()
        
        st.divider()
        
        # Auto-play
        st.markdown("### 🤖 AUTO")
        auto_speed = st.slider("Speed", 100, 500, 200, 50, key="speed")
        
        if st.button("▶️ AUTO PLAY" if not st.session_state.auto_play else "⏹️ STOP", 
                    use_container_width=True, disabled=game.game_over):
            st.session_state.auto_play = not st.session_state.auto_play
            st.rerun()
    
    # Center: Game Board
    with game_col:
        img = render_game(game)
        st.image(img, use_container_width=True)
        
        if game.game_over:
            if game.won:
                st.success("🎉 **YOU WIN!** All dots collected!")
            else:
                st.error("💀 **GAME OVER!** Click NEW GAME to try again.")
    
    # Right: Legend
    with right_col:
        st.markdown("### 📖 LEGEND")
        st.markdown("""
        🟡 **Pacman** (You)
        
        🔴 **Blinky**
        🩷 **Pinky**
        🩵 **Inky**
        🟠 **Clyde**
        
        🔵 **Scared Ghost**
        (Eat them!)
        
        ⚪ **Dot** (+10)
        ⭕ **Power** (+50)
        👻 **Ghost** (+200)
        """)
        
        st.divider()
        
        st.markdown("### 🎯 GOAL")
        st.markdown("Collect all dots!")
    
    # Auto-play logic
    if st.session_state.auto_play and not game.game_over:
        px, py = game.pacman_pos
        valid = game.get_valid_moves(px, py)
        
        if valid:
            # Simple AI
            min_ghost_dist = min(abs(gx - px) + abs(gy - py) for gx, gy in game.ghost_positions)
            
            if min_ghost_dist < 4:
                # Run away
                best = max(valid, key=lambda a: min(
                    abs((px + ACTION_TO_DIR[a][0]) - gx) + abs((py + ACTION_TO_DIR[a][1]) - gy)
                    for gx, gy in game.ghost_positions
                ))
            else:
                # Find nearest dot
                best = np.random.choice(valid)
                min_dist = float('inf')
                for a in valid:
                    dx, dy = ACTION_TO_DIR[a]
                    nx, ny = px + dx, py + dy
                    for y in range(GRID_HEIGHT):
                        for x in range(GRID_WIDTH):
                            if game.maze[y, x] in [2, 3]:
                                d = abs(x - nx) + abs(y - ny)
                                if d < min_dist:
                                    min_dist = d
                                    best = a
            
            game.move_pacman(ACTION_TO_DIR[best])
        
        time.sleep(auto_speed / 1000)
        st.rerun()


if __name__ == "__main__":
    main()
