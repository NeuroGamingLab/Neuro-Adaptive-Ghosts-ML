import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
STOP = (0, 0)

class Maze:
    def __init__(self):
        # Create a simple maze layout
        # 1 = wall, 0 = path, 2 = dot, 3 = power pellet
        self.grid = [
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
            [2,2,2,2,2,2,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,2,2,2,2,2,2,2,2,2,2],
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
        ]
        self.dots_count = sum(row.count(2) + row.count(3) for row in self.grid)
        
    def is_wall(self, x, y):
        if x < 0 or x >= len(self.grid[0]) or y < 0 or y >= len(self.grid):
            return True
        return self.grid[y][x] == 1
    
    def get_cell(self, x, y):
        if x < 0 or x >= len(self.grid[0]) or y < 0 or y >= len(self.grid):
            return 1
        return self.grid[y][x]
    
    def set_cell(self, x, y, value):
        if 0 <= x < len(self.grid[0]) and 0 <= y < len(self.grid):
            self.grid[y][x] = value
    
    def draw(self, screen):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                cell = self.grid[y][x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if cell == 1:  # Wall
                    pygame.draw.rect(screen, BLUE, rect)
                elif cell == 2:  # Dot
                    center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(screen, YELLOW, center, 2)
                elif cell == 3:  # Power pellet
                    center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(screen, YELLOW, center, 6)

class Pacman:
    def __init__(self, x, y):
        self.grid_x = x
        self.grid_y = y
        self.pixel_x = x * CELL_SIZE
        self.pixel_y = y * CELL_SIZE
        self.direction = STOP
        self.next_direction = STOP
        self.speed = 0.1
        self.offset = 0
        self.mouth_open = True
        self.mouth_angle = 0
        
    def update(self, maze):
        # Try to change direction if requested
        if self.next_direction != STOP:
            next_x = self.grid_x + self.next_direction[0]
            next_y = self.grid_y + self.next_direction[1]
            if not maze.is_wall(next_x, next_y):
                self.direction = self.next_direction
                self.next_direction = STOP
        
        # Move in current direction
        if self.direction != STOP:
            self.offset += self.speed
            
            if self.offset >= 1.0:
                self.offset = 0
                self.grid_x += self.direction[0]
                self.grid_y += self.direction[1]
                
                # Wrap around screen edges
                if self.grid_x < 0:
                    self.grid_x = GRID_WIDTH - 1
                elif self.grid_x >= GRID_WIDTH:
                    self.grid_x = 0
                
                # Check for wall collision
                if maze.is_wall(self.grid_x, self.grid_y):
                    self.grid_x -= self.direction[0]
                    self.grid_y -= self.direction[1]
                    self.direction = STOP
                    self.offset = 0
        
        self.pixel_x = self.grid_x * CELL_SIZE + self.offset * CELL_SIZE * self.direction[0]
        self.pixel_y = self.grid_y * CELL_SIZE + self.offset * CELL_SIZE * self.direction[1]
        
        # Animate mouth
        self.mouth_angle += 0.2
        self.mouth_open = (int(self.mouth_angle) % 2) == 0
        
    def set_direction(self, direction):
        self.next_direction = direction
        
    def draw(self, screen):
        center_x = int(self.pixel_x + CELL_SIZE // 2)
        center_y = int(self.pixel_y + CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - 2
        
        if self.mouth_open:
            # Draw pacman with mouth
            start_angle = 0
            if self.direction == RIGHT:
                start_angle = 30
            elif self.direction == LEFT:
                start_angle = 210
            elif self.direction == UP:
                start_angle = 120
            elif self.direction == DOWN:
                start_angle = 300
            else:
                start_angle = 30
            
            pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)
            # Draw mouth by drawing a triangle
            if self.direction == RIGHT:
                points = [(center_x, center_y), 
                         (center_x + radius, center_y - radius // 2),
                         (center_x + radius, center_y + radius // 2)]
            elif self.direction == LEFT:
                points = [(center_x, center_y),
                         (center_x - radius, center_y - radius // 2),
                         (center_x - radius, center_y + radius // 2)]
            elif self.direction == UP:
                points = [(center_x, center_y),
                         (center_x - radius // 2, center_y - radius),
                         (center_x + radius // 2, center_y - radius)]
            elif self.direction == DOWN:
                points = [(center_x, center_y),
                         (center_x - radius // 2, center_y + radius),
                         (center_x + radius // 2, center_y + radius)]
            else:
                points = [(center_x, center_y),
                         (center_x + radius, center_y - radius // 2),
                         (center_x + radius, center_y + radius // 2)]
            
            pygame.draw.polygon(screen, BLACK, points)
        else:
            pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)

class Ghost:
    def __init__(self, x, y, color):
        self.grid_x = x
        self.grid_y = y
        self.pixel_x = x * CELL_SIZE
        self.pixel_y = y * CELL_SIZE
        self.color = color
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.speed = 0.08
        self.offset = 0
        self.frightened = False
        self.frightened_timer = 0
        
    def update(self, maze, pacman):
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False
        
        # Simple AI: move randomly but avoid walls
        self.offset += self.speed
        
        if self.offset >= 1.0:
            self.offset = 0
            self.grid_x += self.direction[0]
            self.grid_y += self.direction[1]
            
            # Wrap around screen edges
            if self.grid_x < 0:
                self.grid_x = GRID_WIDTH - 1
            elif self.grid_x >= GRID_WIDTH:
                self.grid_x = 0
            
            # Check for wall collision
            if maze.is_wall(self.grid_x, self.grid_y):
                self.grid_x -= self.direction[0]
                self.grid_y -= self.direction[1]
                # Choose a new random direction
                possible_dirs = [UP, DOWN, LEFT, RIGHT]
                random.shuffle(possible_dirs)
                for dir in possible_dirs:
                    next_x = self.grid_x + dir[0]
                    next_y = self.grid_y + dir[1]
                    if not maze.is_wall(next_x, next_y):
                        self.direction = dir
                        break
            else:
                # Occasionally change direction randomly
                if random.random() < 0.1:
                    possible_dirs = [UP, DOWN, LEFT, RIGHT]
                    random.shuffle(possible_dirs)
                    for dir in possible_dirs:
                        next_x = self.grid_x + dir[0]
                        next_y = self.grid_y + dir[1]
                        if not maze.is_wall(next_x, next_y) and dir != (-self.direction[0], -self.direction[1]):
                            self.direction = dir
                            break
        
        self.pixel_x = self.grid_x * CELL_SIZE + self.offset * CELL_SIZE * self.direction[0]
        self.pixel_y = self.grid_y * CELL_SIZE + self.offset * CELL_SIZE * self.direction[1]
        
    def set_frightened(self, duration=300):
        self.frightened = True
        self.frightened_timer = duration
        
    def draw(self, screen):
        center_x = int(self.pixel_x + CELL_SIZE // 2)
        center_y = int(self.pixel_y + CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - 2
        
        color = (0, 0, 255) if self.frightened else self.color
        
        # Draw ghost body (rounded rectangle)
        rect = pygame.Rect(center_x - radius, center_y - radius // 2, radius * 2, radius * 2)
        pygame.draw.ellipse(screen, color, rect)
        
        # Draw ghost bottom (wavy)
        bottom_rect = pygame.Rect(center_x - radius, center_y, radius * 2, radius)
        pygame.draw.rect(screen, color, bottom_rect)
        
        # Draw eyes
        eye_size = 3
        pygame.draw.circle(screen, WHITE, (center_x - radius // 2, center_y - radius // 2), eye_size)
        pygame.draw.circle(screen, WHITE, (center_x + radius // 2, center_y - radius // 2), eye_size)
        pygame.draw.circle(screen, BLACK, (center_x - radius // 2, center_y - radius // 2), eye_size // 2)
        pygame.draw.circle(screen, BLACK, (center_x + radius // 2, center_y - radius // 2), eye_size // 2)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Pacman Game")
        self.clock = pygame.time.Clock()
        self.maze = Maze()
        self.pacman = Pacman(20, 23)
        self.ghosts = [
            Ghost(18, 14, RED),
            Ghost(20, 14, PINK),
            Ghost(22, 14, CYAN),
            Ghost(20, 12, ORANGE)
        ]
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.won = False
        self.font = pygame.font.Font(None, 36)
        
    def check_collisions(self):
        # Check if pacman eats a dot
        cell = self.maze.get_cell(self.pacman.grid_x, self.pacman.grid_y)
        if cell == 2:  # Dot
            self.maze.set_cell(self.pacman.grid_x, self.pacman.grid_y, 0)
            self.score += 10
            self.maze.dots_count -= 1
        elif cell == 3:  # Power pellet
            self.maze.set_cell(self.pacman.grid_x, self.pacman.grid_y, 0)
            self.score += 50
            self.maze.dots_count -= 1
            # Make all ghosts frightened
            for ghost in self.ghosts:
                ghost.set_frightened(300)
        
        # Check if pacman collides with a ghost
        for ghost in self.ghosts:
            if (self.pacman.grid_x == ghost.grid_x and 
                self.pacman.grid_y == ghost.grid_y):
                if ghost.frightened:
                    # Ghost is eaten
                    ghost.grid_x = 18
                    ghost.grid_y = 14
                    ghost.frightened = False
                    self.score += 200
                else:
                    # Pacman is caught
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        # Reset positions
                        self.pacman.grid_x = 20
                        self.pacman.grid_y = 23
                        self.pacman.direction = STOP
                        self.pacman.next_direction = STOP
                        self.pacman.offset = 0
                        for ghost in self.ghosts:
                            ghost.grid_x = 18 + self.ghosts.index(ghost) * 2
                            ghost.grid_y = 14
                            ghost.offset = 0
        
        # Check win condition
        if self.maze.dots_count == 0:
            self.won = True
            self.game_over = True
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.pacman.set_direction(UP)
                elif event.key == pygame.K_DOWN:
                    self.pacman.set_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    self.pacman.set_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.pacman.set_direction(RIGHT)
                elif event.key == pygame.K_r and self.game_over:
                    # Restart game
                    self.__init__()
        return True
    
    def update(self):
        if not self.game_over:
            self.pacman.update(self.maze)
            for ghost in self.ghosts:
                ghost.update(self.maze, self.pacman)
            self.check_collisions()
    
    def draw(self):
        self.screen.fill(BLACK)
        self.maze.draw(self.screen)
        
        if not self.game_over:
            self.pacman.draw(self.screen)
            for ghost in self.ghosts:
                ghost.draw(self.screen)
        
        # Draw score and lives
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.lives}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 50))
        
        # Draw game over message
        if self.game_over:
            if self.won:
                message = self.font.render("YOU WIN! Press R to restart", True, YELLOW)
            else:
                message = self.font.render("GAME OVER! Press R to restart", True, RED)
            text_rect = message.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(message, text_rect)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
