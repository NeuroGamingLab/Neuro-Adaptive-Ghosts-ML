// Constants
const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;
const CELL_SIZE = 20;
const GRID_WIDTH = WINDOW_WIDTH / CELL_SIZE;
const GRID_HEIGHT = WINDOW_HEIGHT / CELL_SIZE;

// Colors
const BLACK = '#000000';
const YELLOW = '#FFFF00';
const WHITE = '#FFFFFF';
const BLUE = '#0000FF';
const RED = '#FF0000';
const PINK = '#FFC0CB';
const CYAN = '#00FFFF';
const ORANGE = '#FFA500';

// Directions
const UP = [0, -1];
const DOWN = [0, 1];
const LEFT = [-1, 0];
const RIGHT = [1, 0];
const STOP = [0, 0];

class Maze {
    constructor() {
        // Create a simple maze layout
        // 1 = wall, 0 = path, 2 = dot, 3 = power pellet
        this.grid = [
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
        ];
        this.dots_count = this.grid.reduce((sum, row) => 
            sum + row.filter(cell => cell === 2 || cell === 3).length, 0);
    }
    
    isWall(x, y) {
        if (x < 0 || x >= this.grid[0].length || y < 0 || y >= this.grid.length) {
            return true;
        }
        return this.grid[y][x] === 1;
    }
    
    getCell(x, y) {
        if (x < 0 || x >= this.grid[0].length || y < 0 || y >= this.grid.length) {
            return 1;
        }
        return this.grid[y][x];
    }
    
    setCell(x, y, value) {
        if (x >= 0 && x < this.grid[0].length && y >= 0 && y < this.grid.length) {
            this.grid[y][x] = value;
        }
    }
    
    draw(ctx) {
        for (let y = 0; y < this.grid.length; y++) {
            for (let x = 0; x < this.grid[y].length; x++) {
                const cell = this.grid[y][x];
                const rectX = x * CELL_SIZE;
                const rectY = y * CELL_SIZE;
                
                if (cell === 1) { // Wall
                    ctx.fillStyle = BLUE;
                    ctx.fillRect(rectX, rectY, CELL_SIZE, CELL_SIZE);
                } else if (cell === 2) { // Dot
                    ctx.fillStyle = YELLOW;
                    ctx.beginPath();
                    ctx.arc(rectX + CELL_SIZE / 2, rectY + CELL_SIZE / 2, 2, 0, Math.PI * 2);
                    ctx.fill();
                } else if (cell === 3) { // Power pellet
                    ctx.fillStyle = YELLOW;
                    ctx.beginPath();
                    ctx.arc(rectX + CELL_SIZE / 2, rectY + CELL_SIZE / 2, 6, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
    }
}

class Pacman {
    constructor(x, y) {
        this.grid_x = x;
        this.grid_y = y;
        this.pixel_x = x * CELL_SIZE;
        this.pixel_y = y * CELL_SIZE;
        this.direction = [...STOP];
        this.next_direction = [...STOP];
        this.speed = 0.1;
        this.offset = 0;
        this.mouth_open = true;
        this.mouth_angle = 0;
    }
    
    update(maze) {
        // Try to change direction if requested
        if (this.next_direction[0] !== 0 || this.next_direction[1] !== 0) {
            const next_x = this.grid_x + this.next_direction[0];
            const next_y = this.grid_y + this.next_direction[1];
            if (!maze.isWall(next_x, next_y)) {
                this.direction = [...this.next_direction];
                this.next_direction = [...STOP];
            }
        }
        
        // Move in current direction
        if (this.direction[0] !== 0 || this.direction[1] !== 0) {
            this.offset += this.speed;
            
            if (this.offset >= 1.0) {
                this.offset = 0;
                this.grid_x += this.direction[0];
                this.grid_y += this.direction[1];
                
                // Wrap around screen edges
                if (this.grid_x < 0) {
                    this.grid_x = GRID_WIDTH - 1;
                } else if (this.grid_x >= GRID_WIDTH) {
                    this.grid_x = 0;
                }
                
                // Check for wall collision
                if (maze.isWall(this.grid_x, this.grid_y)) {
                    this.grid_x -= this.direction[0];
                    this.grid_y -= this.direction[1];
                    this.direction = [...STOP];
                    this.offset = 0;
                }
            }
        }
        
        this.pixel_x = this.grid_x * CELL_SIZE + this.offset * CELL_SIZE * this.direction[0];
        this.pixel_y = this.grid_y * CELL_SIZE + this.offset * CELL_SIZE * this.direction[1];
        
        // Animate mouth
        this.mouth_angle += 0.2;
        this.mouth_open = (Math.floor(this.mouth_angle) % 2) === 0;
    }
    
    setDirection(direction) {
        this.next_direction = [...direction];
    }
    
    draw(ctx) {
        const center_x = this.pixel_x + CELL_SIZE / 2;
        const center_y = this.pixel_y + CELL_SIZE / 2;
        const radius = CELL_SIZE / 2 - 2;
        
        ctx.fillStyle = YELLOW;
        
        if (this.mouth_open) {
            // Draw pacman with mouth (pie slice shape)
            // Arc covers 300° (5π/3), leaving 60° gap for mouth
            ctx.beginPath();
            if (this.direction[0] === 1 && this.direction[1] === 0) { // RIGHT
                // Mouth at 0°, arc from 30° to 330°
                ctx.arc(center_x, center_y, radius, Math.PI / 6, Math.PI * 11 / 6);
            } else if (this.direction[0] === -1 && this.direction[1] === 0) { // LEFT
                // Mouth at 180°, arc from 210° to 150° (going through 0°)
                ctx.arc(center_x, center_y, radius, Math.PI * 7 / 6, Math.PI * 17 / 6);
            } else if (this.direction[0] === 0 && this.direction[1] === -1) { // UP
                // Mouth at 270°, arc from 300° to 240° (going through 0°)
                ctx.arc(center_x, center_y, radius, Math.PI * 5 / 3, Math.PI * 10 / 3);
            } else if (this.direction[0] === 0 && this.direction[1] === 1) { // DOWN
                // Mouth at 90°, arc from 120° to 60° (going through 180°)
                ctx.arc(center_x, center_y, radius, Math.PI * 2 / 3, Math.PI * 7 / 3);
            } else {
                // Default: facing right
                ctx.arc(center_x, center_y, radius, Math.PI / 6, Math.PI * 11 / 6);
            }
            ctx.lineTo(center_x, center_y);
            ctx.fill();
        } else {
            ctx.beginPath();
            ctx.arc(center_x, center_y, radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

class Ghost {
    constructor(x, y, color) {
        this.grid_x = x;
        this.grid_y = y;
        this.pixel_x = x * CELL_SIZE;
        this.pixel_y = y * CELL_SIZE;
        this.color = color;
        const directions = [UP, DOWN, LEFT, RIGHT];
        this.direction = [...directions[Math.floor(Math.random() * directions.length)]];
        this.speed = 0.08;
        this.offset = 0;
        this.frightened = false;
        this.frightened_timer = 0;
    }
    
    update(maze, pacman) {
        if (this.frightened) {
            this.frightened_timer--;
            if (this.frightened_timer <= 0) {
                this.frightened = false;
            }
        }
        
        // Simple AI: move randomly but avoid walls
        this.offset += this.speed;
        
        if (this.offset >= 1.0) {
            this.offset = 0;
            this.grid_x += this.direction[0];
            this.grid_y += this.direction[1];
            
            // Wrap around screen edges
            if (this.grid_x < 0) {
                this.grid_x = GRID_WIDTH - 1;
            } else if (this.grid_x >= GRID_WIDTH) {
                this.grid_x = 0;
            }
            
            // Check for wall collision
            if (maze.isWall(this.grid_x, this.grid_y)) {
                this.grid_x -= this.direction[0];
                this.grid_y -= this.direction[1];
                // Choose a new random direction
                const possible_dirs = [UP, DOWN, LEFT, RIGHT];
                this.shuffleArray(possible_dirs);
                for (const dir of possible_dirs) {
                    const next_x = this.grid_x + dir[0];
                    const next_y = this.grid_y + dir[1];
                    if (!maze.isWall(next_x, next_y)) {
                        this.direction = [...dir];
                        break;
                    }
                }
            } else {
                // Occasionally change direction randomly
                if (Math.random() < 0.1) {
                    const possible_dirs = [UP, DOWN, LEFT, RIGHT];
                    this.shuffleArray(possible_dirs);
                    for (const dir of possible_dirs) {
                        const next_x = this.grid_x + dir[0];
                        const next_y = this.grid_y + dir[1];
                        if (!maze.isWall(next_x, next_y) && 
                            !(dir[0] === -this.direction[0] && dir[1] === -this.direction[1])) {
                            this.direction = [...dir];
                            break;
                        }
                    }
                }
            }
        }
        
        this.pixel_x = this.grid_x * CELL_SIZE + this.offset * CELL_SIZE * this.direction[0];
        this.pixel_y = this.grid_y * CELL_SIZE + this.offset * CELL_SIZE * this.direction[1];
    }
    
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
    
    setFrightened(duration = 300) {
        this.frightened = true;
        this.frightened_timer = duration;
    }
    
    draw(ctx) {
        const center_x = this.pixel_x + CELL_SIZE / 2;
        const center_y = this.pixel_y + CELL_SIZE / 2;
        const radius = CELL_SIZE / 2 - 2;
        
        const color = this.frightened ? '#0000FF' : this.color;
        
        // Draw ghost body (rounded rectangle)
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(center_x, center_y - radius / 2, radius, 0, Math.PI, true);
        ctx.fillRect(center_x - radius, center_y - radius / 2, radius * 2, radius * 2);
        
        // Draw ghost bottom (wavy)
        ctx.fillRect(center_x - radius, center_y, radius * 2, radius);
        
        // Draw eyes
        const eye_size = 3;
        ctx.fillStyle = WHITE;
        ctx.beginPath();
        ctx.arc(center_x - radius / 2, center_y - radius / 2, eye_size, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(center_x + radius / 2, center_y - radius / 2, eye_size, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = BLACK;
        ctx.beginPath();
        ctx.arc(center_x - radius / 2, center_y - radius / 2, eye_size / 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(center_x + radius / 2, center_y - radius / 2, eye_size / 2, 0, Math.PI * 2);
        ctx.fill();
    }
}

class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.maze = new Maze();
        this.pacman = new Pacman(20, 23);
        this.ghosts = [
            new Ghost(18, 14, RED),
            new Ghost(20, 14, PINK),
            new Ghost(22, 14, CYAN),
            new Ghost(20, 12, ORANGE)
        ];
        this.score = 0;
        this.lives = 3;
        this.game_over = false;
        this.won = false;
        this.lastTime = 0;
        
        this.setupEventListeners();
        this.updateUI();
        this.gameLoop(0);
    }
    
    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp') {
                this.pacman.setDirection(UP);
            } else if (e.key === 'ArrowDown') {
                this.pacman.setDirection(DOWN);
            } else if (e.key === 'ArrowLeft') {
                this.pacman.setDirection(LEFT);
            } else if (e.key === 'ArrowRight') {
                this.pacman.setDirection(RIGHT);
            } else if (e.key === 'r' || e.key === 'R') {
                if (this.game_over) {
                    this.restart();
                }
            }
        });
    }
    
    checkCollisions() {
        // Check if pacman eats a dot
        const cell = this.maze.getCell(this.pacman.grid_x, this.pacman.grid_y);
        if (cell === 2) { // Dot
            this.maze.setCell(this.pacman.grid_x, this.pacman.grid_y, 0);
            this.score += 10;
            this.maze.dots_count--;
        } else if (cell === 3) { // Power pellet
            this.maze.setCell(this.pacman.grid_x, this.pacman.grid_y, 0);
            this.score += 50;
            this.maze.dots_count--;
            // Make all ghosts frightened
            this.ghosts.forEach(ghost => ghost.setFrightened(300));
        }
        
        // Check if pacman collides with a ghost
        for (const ghost of this.ghosts) {
            if (this.pacman.grid_x === ghost.grid_x && 
                this.pacman.grid_y === ghost.grid_y) {
                if (ghost.frightened) {
                    // Ghost is eaten
                    ghost.grid_x = 18;
                    ghost.grid_y = 14;
                    ghost.frightened = false;
                    this.score += 200;
                } else {
                    // Pacman is caught
                    this.lives--;
                    if (this.lives <= 0) {
                        this.game_over = true;
                    } else {
                        // Reset positions
                        this.pacman.grid_x = 20;
                        this.pacman.grid_y = 23;
                        this.pacman.direction = [...STOP];
                        this.pacman.next_direction = [...STOP];
                        this.pacman.offset = 0;
                        this.ghosts.forEach((ghost, index) => {
                            ghost.grid_x = 18 + index * 2;
                            ghost.grid_y = 14;
                            ghost.offset = 0;
                        });
                    }
                }
            }
        }
        
        // Check win condition
        if (this.maze.dots_count === 0) {
            this.won = true;
            this.game_over = true;
        }
    }
    
    update() {
        if (!this.game_over) {
            this.pacman.update(this.maze);
            this.ghosts.forEach(ghost => ghost.update(this.maze, this.pacman));
            this.checkCollisions();
            this.updateUI();
        }
    }
    
    draw() {
        // Clear canvas
        this.ctx.fillStyle = BLACK;
        this.ctx.fillRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        
        // Draw maze
        this.maze.draw(this.ctx);
        
        if (!this.game_over) {
            // Draw pacman
            this.pacman.draw(this.ctx);
            
            // Draw ghosts
            this.ghosts.forEach(ghost => ghost.draw(this.ctx));
        }
        
        // Draw game over message
        if (this.game_over) {
            const gameOverDiv = document.getElementById('gameOver');
            const messageDiv = document.getElementById('gameOverMessage');
            gameOverDiv.classList.remove('hidden');
            if (this.won) {
                messageDiv.textContent = 'YOU WIN!';
                messageDiv.style.color = YELLOW;
            } else {
                messageDiv.textContent = 'GAME OVER!';
                messageDiv.style.color = RED;
            }
        } else {
            document.getElementById('gameOver').classList.add('hidden');
        }
    }
    
    updateUI() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('lives').textContent = this.lives;
    }
    
    restart() {
        this.maze = new Maze();
        this.pacman = new Pacman(20, 23);
        this.ghosts = [
            new Ghost(18, 14, RED),
            new Ghost(20, 14, PINK),
            new Ghost(22, 14, CYAN),
            new Ghost(20, 12, ORANGE)
        ];
        this.score = 0;
        this.lives = 3;
        this.game_over = false;
        this.won = false;
        this.updateUI();
    }
    
    gameLoop(timestamp) {
        const deltaTime = timestamp - this.lastTime;
        this.lastTime = timestamp;
        
        this.update();
        this.draw();
        
        requestAnimationFrame((ts) => this.gameLoop(ts));
    }
}

// Start the game when page loads
window.addEventListener('DOMContentLoaded', () => {
    new Game();
});
