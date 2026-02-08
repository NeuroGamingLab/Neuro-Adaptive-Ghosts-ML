"""
3D sphere view for Pacman: play surface rendered on a sphere that rotates
as the player moves. Requires PyOpenGL.
"""

import math
import numpy as np

# Optional OpenGL
try:
    from OpenGL.GL import (
        glEnable, glDisable, glClear, glClearColor, glViewport,
        glMatrixMode, glLoadIdentity, glTranslatef,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
        GL_PROJECTION, GL_MODELVIEW, GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glDeleteTextures,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
        glBegin, glEnd, glVertex3f, glColor3f, glTexCoord2f,
        GL_QUADS, GL_LINES, glLineWidth, glNormal3f,
    )
    from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric, gluSphere, gluQuadricTexture, gluDeleteQuadric
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


# Grid constants (must match main.py)
GRID_WIDTH = 40
GRID_HEIGHT = 31

# Colors (R,G,B) 0-1 for OpenGL
COLOR_WALL = (0.0, 0.0, 1.0)
COLOR_DOT = (1.0, 1.0, 0.0)
COLOR_POWER = (1.0, 1.0, 0.0)
COLOR_BG = (0.0, 0.0, 0.0)
COLOR_PACMAN = (1.0, 1.0, 0.0)
GHOST_COLORS_GL = [
    (1.0, 0.0, 0.0),   # Red
    (1.0, 0.75, 0.8),  # Pink
    (0.0, 1.0, 1.0),   # Cyan
    (1.0, 0.65, 0.0),  # Orange
]


def grid_to_angles(gx, gy):
    """Map grid cell (gx, gy) to sphere angles (theta, phi).
    theta = longitude [0, 2*pi], phi = latitude [-pi/2, pi/2].
    """
    theta = 2.0 * math.pi * (gx % GRID_WIDTH) / GRID_WIDTH
    # y=0 -> top of grid -> phi = pi/2; y=GRID_HEIGHT-1 -> phi = -pi/2
    phi = math.pi * (0.5 - (gy % GRID_HEIGHT) / GRID_HEIGHT)
    return theta, phi


def angles_to_xyz(theta, phi, radius=1.0):
    """Convert (theta, phi) to 3D point on sphere. Y-up."""
    x = radius * math.cos(phi) * math.cos(theta)
    y = radius * math.sin(phi)
    z = radius * math.cos(phi) * math.sin(theta)
    return (x, y, z)


def grid_to_xyz(gx, gy, radius=1.0):
    """Grid position to 3D point on sphere surface."""
    theta, phi = grid_to_angles(gx, gy)
    return angles_to_xyz(theta, phi, radius)


def build_maze_texture(grid):
    """Build RGB texture from maze grid (equirectangular). Returns texture id."""
    w, h = 256, 128
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    for j in range(h):
        for i in range(w):
            theta = 2.0 * math.pi * i / w
            phi = math.pi * (0.5 - j / h)
            gx = int((theta / (2 * math.pi)) * GRID_WIDTH) % GRID_WIDTH
            gy = int((0.5 - phi / math.pi) * GRID_HEIGHT)
            gy = max(0, min(GRID_HEIGHT - 1, gy))
            cell = grid[gy, gx]
            if cell == 1:
                tex[j, i] = (0, 0, 255)
            elif cell == 2:
                tex[j, i] = (255, 255, 0)
            elif cell == 3:
                tex[j, i] = (255, 255, 0)
            else:
                tex[j, i] = (0, 0, 0)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, tex)
    return texture_id


def draw_sphere_with_texture(radius, texture_id, slices=48, stacks=24):
    """Draw a sphere with texture using lat/long segments."""
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    for stack in range(stacks):
        phi0 = math.pi * (0.5 - stack / stacks)
        phi1 = math.pi * (0.5 - (stack + 1) / stacks)
        glBegin(GL_QUADS)
        for slice in range(slices):
            theta0 = 2 * math.pi * slice / slices
            theta1 = 2 * math.pi * (slice + 1) / slices
            for (t, p) in [(theta0, phi0), (theta1, phi0), (theta1, phi1), (theta0, phi1)]:
                u = t / (2 * math.pi)
                v = 0.5 - p / math.pi
                glTexCoord2f(u, v)
                x, y, z = angles_to_xyz(t, p, radius)
                glNormal3f(x / radius, y / radius, z / radius)
                glVertex3f(x, y, z)
        glEnd()
    glDisable(GL_TEXTURE_2D)


class SphereView:
    """Renders the game on a 3D sphere with camera following the player."""

    def __init__(self, width, height):
        if not HAS_OPENGL:
            raise RuntimeError("PyOpenGL is required for sphere view. Install with: pip install PyOpenGL PyOpenGL_accelerate")
        self.width = width
        self.height = height
        self.sphere_radius = 1.0
        self.camera_distance = 2.8
        self.texture_id = None

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50.0, self.width / max(1, self.height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def draw_frame(self, maze, pacman, ghosts):
        """Draw one frame: sphere with maze texture, camera follows pacman, entities on sphere."""
        grid = maze.grid
        # Update texture from current maze state (delete previous to avoid leak)
        if self.texture_id is not None:
            try:
                glDeleteTextures(1, [self.texture_id])
            except Exception:
                pass
        self.texture_id = build_maze_texture(grid)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera: position at pacman's point on sphere, moved outward; look at origin
        px, py, pz = grid_to_xyz(pacman.x, pacman.y, self.sphere_radius)
        cam_x = px * self.camera_distance
        cam_y = py * self.camera_distance
        cam_z = pz * self.camera_distance
        gluLookAt(cam_x, cam_y, cam_z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # Draw sphere with maze texture
        draw_sphere_with_texture(self.sphere_radius, self.texture_id)

        # Hunting lines: ghost → Pacman (or target) when not frightened
        pad = 0.06
        px, py, pz = grid_to_xyz(pacman.x, pacman.y, self.sphere_radius)
        n_p = math.sqrt(px*px + py*py + pz*pz) or 1.0
        px += px / n_p * pad
        py += py / n_p * pad
        pz += pz / n_p * pad
        glDisable(GL_TEXTURE_2D)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for i, ghost in enumerate(ghosts):
            if ghost.frightened:
                continue
            gx, gy, gz = grid_to_xyz(ghost.x, ghost.y, self.sphere_radius)
            n = math.sqrt(gx*gx + gy*gy + gz*gz) or 1.0
            gx += gx / n * pad
            gy += gy / n * pad
            gz += gz / n * pad
            tx, ty, tz = px, py, pz
            if ghost.target is not None:
                tx, ty, tz = grid_to_xyz(ghost.target[0], ghost.target[1], self.sphere_radius)
                n_t = math.sqrt(tx*tx + ty*ty + tz*tz) or 1.0
                tx += tx / n_t * pad
                ty += ty / n_t * pad
                tz += tz / n_t * pad
            color = GHOST_COLORS_GL[i % len(GHOST_COLORS_GL)]
            glColor3f(*color)
            glVertex3f(gx, gy, gz)
            glVertex3f(tx, ty, tz)
        glEnd()

        # Draw Pacman (slightly above surface)
        px, py, pz = grid_to_xyz(pacman.x, pacman.y, self.sphere_radius)
        n = math.sqrt(px*px + py*py + pz*pz) or 1.0
        pad = 0.06
        px += px / n * pad
        py += py / n * pad
        pz += pz / n * pad
        glColor3f(*COLOR_PACMAN)
        glTranslatef(px, py, pz)
        q = gluNewQuadric()
        gluSphere(q, 0.06, 12, 8)
        gluDeleteQuadric(q)
        glTranslatef(-px, -py, -pz)

        # Draw ghosts
        for i, ghost in enumerate(ghosts):
            gx, gy, gz = grid_to_xyz(ghost.x, ghost.y, self.sphere_radius)
            n = math.sqrt(gx*gx + gy*gy + gz*gz) or 1.0
            gx += gx / n * pad
            gy += gy / n * pad
            gz += gz / n * pad
            color = (0.3, 0.3, 1.0) if ghost.frightened else GHOST_COLORS_GL[i % len(GHOST_COLORS_GL)]
            glColor3f(*color)
            glTranslatef(gx, gy, gz)
            q = gluNewQuadric()
            gluSphere(q, 0.05, 10, 6)
            gluDeleteQuadric(q)
            glTranslatef(-gx, -gy, -gz)

        return self.texture_id  # caller may need to delete later


def has_opengl():
    return HAS_OPENGL
