import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

WIDTH, HEIGHT = 320, 320
SPHERE_RADIUS = 0.5
SPHERE_POSITION = np.array([0.0, 0.0, -5.0])
SPHERE_VELOCITY = np.array([0.0, 0.1, 0.0])
GRAVITY = np.array([0.0, -0.01, 0.0])
TIME_STEP = 0.05

pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)

glutInit()

gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

pygame.display.set_caption("Phi-4")

def draw_sphere(position):
    """Draw a sphere at the given position."""
    glPushMatrix()
    glTranslatef(*position)
    glutSolidSphere(SPHERE_RADIUS, 32, 32)
    glPopMatrix()

def update_sphere_position():
    """Update the sphere's position based on velocity and gravity."""
    global SPHERE_POSITION, SPHERE_VELOCITY
    SPHERE_POSITION += SPHERE_VELOCITY
    SPHERE_VELOCITY += GRAVITY
    if SPHERE_POSITION[1] - SPHERE_RADIUS < -2.0:
        SPHERE_POSITION[1] = 2.0 + SPHERE_RADIUS
        SPHERE_VELOCITY[1] = -SPHERE_VELOCITY[1] * 0.9

def render_scene():
    """Render the scene with basic lighting."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    light_position = [5.0, 5.0, 5.0, 1.0]
    light_ambient = [0.2, 0.2, 0.2, 1.0]
    light_diffuse = [0.8, 0.8, 0.8, 1.0]
    light_specular = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

    material_ambient = [0.1, 0.1, 0.1, 1.0]
    material_diffuse = [0.7, 0.7, 0.7, 1.0]
    material_specular = [1.0, 1.0, 1.0, 1.0]
    material_shininess = 50.0

    glMaterialfv(GL_FRONT, GL_AMBIENT, material_ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular)
    glMaterialf(GL_FRONT, GL_SHININESS, material_shininess)

    draw_sphere(SPHERE_POSITION)

    pygame.display.flip()

def main():
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        update_sphere_position()
        render_scene()
        clock.tick(1 / TIME_STEP)

if __name__ == "__main__":
    main()