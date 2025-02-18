import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

WIDTH, HEIGHT = 320, 320
SPHERE_RADIUS = 32
LIGHT_POSITION = np.array([0, 0, -500])

class Sphere:
    def __init__(self, center, radius, color, velocity):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.color = np.array(color, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def update(self):
        self.center[1] += self.velocity[1]
        if self.center[1] > HEIGHT / 2 - self.radius or self.center[1] < -HEIGHT / 2 + self.radius:
            self.velocity[1] = -self.velocity[1]

def ray_trace(sphere, light_pos, ray_origin, ray_direction):
    oc = ray_origin - sphere.center
    b = np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere.radius ** 2
    discriminant = b ** 2 - c

    if discriminant >= 0:
        t1 = -b - np.sqrt(discriminant)
        t2 = -b + np.sqrt(discriminant)
        t = min(t1, t2)
        if t > 0:
            intersection = ray_origin + t * ray_direction
            normal = (intersection - sphere.center) / sphere.radius
            light_dir = light_pos - intersection
            light_dir = light_dir / np.linalg.norm(light_dir)
            diffuse = max(np.dot(normal, light_dir), 0) * sphere.color
            view_dir = -ray_direction
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            specular = max(np.dot(view_dir, reflect_dir), 0) ** 32 * np.array([1, 1, 1])

            return diffuse + specular
    return np.array([0, 0, 0])

sphere = Sphere(center=[0, 0, -300], radius=SPHERE_RADIUS, color=[1, 0, 0], velocity=[0, 5, 0])

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(WIDTH) / float(HEIGHT), 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)
    sphere.update()
    image = np.zeros((HEIGHT, WIDTH, 3))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ray_origin = np.array([0, 0, 0])
            ray_direction = np.array([(x - WIDTH / 2) / (WIDTH / 2), (y - HEIGHT / 2) / (HEIGHT / 2), -1])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            color = ray_trace(sphere, LIGHT_POSITION, ray_origin, ray_direction)
            image[y, x] = color

    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, image)
    glutSwapBuffers()

def reshape(w, h):
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = w, h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(w) / float(h), 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b'Mistral')
    init()
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutReshapeFunc(reshape)
    glutMainLoop()

if __name__ == "__main__":
    main()