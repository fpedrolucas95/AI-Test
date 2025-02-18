import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

WIDTH, HEIGHT = 320, 320

sphere_center = np.array([0.0, 0.0, -5.0])
sphere_radius = 1.0
light_position = np.array([5.0, 5.0, 5.0])
camera_position = np.array([0.0, 0.0, 0.0])
ambient_color = np.array([0.1, 0.1, 0.1])
diffuse_color = np.array([0.7, 0.7, 0.7])
specular_color = np.array([1.0, 1.0, 1.0])
shininess = 32.0
time = 0.0
bounce_height = 2.0

def ray_sphere_intersection(ray_origin, ray_direction, center, radius):
    """
    Calculate if a ray intersects with a sphere and return the intersection point if it does.
    """
    oc = ray_origin - center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant > 0:
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t > 0:
            return ray_origin + t * ray_direction
    return None

def compute_lighting(point, normal):
    """
    Compute lighting for the given point and surface normal.
    """
    light_dir = normalize(light_position - point)
    view_dir = normalize(camera_position - point)
    half_vector = normalize(light_dir + view_dir)
    
    color = ambient_color
    
    diffuse = max(0.0, np.dot(normal, light_dir))
    color += diffuse * diffuse_color
    
    specular = max(0.0, np.dot(normal, half_vector)) ** shininess
    color += specular * specular_color
    
    return np.clip(color, 0, 1)

def normalize(vector):
    """ Normalize a vector """
    return vector / np.linalg.norm(vector)

def trace_ray(ray_origin, ray_direction):
    """
    Trace a ray through the scene and compute the color.
    """
    intersection = ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius)
    if intersection is not None:
        normal = normalize(intersection - sphere_center)
        return compute_lighting(intersection, normal)
    return np.zeros(3)

def display():
    global time
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    sphere_center[1] = bounce_height * np.sin(time)
    time += 0.05

    for x in range(WIDTH):
        for y in range(HEIGHT):
            screen_x = (x - WIDTH / 2) / (WIDTH / 2)
            screen_y = (y - HEIGHT / 2) / (HEIGHT / 2) * -1
            ray_direction = normalize(np.array([screen_x, screen_y, -1.0]) - camera_position)
            
            color = trace_ray(camera_position, ray_direction)
            
            glColor3f(*color)
            glBegin(GL_POINTS)
            glVertex2i(x, y)
            glEnd()

    glutSwapBuffers()

def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)
    glMatrixMode(GL_MODELVIEW)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow(b"Grok-2")
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutReshapeFunc(reshape)
    glutMainLoop()

if __name__ == "__main__":
    main()