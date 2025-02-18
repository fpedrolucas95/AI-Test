import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

width, height = 320, 320
sphere_radius = 0.5
sphere_center = np.array([0.0, 0.0, 0.0])
light_position = np.array([5.0, 5.0, 5.0])
light_intensity = np.array([1.0, 1.0, 1.0])
background_color = np.array([0.1, 0.1, 0.1, 1.0])
pygame.display.set_caption("Gemini-2")

sphere_color = np.array([0.8, 0.2, 0.2]) 
sphere_specular = np.array([0.5, 0.5, 0.5])
sphere_shininess = 50.0
camera_position = np.array([0.0, 0.0, 2.0])
camera_direction = np.array([0.0, 0.0, -1.0])
camera_up = np.array([0.0, 1.0, 0.0])

def ray_trace(ray_origin, ray_direction):
    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c

    if discriminant > 0:
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        t = min(t1, t2)

        if t > 0:
            intersection_point = ray_origin + t * ray_direction
            normal = (intersection_point - sphere_center) / sphere_radius
            light_direction = (light_position - intersection_point)
            light_distance = np.linalg.norm(light_direction)
            light_direction = light_direction / light_distance
            diffuse_intensity = np.maximum(0.0, np.dot(normal, light_direction)) * sphere_color * light_intensity
            reflection_direction = 2.0 * np.dot(normal, light_direction) * normal - light_direction
            specular_intensity = np.power(np.maximum(0.0, np.dot(reflection_direction, -ray_direction)), sphere_shininess) * sphere_specular * light_intensity

            return np.append(diffuse_intensity + specular_intensity, 1.0)

    return background_color

def render_frame(sphere_y):
    glBegin(GL_POINTS)
    for x in range(width):
        for y in range(height):
            x_ndc = (x + 0.5) / width * 2.0 - 1.0
            y_ndc = (y + 0.5) / height * 2.0 - 1.0

            ray_direction = np.array([x_ndc, -y_ndc, -1.0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            sphere_center[1] = sphere_y 

            pixel_color = ray_trace(camera_position, ray_direction)
            glColor4f(*pixel_color)
            glVertex2f(x / width * 2 -1 , y / height * 2 - 1)
    glEnd()

pygame.init()
screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
glViewport(0, 0, width, height)
glClearColor(*background_color)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(-1, 1, -1, 1, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
running = True
sphere_y = 0.0
sphere_y_speed = 0.01

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT)

    sphere_y += sphere_y_speed
    if sphere_y > 0.8 or sphere_y < -0.8:
        sphere_y_speed *= -1

    render_frame(sphere_y)

    pygame.display.flip()
    pygame.time.Clock().tick(60)

pygame.quit()