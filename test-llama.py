import pygame
import numpy as np

WIDTH, HEIGHT = 320, 320
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SOURCE = np.array([5.0, 5.0, 5.0])
LIGHT_INTENSITY = 1.0
MATERIAL_DIFFUSE = 0.5
MATERIAL_SPECULAR = 0.5
MATERIAL_SHININESS = 32.0

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, ray_origin, ray_direction):
        L = self.center - ray_origin
        tca = np.dot(L, ray_direction)
        if tca < 0:
            return None
        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius * self.radius:
            return None
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        return t0, t1

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

def calculate_lighting(normal, view_direction, light_direction):
    diffuse = max(0, np.dot(normal, light_direction)) * MATERIAL_DIFFUSE
    specular = max(0, np.dot(normal, light_direction)) ** MATERIAL_SHININESS * MATERIAL_SPECULAR
    return diffuse + specular

def render(sphere, screen):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            ray_origin = np.array([0.0, 0.0, 0.0])
            ray_direction = np.array([(x - WIDTH // 2) / WIDTH, (y - HEIGHT // 2) / HEIGHT, 1.0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            ray = Ray(ray_origin, ray_direction)

            intersection = sphere.intersect(ray.origin, ray.direction)
            if intersection is not None:
                intersection_point = ray.origin + ray.direction * intersection[0]
                normal = (intersection_point - sphere.center) / sphere.radius
                view_direction = -ray.direction
                light_direction = (LIGHT_SOURCE - intersection_point) / np.linalg.norm(LIGHT_SOURCE - intersection_point)
                lighting = calculate_lighting(normal, view_direction, light_direction)
                color = int(lighting * 255)
                screen.set_at((x, y), (color, color, color))
            else:
                screen.set_at((x, y), BLACK)

def main():
    pygame.init()
    pygame.display.set_caption("Llama-3.3-70B")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    sphere = Sphere(np.array([0.0, 0.0, -5.0]), 1.0)
    velocity = 0.01

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sphere.center[1] += velocity
        if sphere.center[1] > 2.0 or sphere.center[1] < -2.0:
            velocity = -velocity

        render(sphere, screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()