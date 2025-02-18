import pygame
import numpy as np
import math
from pygame.locals import *

pygame.init()
pygame.display.set_caption("Deepseek-R1")
orig_width, orig_height = 320, 320
ss_factor = 1
width, height = orig_width * ss_factor, orig_height * ss_factor
screen = pygame.display.set_mode((orig_width, orig_height))
aspect_ratio = width / height
x = np.linspace(-1, 1, width)
y = np.linspace(1, -1, height) * (1 / aspect_ratio)
X, Y = np.meshgrid(x, y)
Z = -np.ones_like(X)
ray_dirs = np.stack((X, Y, Z), axis=-1)
ray_dirs /= np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
sphere_radius = 1
sphere_color = np.array([1.0, 0.2, 0.2])
sphere_specular = np.array([1.0, 1.0, 1.0])
shininess = 64
light_color = np.array([1.0, 1.0, 1.0])
ambient_intensity = 0.1
time = 0.0
clock = pygame.time.Clock()

def calculate_intersection(ray_dirs, sphere_center, radius):
    oc = -sphere_center
    b = 2 * np.sum(ray_dirs * oc, axis=-1)
    c = np.dot(oc, oc) - radius**2
    discriminant = b**2 - 4 * c
    
    hit_mask = discriminant >= 0
    t = np.full(discriminant.shape, -1.0)
    
    if np.any(hit_mask):
        sqrt_disc = np.sqrt(discriminant[hit_mask])
        b_hit = b[hit_mask]
        t0 = (-b_hit - sqrt_disc) / 2
        t1 = (-b_hit + sqrt_disc) / 2
        
        t_valid = np.where((t0 > 0) & (t1 > 0), np.minimum(t0, t1),
                          np.where(t0 > 0, t0, t1))
        valid_mask = t_valid > 0
        hit_mask[hit_mask] = valid_mask
        t[hit_mask] = t_valid[valid_mask]
    
    return hit_mask, t

def calculate_lighting(hit_mask, t, ray_dirs, sphere_center, radius, light_pos):
    hit_indices = np.where(hit_mask)
    pixels = np.zeros((height, width, 3))
    
    if not np.any(hit_mask):
        return pixels
    
    D_hit = ray_dirs[hit_indices]
    t_hit = t[hit_mask]
    P = t_hit[:, np.newaxis] * D_hit
    
    normal = (P - sphere_center) / radius
    
    L = light_pos - P
    L_len = np.linalg.norm(L, axis=-1, keepdims=True)
    L_dir = L / L_len
    
    N_dot_L = np.sum(normal * L_dir, axis=-1)
    N_dot_L = np.maximum(N_dot_L, 0)
    diffuse = N_dot_L[:, np.newaxis] * light_color * sphere_color
    
    R = 2 * N_dot_L[:, np.newaxis] * normal - L_dir
    V_dir = -P / np.linalg.norm(P, axis=-1, keepdims=True)
    R_dot_V = np.sum(R * V_dir, axis=-1)
    R_dot_V = np.maximum(R_dot_V, 0)
    specular = (R_dot_V ** shininess)[:, np.newaxis] * light_color * sphere_specular
    
    color = ambient_intensity * sphere_color + diffuse + specular
    pixels[hit_indices] = np.clip(color, 0, 1)
    
    return pixels

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
    
    sphere_y = 2.0 * math.sin(time) 
    sphere_center = np.array([0.0, sphere_y, -4.0])
    light_pos = np.array([
        4.0 * math.cos(time*0.5),
        5.0 + 1.0 * math.sin(time*0.7),
        -3.0 + 2.0 * math.sin(time*0.3)
    ])
    time += 0.02
    
    hit_mask, t = calculate_intersection(ray_dirs, sphere_center, sphere_radius)
    pixels = calculate_lighting(hit_mask, t, ray_dirs, sphere_center, sphere_radius, light_pos)
    oc = sphere_center
    distance_to_center = np.linalg.norm(ray_dirs * np.array([1, aspect_ratio, 1]), axis=-1)
    soft_mask = np.clip(1.0 - (distance_to_center - sphere_radius) * 0.5, 0.0, 1.0)[..., np.newaxis]
    background = np.array([0.1, 0.2, 0.4])
    final_pixels = background * (1 - soft_mask) + pixels * soft_mask
    surf = pygame.surfarray.make_surface((final_pixels * 255).astype(np.uint8))
    scaled_surf = pygame.transform.smoothscale(surf, (orig_width, orig_height))
    screen.blit(scaled_surf, (0, 0))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()