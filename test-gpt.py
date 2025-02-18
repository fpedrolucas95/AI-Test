import pygame
import numpy as np
import sys
import time

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)

def intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius):
    if ray_origin.ndim == 1:
        oc = ray_origin[np.newaxis, np.newaxis, :] - sphere_center
    else:
        oc = ray_origin - sphere_center
    b = 2.0 * np.sum(ray_dir * oc, axis=2)
    c = np.sum(oc ** 2, axis=2) - sphere_radius ** 2
    discriminant = b ** 2 - 4 * c
    t = np.full(ray_dir.shape[:2], np.inf)
    mask = discriminant >= 0
    sqrt_disc = np.sqrt(discriminant[mask])
    t0 = (-b[mask] - sqrt_disc) / 2.0
    t1 = (-b[mask] + sqrt_disc) / 2.0
    t_candidate = np.where(t0 > 1e-3, t0, np.inf)
    t_candidate = np.where((t1 > 1e-3) & (t1 < t_candidate), t1, t_candidate)
    t[mask] = t_candidate
    return t

def intersect_plane(ray_origin, ray_dir, plane_point, plane_normal):
    denom = np.sum(ray_dir * plane_normal, axis=2)
    t = np.full(ray_dir.shape[:2], np.inf)
    if ray_origin.ndim == 1:
        num = np.dot(plane_point - ray_origin, plane_normal)
    else:
        num = np.sum((plane_point - ray_origin) * plane_normal, axis=2)
    valid = np.abs(denom) > 1e-6
    t_val = num / denom
    t[valid] = np.where(t_val[valid] > 1e-3, t_val[valid], np.inf)
    return t

def reflect(I, N):
    return I - 2 * np.expand_dims(np.sum(I * N, axis=2), axis=2) * N

def render_scene(time_elapsed, width, height):
    camera = np.array([0.0, 0.0, 0.0])
    sphere_center = np.array([0.0, 1.0 + 0.5 * np.sin(time_elapsed * 2), 5.0])
    sphere_radius = 1.0
    plane_point = np.array([0.0, -1.0, 0.0])
    plane_normal = np.array([0.0, 1.0, 0.0])
    light_pos = np.array([5.0 * np.cos(time_elapsed), 5.0, 5.0 * np.sin(time_elapsed)])
    material_sphere = {
        "color": np.array([1.0, 0.0, 0.0]),
        "ambient": 0.1,
        "diffuse": 0.6,
        "specular": 0.3,
        "shininess": 50,
    }
    material_plane = {
        "color": np.array([0.5, 0.5, 0.5]),
        "ambient": 0.1,
        "diffuse": 0.9,
        "specular": 0.0,
        "shininess": 1,
    }
    aspect = width / height
    fov = np.pi / 3
    i = np.arange(width)
    j = np.arange(height)
    px, py = np.meshgrid(i, j)
    x = (2 * (px + 0.5) / width - 1) * np.tan(fov / 2) * aspect
    y = (1 - 2 * (py + 0.5) / height) * np.tan(fov / 2)
    z = np.ones_like(x)
    ray_dir = np.stack((x, y, z), axis=2)
    ray_dir = normalize(ray_dir)
    ray_origin = camera
    t_sphere = intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius)
    t_plane = intersect_plane(ray_origin, ray_dir, plane_point, plane_normal)
    t = np.minimum(t_sphere, t_plane)
    object_hit = np.zeros((height, width), dtype=np.int32)
    object_hit[t_sphere < t_plane] = 1
    object_hit[t_plane < t_sphere] = 2
    image = np.zeros((height, width, 3))
    hit_mask = t < np.inf
    p = ray_dir * t[..., np.newaxis]
    normal = np.zeros_like(p)
    mask_sphere = object_hit == 1
    if np.any(mask_sphere):
        normal[mask_sphere] = normalize(p[mask_sphere] - sphere_center)
    mask_plane = object_hit == 2
    if np.any(mask_plane):
        normal[mask_plane] = plane_normal
    view_dir = normalize(camera - p)
    L = normalize(light_pos - p)
    epsilon = 1e-3
    shadow_origin = p + normal * epsilon
    t_shadow_sphere = intersect_sphere(shadow_origin, L, sphere_center, sphere_radius)
    t_shadow_plane = intersect_plane(shadow_origin, L, plane_point, plane_normal)
    t_shadow = np.minimum(t_shadow_sphere, t_shadow_plane)
    dist_to_light = np.linalg.norm(light_pos - p, axis=2)
    in_shadow = t_shadow < dist_to_light
    mat_ambient = np.zeros((height, width))
    mat_diffuse = np.zeros((height, width))
    mat_specular = np.zeros((height, width))
    mat_shininess = np.zeros((height, width))
    mat_color = np.zeros((height, width, 3))
    mat_ambient[mask_sphere] = material_sphere["ambient"]
    mat_diffuse[mask_sphere] = material_sphere["diffuse"]
    mat_specular[mask_sphere] = material_sphere["specular"]
    mat_shininess[mask_sphere] = material_sphere["shininess"]
    mat_color[mask_sphere] = material_sphere["color"]
    mat_ambient[mask_plane] = material_plane["ambient"]
    mat_diffuse[mask_plane] = material_plane["diffuse"]
    mat_specular[mask_plane] = material_plane["specular"]
    mat_shininess[mask_plane] = material_plane["shininess"]
    mat_color[mask_plane] = material_plane["color"]
    dot_nl = np.sum(normal * L, axis=2)
    dot_nl = np.maximum(dot_nl, 0)
    diffuse_term = mat_diffuse[..., np.newaxis] * mat_color * dot_nl[..., np.newaxis]
    R = reflect(-L, normal)
    R = normalize(R)
    dot_rv = np.sum(R * view_dir, axis=2)
    dot_rv = np.maximum(dot_rv, 0)
    specular_term = mat_specular[..., np.newaxis] * (
        dot_rv[..., np.newaxis] ** mat_shininess[..., np.newaxis]
    )
    ambient_term = mat_ambient[..., np.newaxis] * mat_color
    shading = ambient_term + np.where(in_shadow[..., np.newaxis], 0, diffuse_term + specular_term)
    image[hit_mask] = shading[hit_mask]
    solid_color = np.array([0.2, 0.2, 0.2])
    image[~hit_mask] = solid_color
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image

def main():
    pygame.init()
    width, height = 320, 320
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("GPT o3-mini-high")
    clock = pygame.time.Clock()
    start_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        current_time = time.time() - start_time
        image = render_scene(current_time, width, height)
        surface = pygame.surfarray.make_surface(np.flipud(np.transpose(image, (1, 0, 2))))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()