"""
Simulador de LiDAR 2D por raycasting analítico (intersección rayo-círculo)
contra obstáculos circulares y los bordes del mapa (tratados como paredes).

Especificación (heredada del sensor configurado en CoppeliaSim para G2,
estilo Hokuyo): FOV=270°, resolución ~0.26° (1040 haces), rango 30 m.
"""
import numpy as np
from world import MAP_W, MAP_H

FOV_DEG = 270.0
N_BEAMS = 1040
MAX_RANGE = 30.0


def beam_angles():
    half = np.deg2rad(FOV_DEG) / 2.0
    return np.linspace(-half, half, N_BEAMS)


def _ray_circle_hits(ox, oy, dirs_x, dirs_y, obstacles):
    n = len(dirs_x)
    dists = np.full(n, np.inf)
    if len(obstacles) == 0:
        return dists

    for cx, cy, r in obstacles:
        fx, fy = ox - cx, oy - cy
        b = 2 * (dirs_x * fx + dirs_y * fy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * c
        valid = disc >= 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0))
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        t = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
        t = np.where(valid, t, np.inf)
        dists = np.minimum(dists, t)
    return dists


def _ray_wall_hits(ox, oy, dirs_x, dirs_y):
    n = len(dirs_x)
    dists = np.full(n, np.inf)

    with np.errstate(divide='ignore', invalid='ignore'):
        for xw in (0.0, MAP_W):
            t = (xw - ox) / dirs_x
            y_hit = oy + t * dirs_y
            valid = (t > 1e-6) & (y_hit >= 0) & (y_hit <= MAP_H)
            dists = np.where(valid, np.minimum(dists, t), dists)
        for yw in (0.0, MAP_H):
            t = (yw - oy) / dirs_y
            x_hit = ox + t * dirs_x
            valid = (t > 1e-6) & (x_hit >= 0) & (x_hit <= MAP_W)
            dists = np.where(valid, np.minimum(dists, t), dists)
    return dists


def scan(lidar_x, lidar_y, lidar_theta, obstacles):
    angles = beam_angles()
    global_angles = lidar_theta + angles
    dirs_x = np.cos(global_angles)
    dirs_y = np.sin(global_angles)

    d_obs = _ray_circle_hits(lidar_x, lidar_y, dirs_x, dirs_y, obstacles)
    d_wall = _ray_wall_hits(lidar_x, lidar_y, dirs_x, dirs_y)

    ranges = np.minimum(d_obs, d_wall)
    ranges = np.minimum(ranges, MAX_RANGE)
    return angles, ranges
