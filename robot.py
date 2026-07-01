"""
Modelo cinemático del sistema G2T (tractor articulado + trailer1).
Estado: [x, y, theta, psi1]
  x, y, theta : pose del tractor en <G>
  psi1        : ángulo de articulación tractor-trailer1

Modelo (ver informe G2, consistente con teoría estándar de sistemas
tipo tractor-trailer, docs 03_Introducción_a_robótica_móvil_cinemática.pdf):

    xdot     = v * cos(theta)
    ydot     = v * sin(theta)
    thetadot = w
    psi1dot  = w - (v / L1) * sin(psi1)

Parámetros heredados de la Guía 2 (medidos desde STL reales):
    L1 = 0.973 m   (distancia hitch tractor -> eje trailer1)
"""
import numpy as np

L1 = 0.973
LIDAR_OFFSET_X = -0.245


def step(state, v, w, dt):
    x, y, theta, psi1 = state
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += w * dt
    psi1 += (w - (v / L1) * np.sin(psi1)) * dt
    return np.array([x, y, theta, psi1])


def trailer_pose(state):
    x, y, theta, psi1 = state
    theta_t = theta - psi1
    hitch_x = x + LIDAR_OFFSET_X * np.cos(theta)
    hitch_y = y + LIDAR_OFFSET_X * np.sin(theta)
    x_t = hitch_x - L1 * np.cos(theta_t)
    y_t = hitch_y - L1 * np.sin(theta_t)
    return np.array([x_t, y_t, theta_t])


def lidar_pose(state):
    x, y, theta, _ = state
    lx = x + LIDAR_OFFSET_X * np.cos(theta)
    ly = y + LIDAR_OFFSET_X * np.sin(theta)
    return lx, ly, theta


class WaypointController:
    def __init__(self, waypoints, v_nom=0.5, k_heading=2.0, wp_tol=0.35):
        self.waypoints = list(waypoints)
        self.idx = 0
        self.v_nom = v_nom
        self.k_heading = k_heading
        self.wp_tol = wp_tol
        self.done = False

    def compute(self, state):
        x, y, theta, _ = state
        if self.idx >= len(self.waypoints):
            self.done = True
            return 0.0, 0.0

        wx, wy = self.waypoints[self.idx]
        dx, dy = wx - x, wy - y
        dist = np.hypot(dx, dy)

        if dist < self.wp_tol:
            self.idx += 1
            if self.idx >= len(self.waypoints):
                self.done = True
                return 0.0, 0.0
            wx, wy = self.waypoints[self.idx]
            dx, dy = wx - x, wy - y
            dist = np.hypot(dx, dy)

        target_heading = np.arctan2(dy, dx)
        heading_err = np.arctan2(np.sin(target_heading - theta), np.cos(target_heading - theta))

        w = self.k_heading * heading_err
        w = np.clip(w, -1.5, 1.5)
        v = self.v_nom * max(0.25, 1.0 - abs(heading_err) / (np.pi / 2))
        return v, w
