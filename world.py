"""
Definición del escenario (mapa + obstáculos) para Guía 3 — versión 2.
Mapa rectangular de MAP_W x MAP_H metros, obstáculos circulares.

Escenario rediseñado con mayor dificultad:
  - MURO de 8 obstáculos circulares (r=0.4 m) en x=5.8 que divide el mapa
    en dos zonas.  Portón angosto de 1.2 m entre y=3.25 y y=4.45
    (centro en y=3.85).  El G2T debe negociar la entrada al portón a ~33°
    y re-orientarse tras cruzarlo — generando psi1 relevante para el
    análisis de factibilidad articulada (Parte 6).
  - Obstáculos de relleno adicionales a ambos lados del muro.
  - Waypoints con curvas más cerradas que fuerzan el cruce del portón
    y producen un recorrido en S más pronunciado.

Geometría del muro (x=5.8, r=0.4):
  Sección inferior (4 círculos):
    centros en y = 0.40, 1.20, 2.00, 2.50
    superficie superior del último: 2.50+0.40 = 2.90  ← borde inf. del portón
    (obs. 3 y 4 se solapan 0.30 m → pared sólida)
  Portón (gap libre):
    y de 2.90 a 4.65  →  ancho efectivo = 1.75 m
    (el trailer barre un rango y=[3.54, 4.00] al cruzar el muro, dado el
    lag articulado (~0.31 m por debajo del tractor al entrar, ~0.15 m por
    encima de la trayectoria del tractor al salir de la curva post-portón);
    el gap mínimo seguro resultante es 1.75 m, con clearance >= 0.15 m)
  Sección superior (4 círculos):
    centros en y = 5.05, 5.65, 6.45, 7.25
    superficie inferior del primero: 5.05-0.40 = 4.65  ← borde sup. del portón
    (subido de 4.85 para dar clearance a la curva post-portón del trailer)
"""
import numpy as np

MAP_W = 12.0
MAP_H = 8.0

START_POSE = (1.0, 1.0, 0.0)
GOAL_XY = (10.0, 6.0)

WAYPOINTS = [
    (1.0, 1.0),
    (3.0, 1.2),
    (4.5, 3.0),
    (5.8, 3.85),
    (6.5, 4.0),
    (7.5, 5.5),
    (8.5, 5.8),
    (10.0, 6.0),
]

OBSTACLES = [
    (3.0, 3.0, 0.35),
    (2.5, 5.5, 0.40),
    (5.0, 2.0, 0.30),
    (5.8, 0.40, 0.40),
    (5.8, 1.20, 0.40),
    (5.8, 2.00, 0.40),
    (5.8, 2.50, 0.40),
    (5.8, 5.05, 0.40),
    (5.8, 5.65, 0.40),
    (5.8, 6.45, 0.40),
    (5.8, 7.25, 0.40),
    (8.0, 4.5,  0.35),
    (8.6, 4.6,  0.30),
    (9.5, 2.0,  0.35),
    (9.0, 7.0,  0.30),
]


def obstacles_array():
    return np.array(OBSTACLES, dtype=float)


def in_bounds(x, y, margin=0.0):
    return margin <= x <= MAP_W - margin and margin <= y <= MAP_H - margin
