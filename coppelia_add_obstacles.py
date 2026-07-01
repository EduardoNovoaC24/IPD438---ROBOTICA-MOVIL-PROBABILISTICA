"""
Agrega los 15 obstáculos del escenario v2 (world.py v2: muro + portón 1.75 m)
a la escena CoppeliaSim, leyendo las posiciones desde output/map.json.

Versión 3: diferenciación visual entre tipos de obstáculos:
  - Obstáculos del MURO (x ≈ 5.8): gris oscuro/rojo
  - Obstáculos de relleno: gris estándar

El script borra obstáculos previos ("obstacle_*") antes de crearlos para
poder re-correrse sin duplicar objetos.

USO:
  1. Abrir escena CoppeliaSim con el G2T montado.
  2. Detener la simulación.
  3. Correr: python3 coppelia_add_obstacles.py
  4. Verificar visualmente (muro + portón + obstáculos de relleno).
  5. Guardar la escena con Ctrl+S.

Requiere: coppeliasim_zmqremoteapi_client (pip install coppeliasim-zmqremoteapi-client)
          CoppeliaSim corriendo con el servidor ZMQ habilitado (puerto 23000).
"""
import json
import math
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

MAP_JSON_PATH = os.path.join("output", "map.json")

OBST_HEIGHT   = 0.30
OBST_Z        = OBST_HEIGHT / 2.0

COLOR_WALL    = [0.75, 0.20, 0.15]
COLOR_FILL    = [0.55, 0.55, 0.55]

WALL_X        = 5.8
WALL_X_TOL    = 0.05

client = RemoteAPIClient()
sim    = client.getObject("sim")

removed = 0
i = 0
while True:
    handle = sim.getObject(f"/obstacle_{i}", {"noError": True})
    if handle == -1:
        break
    sim.removeObjects([handle])
    removed += 1
    i += 1
print(f"Obstáculos previos eliminados: {removed}")

if not os.path.exists(MAP_JSON_PATH):
    raise FileNotFoundError(
        f"No se encontró '{MAP_JSON_PATH}'.\n"
        "Ejecuta primero: python3 generate_dataset.py"
    )

with open(MAP_JSON_PATH) as f:
    map_def = json.load(f)

obstacles = map_def["obstacles"]
print(f"Agregando {len(obstacles)} obstáculos (world.py v2: muro + portón 1.75 m)...")

n_wall = 0
for i, (cx, cy, r) in enumerate(obstacles):
    diameter = 2.0 * r
    handle = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder,
        [diameter, diameter, OBST_HEIGHT],
        0,
    )
    sim.setObjectAlias(handle, f"obstacle_{i}")
    sim.setObjectPosition(handle, -1, [float(cx), float(cy), float(OBST_Z)])

    is_wall = abs(cx - WALL_X) < WALL_X_TOL
    color = COLOR_WALL if is_wall else COLOR_FILL
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)

    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 1)

    tag = "MURO" if is_wall else "relleno"
    print(f"  obstacle_{i:02d} [{tag}]: centro=({cx:.2f}, {cy:.2f}) r={r} m")
    if is_wall:
        n_wall += 1

print(f"\nResumen: {n_wall} obstáculos del muro (rojo) + "
      f"{len(obstacles) - n_wall} de relleno (gris)")
print("Portón en x=5.8, gap libre y=[2.90, 4.65] (1.75 m).")
print("\nListo. Verifica visualmente y guarda la escena (Ctrl+S).")
