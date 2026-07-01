"""
Parte 1 - Guía 3: Escenario de simulación y generación de dataset.

Ejecuta el sistema G2T a lo largo de un recorrido con curvas cerradas que
cruza el portón del muro (world.py v2), registrando:
  - Ground truth de pose (tractor + psi1) e integración fina del modelo
  - Odometría (integración gruesa del mismo modelo -> deriva natural)
  - Escaneos LiDAR 2D (270°, 1040 haces, 30 m) en cada paso de dataset
  - Comandos de control (v, w)

Verificación de colisión (función check_collisions):
  Para cada paso del dataset calcula el clearance REAL entre la SUPERFICIE
  del tractor (radio_seguridad = half_w_tractor = 0.241 m) y la SUPERFICIE
  del trailer (radio_seguridad = half_w_trailer = 0.347 m) con cada
  obstáculo circular.  Clearance mínimo aceptable: 0.15 m.

Salidas (en ./output):
  dataset_poses.csv     -> t, gt_*, odom_*, v_cmd, w_cmd
  dataset_scans.npz     -> angles (rad), ranges (N_steps x N_beams)
  map.json              -> definición del mapa/obstáculos/waypoints
  fig_trayectoria.png   -> ground truth vs odometría vs mapa
  fig_lidar_ejemplo.png -> ejemplo de escaneo LiDAR sobre el mapa
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import world
import robot
import lidar

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

HALF_W_TRACTOR = 0.241
HALF_W_TRAILER = 0.347
CLEARANCE_MIN  = 0.15

FINE_DT = 0.01
DATASET_DT = 0.1
MAX_TIME = 90.0
GOAL_TOL = 0.3

obstacles = world.obstacles_array()
controller = robot.WaypointController(world.WAYPOINTS, v_nom=0.5, k_heading=2.0)

gt_state = np.array(world.START_POSE + (0.0,))
odom_state = gt_state.copy()

t = 0.0
n_fine_substeps = int(round(DATASET_DT / FINE_DT))

rows = []
scans = []
angles_ref = lidar.beam_angles()

while t < MAX_TIME:
    v_cmd, w_cmd = controller.compute(gt_state)

    for _ in range(n_fine_substeps):
        gt_state = robot.step(gt_state, v_cmd, w_cmd, FINE_DT)

    odom_state = robot.step(odom_state, v_cmd, w_cmd, DATASET_DT)

    lx, ly, ltheta = robot.lidar_pose(gt_state)
    _, ranges = lidar.scan(lx, ly, ltheta, obstacles)
    scans.append(ranges)

    rows.append([
        t,
        gt_state[0], gt_state[1], gt_state[2], gt_state[3],
        odom_state[0], odom_state[1], odom_state[2], odom_state[3],
        v_cmd, w_cmd,
    ])

    t += DATASET_DT

    if controller.done:
        break

rows = np.array(rows)
scans = np.array(scans)

print(f"Pasos simulados: {len(rows)}  |  tiempo final: {rows[-1,0]:.2f} s")
print(f"Pose final (gt): x={rows[-1,1]:.2f} y={rows[-1,2]:.2f} "
      f"theta={np.rad2deg(rows[-1,3]):.1f} deg  psi1={np.rad2deg(rows[-1,4]):.1f} deg")
print(f"Forma escaneos LiDAR: {scans.shape}  (steps x beams)")


def check_collisions(rows, obstacles, half_w_tractor=HALF_W_TRACTOR,
                     half_w_trailer=HALF_W_TRAILER):
    obs_arr = np.asarray(obstacles, dtype=float)
    min_clearance = np.inf
    step_idx = -1
    obs_idx_min = -1
    who = ""

    for i, row in enumerate(rows):
        state = np.array([row[1], row[2], row[3], row[4]])
        x, y = state[0], state[1]

        d_tractor = (np.hypot(obs_arr[:, 0] - x, obs_arr[:, 1] - y)
                     - obs_arr[:, 2] - half_w_tractor)
        min_t = float(d_tractor.min())
        min_t_idx = int(d_tractor.argmin())

        trailer = robot.trailer_pose(state)
        xt, yt = trailer[0], trailer[1]
        d_trailer = (np.hypot(obs_arr[:, 0] - xt, obs_arr[:, 1] - yt)
                     - obs_arr[:, 2] - half_w_trailer)
        min_tr = float(d_trailer.min())
        min_tr_idx = int(d_trailer.argmin())

        if min_t < min_clearance:
            min_clearance = min_t
            step_idx = i
            obs_idx_min = min_t_idx
            who = "tractor"

        if min_tr < min_clearance:
            min_clearance = min_tr
            step_idx = i
            obs_idx_min = min_tr_idx
            who = "trailer"

    return min_clearance, step_idx, obs_idx_min, who


min_cl, si, oi, who = check_collisions(rows, world.OBSTACLES)
obs_info = world.OBSTACLES[oi]
t_min = rows[si, 0]
print(f"\n--- Verificación de clearance real (ancho físico incluido) ---")
print(f"Clearance mínimo ({who}): {min_cl:.3f} m")
print(f"  Ocurre en t={t_min:.2f} s (paso {si}), obstáculo [{oi}] = {obs_info}")
if min_cl < CLEARANCE_MIN:
    print(f"⚠  ADVERTENCIA: clearance < {CLEARANCE_MIN} m — POSIBLE COLISIÓN de {who}.")
else:
    print(f"✓  Clearance real >= {CLEARANCE_MIN} m en todo el recorrido (OK).")

header = "t,gt_x,gt_y,gt_theta,gt_psi1,odom_x,odom_y,odom_theta,odom_psi1,v_cmd,w_cmd"
np.savetxt(os.path.join(OUT_DIR, "dataset_poses.csv"), rows, delimiter=",",
           header=header, comments="")
np.savez(os.path.join(OUT_DIR, "dataset_scans.npz"), angles=angles_ref, ranges=scans)

map_def = {
    "map_w": world.MAP_W,
    "map_h": world.MAP_H,
    "start_pose": world.START_POSE,
    "goal_xy": world.GOAL_XY,
    "waypoints": world.WAYPOINTS,
    "obstacles": world.OBSTACLES,
    "lidar": {"fov_deg": lidar.FOV_DEG, "n_beams": lidar.N_BEAMS, "max_range": lidar.MAX_RANGE},
    "L1": robot.L1,
}
with open(os.path.join(OUT_DIR, "map.json"), "w") as f:
    json.dump(map_def, f, indent=2)

fig, ax = plt.subplots(figsize=(9, 6.5))
for cx, cy, r in world.OBSTACLES:
    ax.add_patch(plt.Circle((cx, cy), r, color="dimgray", alpha=0.7))
wp = np.array(world.WAYPOINTS)
ax.plot(wp[:, 0], wp[:, 1], "o--", color="lightgray", label="Waypoints (ref.)", zorder=1)
ax.plot(rows[:, 1], rows[:, 2], "-", color="tab:blue", lw=2, label="Ground truth")
ax.plot(rows[:, 5], rows[:, 6], "--", color="tab:red", lw=1.5, label="Odometría")
ax.plot(*world.START_POSE[:2], "g^", ms=12, label="Inicio")
ax.plot(*world.GOAL_XY, "r*", ms=16, label="Meta")
ax.set_xlim(0, world.MAP_W)
ax.set_ylim(0, world.MAP_H)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Escenario G2T v2 — Parte 1: dataset simulado\n"
             f"clearance mínimo real = {min_cl:.3f} m ({who})")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_trayectoria.png"), dpi=150)
plt.close(fig)

target_xy = np.array([8.3, 4.7])
d_to_target = np.hypot(rows[:, 1] - target_xy[0], rows[:, 2] - target_xy[1])
mid = int(np.argmin(d_to_target))
lx, ly, ltheta = robot.lidar_pose(np.array([rows[mid, 1], rows[mid, 2],
                                             rows[mid, 3], rows[mid, 4]]))
angles_mid, ranges_mid = lidar.scan(lx, ly, ltheta, obstacles)

fig2, ax2 = plt.subplots(figsize=(9, 6.5))
for cx, cy, r in world.OBSTACLES:
    ax2.add_patch(plt.Circle((cx, cy), r, color="dimgray", alpha=0.5))
ax2.plot(rows[:, 1], rows[:, 2], "-", color="tab:blue", alpha=0.3, lw=1)
ax2.plot(lx, ly, "ko", ms=6, label="Pose LiDAR")

hit_x = lx + ranges_mid * np.cos(ltheta + angles_mid)
hit_y = ly + ranges_mid * np.sin(ltheta + angles_mid)
ax2.scatter(hit_x, hit_y, s=3, c="tab:orange", label="Puntos LiDAR detectados")

ax2.set_xlim(0, world.MAP_W)
ax2.set_ylim(0, world.MAP_H)
ax2.set_aspect("equal")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_title(f"Ejemplo de escaneo LiDAR (t={rows[mid,0]:.1f}s, "
              f"{lidar.N_BEAMS} haces, FOV={lidar.FOV_DEG}°)")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig_lidar_ejemplo.png"), dpi=150)
plt.close(fig2)

print("\nDataset y figuras guardados en ./output")
