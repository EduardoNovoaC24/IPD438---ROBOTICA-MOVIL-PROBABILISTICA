"""
Parte 2 - Guía 3: SLAM mediante mapeo por occupancy grid (log-odds).

Enfoque MVP elegido (declarado y justificado en el informe):
    En vez de EKF-SLAM completo (que estima conjuntamente pose + mapa),
    se usa un occupancy grid binario actualizado con la MEJOR pose
    disponible en cada instante -> aquí, la pose de ODOMETRÍA (no ground
    truth), que es la que un sistema real tendría disponible sin acceso
    a la posición exacta.

    Esto es "mapping with an estimated pose" (vs. mapping-with-known-poses
    o full SLAM). La calidad del mapa resultante refleja directamente el
    error acumulado de la odometría, lo cual se usa como evidencia para
    justificar la fusión EKF de la Parte 3.

Modelo de sensor inverso (binario, log-odds):
    p_occ = 0.7  ->  l_occ = ln(p_occ/(1-p_occ))   =  +0.847
    p_free = 0.3 ->  l_free = ln(p_free/(1-p_free)) = -0.847
    l0 = 0 (prior 0.5, desconocido)

Para cada haz LiDAR:
    - Si el rayo impacta un obstáculo (rango < rango máx): todas las
      celdas atravesadas se marcan como libres (l_free), la celda del
      impacto se marca como ocupada (l_occ).
    - Si el rayo no impacta nada (rango == rango máx): todas las celdas
      atravesadas se marcan libres, sin marcar celda ocupada (no hay
      evidencia de obstáculo, solo de espacio libre).

Trayectoria estimada (odometría) se compara contra ground truth (RMSE).
"""
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "output"
RES = 0.10
L_OCC = 0.847
L_FREE = -0.847
L_CLAMP = 6.0
BEAM_STRIDE = 5

with open(f"{DATA_DIR}/map.json") as f:
    map_def = json.load(f)

MAP_W = map_def["map_w"]
MAP_H = map_def["map_h"]
OBSTACLES = map_def["obstacles"]

poses = np.genfromtxt(f"{DATA_DIR}/dataset_poses.csv", delimiter=",", names=True)
scans = np.load(f"{DATA_DIR}/dataset_scans.npz")
angles = scans["angles"][::BEAM_STRIDE]
ranges_all = scans["ranges"][:, ::BEAM_STRIDE]
MAX_RANGE = map_def["lidar"]["max_range"]

LIDAR_OFFSET_X = -0.245


def lidar_pose_from(x, y, theta):
    lx = x + LIDAR_OFFSET_X * np.cos(theta)
    ly = y + LIDAR_OFFSET_X * np.sin(theta)
    return lx, ly


def world_to_grid(x, y):
    col = int(np.clip(x / RES, 0, n_cols - 1))
    row = int(np.clip(y / RES, 0, n_rows - 1))
    return col, row


def bresenham(c0, r0, c1, r1):
    cells = []
    dc, dr = abs(c1 - c0), abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        cells.append((c, r))
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


n_cols = int(round(MAP_W / RES))
n_rows = int(round(MAP_H / RES))
log_odds = np.zeros((n_rows, n_cols))

n_steps = len(poses)
for i in range(n_steps):
    ox, oy, otheta = poses["odom_x"][i], poses["odom_y"][i], poses["odom_theta"][i]
    lx, ly = lidar_pose_from(ox, oy, otheta)
    c0, r0 = world_to_grid(lx, ly)

    global_angles = otheta + angles
    beam_ranges = ranges_all[i]

    for k in range(len(angles)):
        rng = beam_ranges[k]
        is_hit = rng < (MAX_RANGE - 1e-3)
        hx = lx + rng * np.cos(global_angles[k])
        hy = ly + rng * np.sin(global_angles[k])
        c1, r1 = world_to_grid(hx, hy)

        cells = bresenham(c0, r0, c1, r1)
        for (cc, rr) in cells[:-1]:
            log_odds[rr, cc] += L_FREE
        if is_hit:
            cc, rr = cells[-1]
            log_odds[rr, cc] += L_OCC
        else:
            cc, rr = cells[-1]
            log_odds[rr, cc] += L_FREE

log_odds = np.clip(log_odds, -L_CLAMP, L_CLAMP)
prob_occ = 1.0 - 1.0 / (1.0 + np.exp(log_odds))

err_xy = np.hypot(poses["odom_x"] - poses["gt_x"], poses["odom_y"] - poses["gt_y"])
rmse_pos = np.sqrt(np.mean(err_xy ** 2))
err_theta = np.rad2deg(np.abs(np.arctan2(
    np.sin(poses["odom_theta"] - poses["gt_theta"]),
    np.cos(poses["odom_theta"] - poses["gt_theta"]))))
rmse_theta = np.sqrt(np.mean(err_theta ** 2))

print(f"Grid: {n_cols}x{n_rows} celdas ({RES} m/celda)")
print(f"RMSE posición (odometría vs GT): {rmse_pos:.3f} m")
print(f"RMSE orientación (odometría vs GT): {rmse_theta:.2f} deg")
print(f"Error posición máximo: {err_xy.max():.3f} m")

fig, ax = plt.subplots(figsize=(9, 6.5))
im = ax.imshow(prob_occ, origin="lower", extent=[0, MAP_W, 0, MAP_H],
                cmap="Greys", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="P(ocupado)")

for cx, cy, r in OBSTACLES:
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="tab:red", lw=1.5, ls="--"))

ax.plot(poses["gt_x"], poses["gt_y"], color="tab:blue", lw=2, label="Ground truth")
ax.plot(poses["odom_x"], poses["odom_y"], color="tab:orange", lw=1.5, ls="--", label="Odometría (usada para mapear)")

ax.set_xlim(0, MAP_W)
ax.set_ylim(0, MAP_H)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Parte 2 - Occupancy grid (log-odds) construido con pose de odometría\n"
             "(círculos rojos punteados = obstáculos reales, para referencia)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/fig_occupancy_grid.png", dpi=150)
plt.close(fig)

np.savez(f"{DATA_DIR}/slam_occupancy_grid.npz", log_odds=log_odds, prob_occ=prob_occ,
         resolution=RES, map_w=MAP_W, map_h=MAP_H)

with open(f"{DATA_DIR}/slam_metrics.json", "w") as f:
    json.dump({
        "rmse_pos_m": float(rmse_pos),
        "rmse_theta_deg": float(rmse_theta),
        "max_err_pos_m": float(err_xy.max()),
        "grid_resolution_m": RES,
        "grid_shape": [n_rows, n_cols],
        "l_occ": L_OCC,
        "l_free": L_FREE,
        "beam_stride": BEAM_STRIDE,
    }, f, indent=2)

print("Guardado: fig_occupancy_grid.png, slam_occupancy_grid.npz, slam_metrics.json")
