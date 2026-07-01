"""
Parte 6 - Guía 3: Seguimiento de trayectoria planificada con el G2T completo.

Ejecuta el sistema G2T (x, y, theta, psi1) siguiendo el camino de campos
potenciales calculado en la Parte 5 (potential_field_path.npz), usando el
mismo WaypointController P sobre heading que se usó en la Parte 1.

Fusión simultánea con EKF (mismos parámetros que la Parte 3) para comparar:
  1) Trayectoria planificada (campos potenciales, para el tractor como punto)
  2) Ground truth ejecutado (integración fina del modelo G2T completo)
  3) EKF estimado durante la ejecución (odometría + sensor pose simulado)
  4) Odometría pura (integración de velocidades ruidosas sin corrección EKF)

Análisis de factibilidad articulada:
  Se grafica psi1(t) — ángulo de articulación tractor-trailer durante la
  ejecución.  Un psi1 elevado (> ~70°) indicaría riesgo de jackknife; la
  trayectoria de campos potenciales, diseñada tratando al tractor como
  punto, puede generar curvas que exijan psi1 considerables en el G2T real.

Salidas (en ./output/):
  trajectory_following_results.npz  -> gt, ekf, odom, t, path_planned
  trajectory_metrics.json           -> psi1_max, RMSE EKF, tracking error, etc.
  fig_trajectory_following.png      -> planificada vs ejecutada vs EKF vs odometría
  fig_psi1_feasibility.png          -> psi1(t) vs límite jackknife
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import robot

DATA_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)

L1          = robot.L1
SIGMA_V     = 0.03
SIGMA_W     = 0.05
SIGMA_GPS_XY = 0.15
SIGMA_THETA  = np.deg2rad(3.0)
SIGMA_PSI1   = np.deg2rad(2.0)
DT          = 0.1
FINE_DT     = 0.01
MAX_TIME    = 150.0
RNG_SEED    = 42

plan_data   = np.load(f"{DATA_DIR}/potential_field_path.npz")
path_planned = plan_data["path"]

with open(f"{DATA_DIR}/map.json") as f:
    map_def = json.load(f)

MAP_W     = map_def["map_w"]
MAP_H     = map_def["map_h"]
OBSTACLES = map_def["obstacles"]
GOAL_XY   = map_def["goal_xy"]

SUBSAMPLE = 6
wp_indices = list(range(0, len(path_planned), SUBSAMPLE))
if wp_indices[-1] != len(path_planned) - 1:
    wp_indices.append(len(path_planned) - 1)
waypoints_exec = [tuple(path_planned[i]) for i in wp_indices]

print(f"Camino planificado: {len(path_planned)} puntos → {len(waypoints_exec)} waypoints (sub-muestreo x{SUBSAMPLE})")

controller = robot.WaypointController(
    waypoints_exec, v_nom=0.4, k_heading=2.5, wp_tol=0.25
)


def f_model(state, v, w, dt):
    x, y, theta, psi1 = state
    return np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + w * dt,
        psi1 + (w - (v / L1) * np.sin(psi1)) * dt,
    ])


def jac_F(state, v, dt):
    x, y, theta, psi1 = state
    F = np.eye(4)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] =  v * np.cos(theta) * dt
    F[3, 3] = 1.0 - (v / L1) * np.cos(psi1) * dt
    return F


def jac_G(state, dt):
    x, y, theta, psi1 = state
    G = np.zeros((4, 2))
    G[0, 0] = np.cos(theta) * dt
    G[1, 0] = np.sin(theta) * dt
    G[2, 1] = dt
    G[3, 0] = -(np.sin(psi1) / L1) * dt
    G[3, 1] = dt
    return G


gt_state = np.array(map_def["start_pose"] + [0.0])

rng = np.random.default_rng(RNG_SEED)

M_ekf = np.diag([SIGMA_V ** 2, SIGMA_W ** 2])
R_ekf = np.diag([SIGMA_GPS_XY ** 2, SIGMA_GPS_XY ** 2, SIGMA_THETA ** 2, SIGMA_PSI1 ** 2])
H_ekf = np.eye(4)
x_est = gt_state.copy()
P_ekf = np.diag([0.05, 0.05, np.deg2rad(5.0), np.deg2rad(5.0)]) ** 2

n_fine_substeps = int(round(DT / FINE_DT))
t = 0.0
rows_gt   = []
rows_ekf  = []
rows_odom = []
timestamps = []

x_odom = gt_state.copy()

while t < MAX_TIME:
    v_cmd, w_cmd = controller.compute(gt_state)

    for _ in range(n_fine_substeps):
        gt_state = robot.step(gt_state, v_cmd, w_cmd, FINE_DT)

    v_exec = v_cmd + rng.normal(0.0, SIGMA_V)
    w_exec = w_cmd + rng.normal(0.0, SIGMA_W)

    x_odom = f_model(x_odom, v_exec, w_exec, DT)

    x_pred = f_model(x_est, v_exec, w_exec, DT)
    F_k    = jac_F(x_est, v_exec, DT)
    G_k    = jac_G(x_est, DT)
    Q_k    = G_k @ M_ekf @ G_k.T
    P_pred = F_k @ P_ekf @ F_k.T + Q_k

    z = gt_state + rng.normal(0.0, [SIGMA_GPS_XY, SIGMA_GPS_XY, SIGMA_THETA, SIGMA_PSI1])
    y_innov = z - H_ekf @ x_pred
    y_innov[2] = np.arctan2(np.sin(y_innov[2]), np.cos(y_innov[2]))
    S = H_ekf @ P_pred @ H_ekf.T + R_ekf
    K = P_pred @ H_ekf.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y_innov
    x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
    P_ekf = (np.eye(4) - K @ H_ekf) @ P_pred

    rows_gt.append(gt_state.copy())
    rows_ekf.append(x_est.copy())
    rows_odom.append(x_odom.copy())
    timestamps.append(t)
    t += DT

    if controller.done:
        break

rows_gt   = np.array(rows_gt)
rows_ekf  = np.array(rows_ekf)
rows_odom = np.array(rows_odom)
timestamps = np.array(timestamps)
n_steps = len(rows_gt)

psi1_deg_abs = np.rad2deg(np.abs(rows_gt[:, 3]))
psi1_max_deg = float(psi1_deg_abs.max())
psi1_mean_deg = float(psi1_deg_abs.mean())

print(f"Pasos simulados: {n_steps}  |  tiempo final: {timestamps[-1]:.2f} s")
print(f"Pose final (gt): x={rows_gt[-1,0]:.2f}  y={rows_gt[-1,1]:.2f}  "
      f"theta={np.rad2deg(rows_gt[-1,2]):.1f}°  psi1={np.rad2deg(rows_gt[-1,3]):.1f}°")
print(f"psi1 máx abs: {psi1_max_deg:.1f}°  |  psi1 promedio abs: {psi1_mean_deg:.1f}°")

err_pos_ekf   = np.hypot(rows_gt[:, 0] - rows_ekf[:, 0], rows_gt[:, 1] - rows_ekf[:, 1])
err_theta_ekf = np.abs(np.arctan2(
    np.sin(rows_gt[:, 2] - rows_ekf[:, 2]),
    np.cos(rows_gt[:, 2] - rows_ekf[:, 2])))

err_pos_odom = np.hypot(rows_gt[:, 0] - rows_odom[:, 0], rows_gt[:, 1] - rows_odom[:, 1])

rmse_ekf  = float(np.sqrt(np.mean(err_pos_ekf ** 2)))
rmse_odom = float(np.sqrt(np.mean(err_pos_odom ** 2)))

gt_xy = rows_gt[:, :2]
track_err = np.array([
    float(np.min(np.hypot(p[0] - path_planned[:, 0], p[1] - path_planned[:, 1])))
    for p in gt_xy
])

psi1_rms_deg = float(np.rad2deg(np.sqrt(np.mean(rows_gt[:, 3] ** 2))))

metrics = {
    "n_steps":           n_steps,
    "tiempo_final_s":    float(timestamps[-1]),
    "psi1_max_deg":      psi1_max_deg,
    "psi1_rms_deg":      psi1_rms_deg,
    "psi1_mean_abs_deg": psi1_mean_deg,
    "jackknife_risk":    bool(psi1_max_deg > 70.0),
    "tracking_error_planificada_vs_ejecutada": {
        "mean_m": float(track_err.mean()),
        "max_m":  float(track_err.max()),
        "rmse_m": float(np.sqrt(np.mean(track_err ** 2))),
    },
    "odometria_vs_gt": {
        "rmse_pos_m": rmse_odom,
        "max_err_pos_m": float(err_pos_odom.max()),
    },
    "ekf_vs_gt": {
        "rmse_pos_m":    rmse_ekf,
        "rmse_theta_deg": float(np.rad2deg(np.sqrt(np.mean(err_theta_ekf ** 2)))),
        "max_err_pos_m": float(err_pos_ekf.max()),
    },
    "mejora_factor_pos": float(rmse_odom / rmse_ekf) if rmse_ekf > 0 else None,
    "sigma_v":  SIGMA_V, "sigma_w":  SIGMA_W,
    "sigma_gps_xy": SIGMA_GPS_XY,
    "sigma_theta_deg": 3.0, "sigma_psi1_deg": 2.0,
}
print(json.dumps(metrics, indent=2))

np.savez(f"{DATA_DIR}/trajectory_following_results.npz",
         gt=rows_gt, ekf=rows_ekf, odom=rows_odom, t=timestamps,
         path_planned=path_planned)

with open(f"{DATA_DIR}/trajectory_metrics.json", "w") as fjson:
    json.dump(metrics, fjson, indent=2)

fig, ax = plt.subplots(figsize=(9.5, 6.5))
for cx, cy, r in OBSTACLES:
    ax.add_patch(plt.Circle((cx, cy), r, color="dimgray", alpha=0.65))
ax.plot(path_planned[:, 0], path_planned[:, 1],
        color="tab:orange", lw=1.8, ls="--", label="Planificada (campos potenciales)")
ax.plot(rows_odom[:, 0], rows_odom[:, 1],
        color="tab:red", lw=1.5, ls="--", label="Odometría pura (sin corrección)")
ax.plot(rows_gt[:, 0], rows_gt[:, 1],
        color="tab:blue", lw=2.2, label="GT ejecutado (G2T completo)")
ax.plot(rows_ekf[:, 0], rows_ekf[:, 1],
        color="tab:green", lw=1.5, ls=":", label="EKF estimado")
ax.plot(*map_def["start_pose"][:2], "g^", ms=12, label="Inicio")
ax.plot(*GOAL_XY, "r*", ms=16, label="Meta")
ax.set_xlim(0, MAP_W)
ax.set_ylim(0, MAP_H)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Parte 6 — Seguimiento de trayectoria planificada (G2T completo con psi1)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/fig_trajectory_following.png", dpi=150)
fig.savefig(f"{DATA_DIR}/fig_traj_following.png", dpi=150)
plt.close(fig)

psi1_deg = np.rad2deg(rows_gt[:, 3])
jackknife_lim = 70.0

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.plot(timestamps, psi1_deg, color="tab:purple", lw=2,
         label=r"$\psi_1$(t) — ángulo de articulación")
ax2.axhline( jackknife_lim, color="red", ls="--", lw=1.2,
             label=f"Límite jackknife (~{jackknife_lim:.0f}°)")
ax2.axhline(-jackknife_lim, color="red", ls="--", lw=1.2)
ax2.axhline(0, color="gray", ls=":", lw=0.8)

ypad = max(5.0, abs(psi1_deg).max() * 0.1)
y_lo = min(psi1_deg.min() - ypad, -jackknife_lim - 5)
y_hi = max(psi1_deg.max() + ypad,  jackknife_lim + 5)
ax2.fill_between(timestamps,  jackknife_lim, y_hi, alpha=0.08, color="red")
ax2.fill_between(timestamps, y_lo, -jackknife_lim, alpha=0.08, color="red")

risk_str = "⚠ RIESGO JACKKNIFE" if psi1_max_deg > jackknife_lim else "dentro de rango seguro"
ax2.set_xlabel("t [s]")
ax2.set_ylabel(r"$\psi_1$ [°]")
ax2.set_title(
    rf"Factibilidad articulada: $\psi_1$(t) — máx: {psi1_max_deg:.1f}° ({risk_str})"
)
ax2.set_ylim(y_lo, y_hi)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{DATA_DIR}/fig_psi1_feasibility.png", dpi=150)
plt.close(fig2)

print("\nGuardado: fig_trajectory_following.png, fig_psi1_feasibility.png, "
      "trajectory_following_results.npz, trajectory_metrics.json")
