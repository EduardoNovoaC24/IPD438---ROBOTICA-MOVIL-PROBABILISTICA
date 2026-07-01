"""
Parte 4 - Guía 3: Incorporación y análisis de ruido.

Se declara explícitamente:
    - Distribución: ruido blanco GAUSSIANO, media cero, i.i.d. en cada paso.
    - Variables afectadas:
        * Actuación/odometría: v, w (comandos ejecutados)
        * Sensor de pose simulado: x, y, theta, psi1 (EKF, Parte 3)
        * LiDAR: rango de cada haz (nuevo en esta parte)
    - Magnitudes: se evalúan 3 niveles (bajo/nominal/alto), ver NOISE_LEVELS.

Responde a los dos requerimientos centrales de la Parte 4:
  (A) Efecto de la MAGNITUD del ruido sobre la estimación EKF
      -> sweep de niveles, comparando odometría pura vs EKF fusión.
  (B) Efecto del ruido (pose + LiDAR) sobre la calidad del mapa SLAM
      -> se reconstruye el occupancy grid en 4 escenarios y se mide
         cuantitativamente contra el mapa real (IoU).
"""
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "output"
L1 = 0.973
DT = 0.1
RES = 0.10

NOISE_LEVELS = {
    "bajo":    dict(sigma_v=0.015, sigma_w=0.025, sigma_gps_xy=0.07, sigma_theta_deg=1.5, sigma_psi1_deg=1.0),
    "nominal": dict(sigma_v=0.030, sigma_w=0.050, sigma_gps_xy=0.15, sigma_theta_deg=3.0, sigma_psi1_deg=2.0),
    "alto":    dict(sigma_v=0.060, sigma_w=0.100, sigma_gps_xy=0.30, sigma_theta_deg=6.0, sigma_psi1_deg=4.0),
}
SIGMA_RANGE_LIDAR = 0.08

poses = np.genfromtxt(f"{DATA_DIR}/dataset_poses.csv", delimiter=",", names=True)
n_steps = len(poses)
gt = np.stack([poses["gt_x"], poses["gt_y"], poses["gt_theta"], poses["gt_psi1"]], axis=1)
v_cmd = poses["v_cmd"]
w_cmd = poses["w_cmd"]

scans = np.load(f"{DATA_DIR}/dataset_scans.npz")
angles_full = scans["angles"]
ranges_full = scans["ranges"]

with open(f"{DATA_DIR}/map.json") as f:
    map_def = json.load(f)
MAP_W, MAP_H = map_def["map_w"], map_def["map_h"]
OBSTACLES = map_def["obstacles"]
MAX_RANGE = map_def["lidar"]["max_range"]
LIDAR_OFFSET_X = -0.245


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
    F[1, 2] = v * np.cos(theta) * dt
    F[3, 3] = 1 - (v / L1) * np.cos(psi1) * dt
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


def run_ekf(sigma_v, sigma_w, sigma_gps_xy, sigma_theta_deg, sigma_psi1_deg, seed=42):
    rng = np.random.default_rng(seed)
    sigma_theta = np.deg2rad(sigma_theta_deg)
    sigma_psi1 = np.deg2rad(sigma_psi1_deg)

    v_exec = v_cmd + rng.normal(0, sigma_v, n_steps)
    w_exec = w_cmd + rng.normal(0, sigma_w, n_steps)

    odom_pure = np.zeros((n_steps, 4))
    odom_pure[0] = gt[0]
    for k in range(1, n_steps):
        odom_pure[k] = f_model(odom_pure[k - 1], v_exec[k], w_exec[k], DT)

    meas_std = np.array([sigma_gps_xy, sigma_gps_xy, sigma_theta, sigma_psi1])
    z_meas = gt + rng.normal(0, meas_std, size=(n_steps, 4))

    M = np.diag([sigma_v ** 2, sigma_w ** 2])
    R = np.diag(meas_std ** 2)
    H = np.eye(4)

    x_est = gt[0].copy()
    P = np.diag([0.05, 0.05, np.deg2rad(5), np.deg2rad(5)]) ** 2
    ekf_states = np.zeros((n_steps, 4))
    ekf_states[0] = x_est

    for k in range(1, n_steps):
        x_pred = f_model(x_est, v_exec[k], w_exec[k], DT)
        F = jac_F(x_est, v_exec[k], DT)
        G = jac_G(x_est, DT)
        Q = G @ M @ G.T
        P_pred = F @ P @ F.T + Q

        y_innov = z_meas[k] - H @ x_pred
        y_innov[2] = np.arctan2(np.sin(y_innov[2]), np.cos(y_innov[2]))
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_est = x_pred + K @ y_innov
        x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
        P = (np.eye(4) - K @ H) @ P_pred
        ekf_states[k] = x_est

    def pos_rmse(traj):
        return np.sqrt(np.mean((traj[:, 0] - gt[:, 0]) ** 2 + (traj[:, 1] - gt[:, 1]) ** 2))

    return odom_pure, ekf_states, pos_rmse(odom_pure), pos_rmse(ekf_states)


sweep_results = {}
odom_nominal = ekf_nominal = None
for level, params in NOISE_LEVELS.items():
    odom_p, ekf_s, rmse_odom, rmse_ekf = run_ekf(**params)
    sweep_results[level] = {
        "rmse_odom_m": float(rmse_odom),
        "rmse_ekf_m": float(rmse_ekf),
        "mejora_factor": float(rmse_odom / max(rmse_ekf, 1e-9)),
    }
    if level == "nominal":
        odom_nominal, ekf_nominal = odom_p, ekf_s

print("=== (A) Sweep de niveles de ruido ===")
for level, r in sweep_results.items():
    print(f"{level:8s} | odom RMSE={r['rmse_odom_m']:.3f} m | EKF RMSE={r['rmse_ekf_m']:.3f} m "
          f"| mejora={r['mejora_factor']:.2f}x")

fig, ax = plt.subplots(figsize=(7.5, 5))
levels = list(NOISE_LEVELS.keys())
x_pos = np.arange(len(levels))
odom_vals = [sweep_results[l]["rmse_odom_m"] for l in levels]
ekf_vals = [sweep_results[l]["rmse_ekf_m"] for l in levels]
width = 0.35
ax.bar(x_pos - width / 2, odom_vals, width, label="Odometría pura", color="tab:red")
ax.bar(x_pos + width / 2, ekf_vals, width, label="EKF (fusión)", color="tab:green")
ax.set_xticks(x_pos)
ax.set_xticklabels(levels)
ax.set_ylabel("RMSE posición [m]")
ax.set_title("Efecto de la magnitud del ruido sobre la estimación\n(con vs. sin fusión sensorial)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/fig_noise_sweep.png", dpi=150)
plt.close(fig)


def lidar_pose_from(x, y, theta):
    lx = x + LIDAR_OFFSET_X * np.cos(theta)
    ly = y + LIDAR_OFFSET_X * np.sin(theta)
    return lx, ly


def bresenham(c0, r0, c1, r1):
    cells = []
    dc, dr = abs(c1 - c0), abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0 ,r0
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
L_OCC, L_FREE, L_CLAMP = 0.847, -0.847, 6.0
BEAM_STRIDE = 5
angles = angles_full[::BEAM_STRIDE]
ranges_clean = ranges_full[:, ::BEAM_STRIDE]


def world_to_grid(x, y):
    col = int(np.clip(x / RES, 0, n_cols - 1))
    row = int(np.clip(y / RES, 0, n_rows - 1))
    return col, row


def build_grid(traj_xytheta, ranges_arr):
    log_odds = np.zeros((n_rows, n_cols))
    for i in range(n_steps):
        ox, oy, otheta = traj_xytheta[i, 0], traj_xytheta[i, 1], traj_xytheta[i, 2]
        lx, ly = lidar_pose_from(ox, oy, otheta)
        c0, r0 = world_to_grid(lx, ly)
        global_angles = otheta + angles
        beam_ranges = ranges_arr[i]
        for k in range(len(angles)):
            rng_k = beam_ranges[k]
            is_hit = rng_k < (MAX_RANGE - 1e-3)
            hx = lx + rng_k * np.cos(global_angles[k])
            hy = ly + rng_k * np.sin(global_angles[k])
            c1, r1 = world_to_grid(hx, hy)
            cells = bresenham(c0, r0, c1, r1)
            for (cc, rr) in cells[:-1]:
                log_odds[rr, cc] += L_FREE
            cc, rr = cells[-1]
            log_odds[rr, cc] += L_OCC if is_hit else L_FREE
    log_odds = np.clip(log_odds, -L_CLAMP, L_CLAMP)
    return 1.0 - 1.0 / (1.0 + np.exp(log_odds))


def true_mask():
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    ys = (np.arange(n_rows) + 0.5) * RES
    xs = (np.arange(n_cols) + 0.5) * RES
    XX, YY = np.meshgrid(xs, ys)
    for cx, cy, r in OBSTACLES:
        mask |= (XX - cx) ** 2 + (YY - cy) ** 2 <= r ** 2
    return mask


def iou(prob_map, mask, thr=0.5):
    pred = prob_map > thr
    inter = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return inter / union if union > 0 else 0.0


rng2 = np.random.default_rng(7)
ranges_noisy = np.clip(ranges_clean + rng2.normal(0, SIGMA_RANGE_LIDAR, ranges_clean.shape), 0, MAX_RANGE)

mask_true = true_mask()

grid_gt_clean = build_grid(gt, ranges_clean)
grid_odomnoisy_clean = build_grid(odom_nominal, ranges_clean)
grid_odomnoisy_noisyrange = build_grid(odom_nominal, ranges_noisy)
grid_ekf_noisyrange = build_grid(ekf_nominal, ranges_noisy)

iou_results = {
    "a_gt_clean": iou(grid_gt_clean, mask_true),
    "b_odomnoisy_clean": iou(grid_odomnoisy_clean, mask_true),
    "c_odomnoisy_noisyrange": iou(grid_odomnoisy_noisyrange, mask_true),
    "d_ekf_noisyrange": iou(grid_ekf_noisyrange, mask_true),
}
print("\n=== (B) IoU del mapa reconstruido vs. mapa real ===")
for k, v in iou_results.items():
    print(f"{k:28s}: IoU = {v:.3f}")

fig2, axes = plt.subplots(2, 2, figsize=(11, 9))
titles = [
    ("Ground truth + LiDAR limpio\n(referencia ideal)", grid_gt_clean, iou_results["a_gt_clean"]),
    ("Odometría (con ruido) + LiDAR limpio", grid_odomnoisy_clean, iou_results["b_odomnoisy_clean"]),
    ("Odometría (con ruido) + LiDAR ruidoso", grid_odomnoisy_noisyrange, iou_results["c_odomnoisy_noisyrange"]),
    ("EKF (fusión) + LiDAR ruidoso", grid_ekf_noisyrange, iou_results["d_ekf_noisyrange"]),
]
for ax, (title, grid, iou_val) in zip(axes.flat, titles):
    ax.imshow(grid, origin="lower", extent=[0, MAP_W, 0, MAP_H], cmap="Greys", vmin=0, vmax=1)
    for cx, cy, r in OBSTACLES:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="tab:red", lw=1, ls="--"))
    ax.set_title(f"{title}\nIoU={iou_val:.3f}", fontsize=10)
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(0, MAP_H)
    ax.set_aspect("equal")
fig2.suptitle("Parte 4 - Efecto del ruido (pose y LiDAR) sobre la calidad del mapa SLAM", fontsize=12)
fig2.tight_layout()
fig2.savefig(f"{DATA_DIR}/fig_noise_slam_effect.png", dpi=150)
plt.close(fig2)

with open(f"{DATA_DIR}/noise_analysis_metrics.json", "w") as fjson:
    json.dump({
        "noise_levels": NOISE_LEVELS,
        "sweep_results": sweep_results,
        "sigma_range_lidar_m": SIGMA_RANGE_LIDAR,
        "iou_results": iou_results,
    }, fjson, indent=2)

print("\nGuardado: fig_noise_sweep.png, fig_noise_slam_effect.png, noise_analysis_metrics.json")
