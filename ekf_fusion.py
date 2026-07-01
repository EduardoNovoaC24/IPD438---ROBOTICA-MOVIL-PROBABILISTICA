"""
Parte 3 - Guía 3: Fusión sensorial mediante EKF.

Vector de estado (reducido de la ec. 1 de la guía a un solo trailer,
consistente con el sistema G2T de la Guía 2):

    x_k = [x_k, y_k, theta_k, psi1_k]^T                        (4x1)

Fuentes de información fusionadas (>= 2, según lo pedido):
    1) ODOMETRÍA (predicción): integración del modelo cinemático G2T
       con los comandos (v, w) tal como los reportaría el sistema de
       propulsión/encoders, afectados por ruido de actuación.
    2) SENSOR DE POSE SIMULADO ("GPS simulado" + brújula + estimación
       de psi1): ground truth degradado con ruido gaussiano, jugando
       el rol de una fuente de corrección externa (opción explícitamente
       permitida por el enunciado: "Ground truth degradado con ruido,
       usado como sensor simulado").

Modelo de predicción (no lineal):
    x'     = x + v*cos(theta)*dt
    y'     = y + v*sin(theta)*dt
    theta' = theta + w*dt
    psi1'  = psi1 + (w - (v/L1)*sin(psi1))*dt

Modelo de medición:
    z = h(x) = x   (medición directa y ruidosa del estado completo)
    H = I_4x4

Ruido de proceso:
    Se modela como ruido aditivo sobre los comandos u=[v,w] con
    covarianza M = diag(sigma_v^2, sigma_w^2), propagado al estado
    mediante el jacobiano G = df/du:   Q = G M G^T

Ruido de medición:
    R = diag(sigma_gps_x^2, sigma_gps_y^2, sigma_theta^2, sigma_psi1^2)

Se compara: odometría pura (sin corrección) vs EKF (con fusión) vs
ground truth, usando RMSE de posición, orientación y ángulo de
articulación.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "output"
L1 = 0.973
DT = 0.1

SIGMA_V = 0.03
SIGMA_W = 0.05
SIGMA_GPS_XY = 0.15
SIGMA_THETA = np.deg2rad(3.0)
SIGMA_PSI1 = np.deg2rad(2.0)

RNG_SEED = 42

poses = np.genfromtxt(f"{DATA_DIR}/dataset_poses.csv", delimiter=",", names=True)
n_steps = len(poses)
gt = np.stack([poses["gt_x"], poses["gt_y"], poses["gt_theta"], poses["gt_psi1"]], axis=1)
v_cmd = poses["v_cmd"]
w_cmd = poses["w_cmd"]

rng = np.random.default_rng(RNG_SEED)


def f(state, v, w, dt):
    x, y, theta, psi1 = state
    return np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + w * dt,
        psi1 + (w - (v / L1) * np.sin(psi1)) * dt,
    ])


def jac_F(state, v, theta_unused, dt):
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


v_exec = v_cmd + rng.normal(0, SIGMA_V, n_steps)
w_exec = w_cmd + rng.normal(0, SIGMA_W, n_steps)

odom_pure = np.zeros((n_steps, 4))
odom_pure[0] = gt[0]
for k in range(1, n_steps):
    odom_pure[k] = f(odom_pure[k - 1], v_exec[k], w_exec[k], DT)

meas_noise_std = np.array([SIGMA_GPS_XY, SIGMA_GPS_XY, SIGMA_THETA, SIGMA_PSI1])
z_meas = gt + rng.normal(0, meas_noise_std, size=(n_steps, 4))

M = np.diag([SIGMA_V ** 2, SIGMA_W ** 2])
R = np.diag(meas_noise_std ** 2)
H = np.eye(4)

x_est = gt[0].copy()
P = np.diag([0.05, 0.05, np.deg2rad(5), np.deg2rad(5)]) ** 2

ekf_states = np.zeros((n_steps, 4))
ekf_states[0] = x_est

for k in range(1, n_steps):
    x_pred = f(x_est, v_exec[k], w_exec[k], DT)
    F = jac_F(x_est, v_exec[k], None, DT)
    G = jac_G(x_est, DT)
    Q = G @ M @ G.T
    P_pred = F @ P @ F.T + Q

    z = z_meas[k]
    y_innov = z - H @ x_pred
    y_innov[2] = np.arctan2(np.sin(y_innov[2]), np.cos(y_innov[2]))
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_est = x_pred + K @ y_innov
    x_est[2] = np.arctan2(np.sin(x_est[2]), np.cos(x_est[2]))
    P = (np.eye(4) - K @ H) @ P_pred

    ekf_states[k] = x_est


def pos_rmse(traj):
    return np.sqrt(np.mean((traj[:, 0] - gt[:, 0]) ** 2 + (traj[:, 1] - gt[:, 1]) ** 2))


def angle_rmse(traj, col):
    err = np.arctan2(np.sin(traj[:, col] - gt[:, col]), np.cos(traj[:, col] - gt[:, col]))
    return np.rad2deg(np.sqrt(np.mean(err ** 2)))


metrics = {
    "odom_pura": {
        "rmse_pos_m": float(pos_rmse(odom_pure)),
        "rmse_theta_deg": float(angle_rmse(odom_pure, 2)),
        "rmse_psi1_deg": float(angle_rmse(odom_pure, 3)),
        "max_err_pos_m": float(np.max(np.hypot(odom_pure[:, 0] - gt[:, 0], odom_pure[:, 1] - gt[:, 1]))),
    },
    "ekf_fusion": {
        "rmse_pos_m": float(pos_rmse(ekf_states)),
        "rmse_theta_deg": float(angle_rmse(ekf_states, 2)),
        "rmse_psi1_deg": float(angle_rmse(ekf_states, 3)),
        "max_err_pos_m": float(np.max(np.hypot(ekf_states[:, 0] - gt[:, 0], ekf_states[:, 1] - gt[:, 1]))),
    },
    "mejora_factor_pos": float(pos_rmse(odom_pure) / max(pos_rmse(ekf_states), 1e-9)),
    "sigma_v": SIGMA_V, "sigma_w": SIGMA_W,
    "sigma_gps_xy": SIGMA_GPS_XY, "sigma_theta_deg": 3.0, "sigma_psi1_deg": 2.0,
}
print(json.dumps(metrics, indent=2))

with open(f"{DATA_DIR}/ekf_metrics.json", "w") as fjson:
    json.dump(metrics, fjson, indent=2)

np.savez(f"{DATA_DIR}/ekf_results.npz", gt=gt, odom_pure=odom_pure, ekf=ekf_states,
         z_meas=z_meas, t=poses["t"])

fig, ax = plt.subplots(figsize=(9, 6.5))
with open(f"{DATA_DIR}/map.json") as fmap:
    map_def = json.load(fmap)
for cx, cy, r in map_def["obstacles"]:
    ax.add_patch(plt.Circle((cx, cy), r, color="dimgray", alpha=0.5))

ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.5, label="Ground truth")
ax.plot(odom_pure[:, 0], odom_pure[:, 1], color="tab:red", lw=1.5, ls="--", label="Odometría pura")
ax.plot(ekf_states[:, 0], ekf_states[:, 1], color="tab:green", lw=1.8, label="EKF (fusión)")
ax.scatter(z_meas[::5, 0], z_meas[::5, 1], s=8, c="tab:orange", alpha=0.5, label="Mediciones GPS simuladas")

ax.set_xlim(0, map_def["map_w"])
ax.set_ylim(0, map_def["map_h"])
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Parte 3 - EKF: fusión odometría + sensor de pose simulado")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/fig_ekf_trayectorias.png", dpi=150)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
err_odom = np.hypot(odom_pure[:, 0] - gt[:, 0], odom_pure[:, 1] - gt[:, 1])
err_ekf = np.hypot(ekf_states[:, 0] - gt[:, 0], ekf_states[:, 1] - gt[:, 1])
ax2.plot(poses["t"], err_odom, color="tab:red", label=f"Odometría pura (RMSE={metrics['odom_pura']['rmse_pos_m']:.3f} m)")
ax2.plot(poses["t"], err_ekf, color="tab:green", label=f"EKF fusión (RMSE={metrics['ekf_fusion']['rmse_pos_m']:.3f} m)")
ax2.set_xlabel("t [s]")
ax2.set_ylabel("Error de posición [m]")
ax2.set_title("Evolución del error de posición vs. ground truth")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{DATA_DIR}/fig_ekf_error_tiempo.png", dpi=150)
plt.close(fig2)

print("Guardado: fig_ekf_trayectorias.png, fig_ekf_error_tiempo.png, ekf_metrics.json, ekf_results.npz")
