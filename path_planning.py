"""
Parte 5 - Guía 3: Planificación de trayectoria mediante campos potenciales
artificiales (Opción A del enunciado).

Simplificación declarada: la planificación trata al TRACTOR como un punto
(su centro) en el espacio de configuración (x,y) -- no se planifica
directamente sobre el sistema articulado completo. La factibilidad para
el trailer (psi1) se evalúa después, en la Parte 6, al ejecutar la
trayectoria con el modelo cinemático G2T completo: ahí se observa si el
seguimiento genera ángulos de articulación excesivos (riesgo de jackknife).

Campo atractivo:
    Uatt(q) = 1/2 * zeta * ||q - qgoal||^2
    grad Uatt = zeta * (q - qgoal)

Campo repulsivo (por obstáculo, rho = distancia a la SUPERFICIE del
obstáculo, no al centro):
    Urep(q) = 1/2 * eta * (1/rho - 1/rho0)^2   si rho <= rho0
            = 0                                 si rho > rho0
    grad Urep = eta * (1/rho0 - 1/rho) * (1/rho^2) * d(rho)/dq
    d(rho)/dq = (q - centro_obs) / ||q - centro_obs||

U(q) = Uatt(q) + Urep(q); se desciende por gradiente con paso fijo.

Manejo de mínimos locales (MVP): si ||grad U|| cae bajo un umbral lejos
de la meta, se aplica una perturbación aleatoria pequeña para escapar,
y se registra el evento para el análisis crítico pedido en el enunciado.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "output"

with open(f"{DATA_DIR}/map.json") as f:
    map_def = json.load(f)
MAP_W, MAP_H = map_def["map_w"], map_def["map_h"]
OBSTACLES = np.array(map_def["obstacles"])
START_XY = np.array(map_def["start_pose"][:2])
GOAL_XY = np.array(map_def["goal_xy"])

ZETA = 1.2
ETA = 2.0
RHO0 = 1.0
STEP = 0.05
MAX_ITERS = 4000
GOAL_TOL = 0.15
STUCK_GRAD_THR = 0.05
STUCK_PATIENCE = 15


def rho_to_obstacles(q, obstacles):
    d_centers = np.hypot(obstacles[:, 0] - q[0], obstacles[:, 1] - q[1])
    return d_centers - obstacles[:, 2], d_centers


def grad_att(q, qgoal, zeta):
    return zeta * (q - qgoal)


def grad_rep(q, obstacles, eta, rho0):
    rho, d_centers = rho_to_obstacles(q, obstacles)
    grad = np.zeros(2)
    for i in range(len(obstacles)):
        if rho[i] <= rho0 and rho[i] > 1e-6:
            direction = (q - obstacles[i, :2]) / max(d_centers[i], 1e-6)
            coef = eta * (1.0 / rho0 - 1.0 / rho[i]) * (1.0 / rho[i] ** 2)
            grad += coef * direction
    return grad


def plan(zeta, eta, rho0, start, goal, obstacles, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    q = start.astype(float).copy()
    path = [q.copy()]
    stuck_count = 0
    n_perturbations = 0
    oscillation_flag = False

    for it in range(MAX_ITERS):
        g_att = grad_att(q, goal, zeta)
        g_rep = grad_rep(q, obstacles, eta, rho0)
        g_total = g_att + g_rep
        gnorm = np.linalg.norm(g_total)

        dist_goal = np.linalg.norm(q - goal)
        if dist_goal < GOAL_TOL:
            break

        if gnorm < STUCK_GRAD_THR:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count > STUCK_PATIENCE:
            kick = rng.normal(0, 1, 2)
            kick = kick / max(np.linalg.norm(kick), 1e-6) * STEP * 3
            q = q + kick
            stuck_count = 0
            n_perturbations += 1
            path.append(q.copy())
            continue

        direction = -g_total / max(gnorm, 1e-9)
        q = q + STEP * direction
        path.append(q.copy())

        if it > 60 and it % 40 == 0:
            recent = np.array(path[-40:])
            net_disp = np.linalg.norm(recent[-1] - recent[0])
            path_len = np.sum(np.linalg.norm(np.diff(recent, axis=0), axis=1))
            if path_len > 1e-6 and net_disp / path_len < 0.15:
                oscillation_flag = True

    path = np.array(path)
    reached = np.linalg.norm(path[-1] - goal) < GOAL_TOL
    if verbose:
        print(f"zeta={zeta} eta={eta} rho0={rho0} | iters={len(path)} | "
              f"llegó={reached} | perturbaciones={n_perturbations} | oscilación={oscillation_flag}")
    return path, reached, n_perturbations, oscillation_flag


path_nom, reached_nom, n_pert_nom, osc_nom = plan(ZETA, ETA, RHO0, START_XY, GOAL_XY, OBSTACLES)

min_dists = []
for q in path_nom:
    rho, _ = rho_to_obstacles(q, OBSTACLES)
    min_dists.append(rho.min())
min_dists = np.array(min_dists)

print(f"\nCamino nominal: {len(path_nom)} puntos, longitud "
      f"{np.sum(np.linalg.norm(np.diff(path_nom, axis=0), axis=1)):.2f} m")
print(f"Distancia mínima a obstáculos durante el recorrido: {min_dists.min():.3f} m")

GAIN_SETTINGS = {
    "eta_bajo (menos evasivo)":  dict(zeta=1.2, eta=0.5, rho0=1.0),
    "nominal":                    dict(zeta=1.2, eta=2.0, rho0=1.0),
    "eta_alto (muy evasivo)":     dict(zeta=1.2, eta=6.0, rho0=1.0),
}
sensitivity_results = {}
paths_sensitivity = {}
for name, params in GAIN_SETTINGS.items():
    p, reached, n_pert, osc = plan(**params, start=START_XY, goal=GOAL_XY, obstacles=OBSTACLES, seed=1)
    rho_along = np.array([rho_to_obstacles(q, OBSTACLES)[0].min() for q in p])
    sensitivity_results[name] = {
        "reached": bool(reached),
        "n_perturbations": n_pert,
        "oscillation": bool(osc),
        "min_dist_to_obstacle_m": float(rho_along.min()),
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))),
    }
    paths_sensitivity[name] = p

print("\n=== Sensibilidad a las ganancias ===")
for k, v in sensitivity_results.items():
    print(k, v)

np.savez(f"{DATA_DIR}/potential_field_path.npz", path=path_nom, min_dists=min_dists)
with open(f"{DATA_DIR}/path_planning_metrics.json", "w") as fjson:
    json.dump({
        "nominal": {"zeta": ZETA, "eta": ETA, "rho0": RHO0,
                    "reached": bool(reached_nom), "n_perturbations": n_pert_nom,
                    "oscillation": bool(osc_nom), "min_dist_obstacle_m": float(min_dists.min()),
                    "path_len_m": float(np.sum(np.linalg.norm(np.diff(path_nom, axis=0), axis=1)))},
        "sensitivity": sensitivity_results,
    }, fjson, indent=2)

xs = np.linspace(0, MAP_W, 240)
ys = np.linspace(0, MAP_H, 160)
XX, YY = np.meshgrid(xs, ys)
U = 0.5 * ZETA * ((XX - GOAL_XY[0]) ** 2 + (YY - GOAL_XY[1]) ** 2)
for cx, cy, r in OBSTACLES:
    rho_grid = np.hypot(XX - cx, YY - cy) - r
    rep = np.where(rho_grid <= RHO0, 0.5 * ETA * (1.0 / np.maximum(rho_grid, 1e-3) - 1.0 / RHO0) ** 2, 0.0)
    U += rep
U_clip = np.clip(U, 0, np.percentile(U, 92))

fig, ax = plt.subplots(figsize=(9.5, 6.5))
cf = ax.contourf(XX, YY, U_clip, levels=40, cmap="viridis")
plt.colorbar(cf, ax=ax, label="U(q) (recortado para visualización)")
for cx, cy, r in OBSTACLES:
    ax.add_patch(plt.Circle((cx, cy), r, color="white", ec="red", lw=1.2))
ax.plot(path_nom[:, 0], path_nom[:, 1], color="orange", lw=2.5, label="Trayectoria planificada")
ax.plot(*START_XY, "g^", ms=12, label="Inicio")
ax.plot(*GOAL_XY, "r*", ms=16, label="Meta")
ax.set_xlim(0, MAP_W)
ax.set_ylim(0, MAP_H)
ax.set_aspect("equal")
ax.set_title(f"Parte 5 - Campos potenciales (ζ={ZETA}, η={ETA}, ρ0={RHO0} m)\n"
             f"perturbaciones por mínimo local: {n_pert_nom} | dist. mín. a obstáculo: {min_dists.min():.2f} m")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/fig_potential_field.png", dpi=150)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(9, 6.5))
for cx, cy, r in OBSTACLES:
    ax2.add_patch(plt.Circle((cx, cy), r, color="lightgray", ec="dimgray"))
colors = {"eta_bajo (menos evasivo)": "tab:blue", "nominal": "tab:orange", "eta_alto (muy evasivo)": "tab:green"}
for name, p in paths_sensitivity.items():
    ax2.plot(p[:, 0], p[:, 1], label=name, color=colors[name], lw=2)
ax2.plot(*START_XY, "g^", ms=12)
ax2.plot(*GOAL_XY, "r*", ms=16)
ax2.set_xlim(0, MAP_W)
ax2.set_ylim(0, MAP_H)
ax2.set_aspect("equal")
ax2.set_title("Sensibilidad de la trayectoria a la ganancia repulsiva η")
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(f"{DATA_DIR}/fig_potential_field_sensitivity.png", dpi=150)
plt.close(fig2)

print("\nGuardado: fig_potential_field.png, fig_potential_field_sensitivity.png, "
      "potential_field_path.npz, path_planning_metrics.json")
