# IPD-482 — Robótica Móvil Probabilística · Guía 3

**Universidad Técnica Federico Santa María · Primer semestre 2026**
**Profesor:** Franco Jorquera Pezoa | **Estudiante:** Eduardo Novoa

---

## Descripción

Implementación completa de un pipeline de navegación autónoma para un sistema **G2T (tractor + 1 trailer articulado)** en un entorno 2D con obstáculos. Cubre SLAM, fusión EKF, planificación de caminos y seguimiento de trayectorias, todo en Python puro. CoppeliaSim se usa únicamente al final para reproducir las poses calculadas y grabar el video de evidencia.

El escenario incluye un **muro de 8 obstáculos** en x = 5.8 m con un portón de **1.75 m** de paso libre, diseñado para forzar maniobras de curva cerrada (~44°) y un cruce del portón a ~33°, lo que genera ángulos de articulación relevantes para el análisis de factibilidad del trailer.

---

## Estructura del repositorio

```
Guia 3/
├── world.py                  # Escenario: mapa 12×8 m, obstáculos, waypoints
├── robot.py                  # Cinemática G2T (tractor + trailer)
├── lidar.py                  # Modelo de sensor LiDAR (FOV 270°, 1040 haces)
├── generate_dataset.py       # Parte 1 — generación de dataset GT + odometría
├── slam_occupancy_grid.py    # Parte 2 — SLAM con occupancy grid log-odds
├── ekf_fusion.py             # Parte 3 — fusión EKF (odometría + sensor de pose)
├── noise_analysis.py         # Parte 4 — análisis de sensibilidad al ruido
├── path_planning.py          # Parte 5 — planificación por campos potenciales
├── trajectory_following.py   # Parte 6 — seguimiento de trayectoria + EKF
├── coppelia_add_obstacles.py # Helper: carga obstáculos en CoppeliaSim
├── coppelia_playback.py      # Helper: reproduce trayectoria en CoppeliaSim
└── output/                   # Figuras, métricas JSON, datasets NPZ/CSV
```

---

## Pipeline de ejecución

Los scripts deben ejecutarse **en orden**; cada uno lee los archivos generados por el anterior.

```
world.py  ──►  robot.py  ──►  lidar.py  ──►  generate_dataset.py
    ──►  slam_occupancy_grid.py
    ──►  ekf_fusion.py
    ──►  noise_analysis.py
    ──►  path_planning.py
    ──►  trajectory_following.py
```

```bash
python generate_dataset.py
python slam_occupancy_grid.py
python ekf_fusion.py
python noise_analysis.py
python path_planning.py
python trajectory_following.py
```

Los resultados se guardan automáticamente en `output/`.

---

## Parámetros físicos del G2T

| Parámetro | Valor |
|---|---|
| Distancia hitch → eje trailer (L₁) | 0.973 m |
| Semi-ancho tractor | 0.241 m |
| Semi-ancho trailer | 0.347 m |
| Offset LiDAR (eje X local tractor) | −0.245 m |
| LiDAR FOV / rayos / rango máx | 270° / 1040 / 30 m |

**Modelo cinemático:**
```
ẋ = v·cos(θ)       ẏ = v·sin(θ)
θ̇ = ω             ψ̇₁ = ω − (v/L₁)·sin(ψ₁)
```

---

## Resultados por parte

| Parte | Algoritmo | Métrica clave |
|---|---|---|
| 1 — Dataset | Integración cinemática | Clearance mínimo real: **0.197 m** ✓ |
| 2 — SLAM | Occupancy grid log-odds + Bresenham | RMSE odometría vs GT: **0.010 m** |
| 3 — EKF | Extended Kalman Filter (pose completa) | Mejora **2.57×** (0.054 → 0.021 m) |
| 4 — Ruido | Sweep ruido bajo/nominal/alto | Mejora EKF crece con ruido (2.17× → 2.94×) |
| 5 — Path planning | Campos potenciales (ζ=1.2, η=2.0, ρ₀=1.0 m) | dist. mín. obstáculo: **0.601 m**, 0 perturbaciones |
| 6 — Traj. following | Controlador P (v=0.4 m/s, k=2.5) + EKF | ψ₁\_max = **35.8°** (jackknife: False), RMSE = 0.017 m |

---

## Dependencias

```bash
pip install numpy scipy matplotlib
```

Para el video de evidencia se requiere **CoppeliaSim** con la escena G2T cargada (jerarquía: `tractor → hitch_arm → Revolute_joint → hitch_pin → trailer1`).

---

## Estado del proyecto

- [x] Parte 1 — Escenario y dataset
- [x] Parte 2 — SLAM occupancy grid
- [x] Parte 3 — Fusión EKF
- [x] Parte 4 — Análisis de ruido
- [x] Parte 5 — Path planning
- [x] Parte 6 — Trajectory following
- [ ] Video en CoppeliaSim (`coppelia_add_obstacles.py` + `coppelia_playback.py`)
- [ ] Informe IEEE en LaTeX (formato IEEEtran, español)

---

## Simplificaciones MVP declaradas

- Occupancy grid binario en lugar de EKF-SLAM completo
- Campos potenciales en lugar de RRT/RRT*
- Controlador P simple en lugar de Pure Pursuit/MPC
- Estado reducido a 1 trailer (enunciado original contempla 2)
- Planificación de campos potenciales sobre punto, clearance del trailer verificada post-planificación
