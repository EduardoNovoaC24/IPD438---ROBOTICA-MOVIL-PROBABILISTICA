"""
Playback de la trayectoria EJECUTADA (Parte 6, ground truth real del
sistema G2T) en CoppeliaSim, para grabar el video de evidencia.

Este script NO calcula nada -- solo reproduce visualmente las poses
(x, y, theta, psi1) que ya se calcularon en Python puro
(output/trajectory_following_results.npz, columna 'gt').

USO:
    1. Asegúrate de haber corrido antes coppelia_add_obstacles.py (una vez).
    2. Abre tu escena GUIA_3_SCENE en CoppeliaSim, simulación DETENIDA.
    3. Activa la grabación de video ANTES de correr este script:
       Barra de herramientas -> ícono de cámara/grabador (o Add-ons ->
       "Video recorder" según tu versión de CoppeliaSim).
    4. Corre este script desde la carpeta del proyecto:
         cd "IPD482 - Robotica Móvil Probabilistica/Guia 3"
         python3 coppelia_playback.py
    5. Cuando termine, detén la grabación.

Parámetros de calibración a ajustar según tu escena:
    Z_TRACTOR    : altura del tractor sobre el piso [m] (0.05 es un buen punto
                   de partida; sube si el modelo aparece enterrado).
    PSI1_SIGN    : 1.0 o -1.0 según la convención de signo del Revolute_joint
                   en tu escena (si el trailer gira al revés, cambia a -1.0).
    PLAYBACK_SPEEDUP : 1 = tiempo real (DT=0.1 s por pose); 2 = el doble.

Requiere: pip install coppeliasim-zmqremoteapi-client numpy
"""
import os
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

RESULTS_PATH = os.path.join("output", "trajectory_following_results.npz")
TRACTOR_PATH = "/tractor"
JOINT_PATH = "/Revolute_joint"
Z_TRACTOR = 0.05
PSI1_SIGN = 1.0
PLAYBACK_SPEEDUP = 1

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(
        f"No se encontró '{RESULTS_PATH}'.\n"
        "Ejecuta primero el pipeline completo:\n"
        "  python3 generate_dataset.py\n"
        "  python3 path_planning.py\n"
        "  python3 trajectory_following.py"
    )

data = np.load(RESULTS_PATH)
t  = data["t"]
gt = data["gt"]

client = RemoteAPIClient()
sim = client.getObject("sim")

tractor = sim.getObject(TRACTOR_PATH)
joint = sim.getObject(JOINT_PATH)

print(f"Reproduciendo {len(t)} poses ({t[-1]:.1f} s de dataset)...")

client.setStepping(True)
sim.startSimulation()

try:
    for i in range(len(t)):
        x, y, theta, psi1 = gt[i]
        sim.setObjectPosition(tractor, -1, [float(x), float(y), Z_TRACTOR])
        sim.setObjectOrientation(tractor, -1, [0.0, 0.0, float(theta)])
        sim.setJointPosition(joint, PSI1_SIGN * float(psi1))
        for _ in range(PLAYBACK_SPEEDUP):
            client.step()
finally:
    sim.stopSimulation()

print("Playback terminado. Detén la grabación de video si aún sigue activa.")
