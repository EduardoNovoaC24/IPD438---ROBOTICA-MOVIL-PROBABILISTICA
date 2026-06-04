import math

def sysCall_init():
    global sim, handles, t, history
    sim = require('sim')
    handles = {}
    handles['tractor']  = sim.getObject('/tractor')
    handles['joint']    = sim.getObject('/Revolute_joint')
    handles['lidar']    = sim.getObject('/visionSensor')

    # Obstáculos opcionales
    try:
        handles['person1'] = sim.getObject('/obstacle_person1')
        print("[G2T] obstacle_person1 OK")
    except:
        handles['person1'] = None

    try:
        handles['box1'] = sim.getObject('/obstacle_box1')
        print("[G2T] obstacle_box1 OK")
    except:
        handles['box1'] = None

    t = 0.0
    history = []
    print("[G2T] Init OK ? Lemniscata con obstáculos")


def get_pose(t):
    w   = 0.28
    r   = 2.2
    eps = 0.0005
    tau  = w * t
    d    = 1.0 + math.sin(tau)**2
    x    = r * math.cos(tau) / d
    y    = r * math.sin(tau) * math.cos(tau) / d
    tau2 = w * (t + eps)
    d2   = 1.0 + math.sin(tau2)**2
    x2   = r * math.cos(tau2) / d2
    y2   = r * math.sin(tau2) * math.cos(tau2) / d2
    theta = math.atan2(y2 - y, x2 - x)
    return x, y, theta


def sysCall_actuation():
    global sim, handles, t, history
    try:
        x0, y0, theta0 = get_pose(t)

        history.append((x0, y0, theta0))
        delay = 20
        if len(history) >= delay + 1:
            x1, y1, theta1 = history[-(delay+1)]
        else:
            x1     = x0 - 0.98 * math.cos(theta0)
            y1     = y0 - 0.98 * math.sin(theta0)
            theta1 = theta0
        if len(history) > 300:
            history = history[-300:]

        psi1 = math.atan2(
            math.sin(theta1 - theta0),
            math.cos(theta1 - theta0)
        )

        hx = x0 + (-0.39) * math.cos(theta0)
        hy = y0 + (-0.39) * math.sin(theta0)

        # Tractor y joint
        sim.setObjectPosition(handles['tractor'],  [x0, y0, 0.0], -1)
        sim.setObjectOrientation(handles['tractor'], [0, 0, theta0], -1)
        sim.setObjectPosition(handles['joint'],    [hx, hy, 0.275], -1)
        sim.setObjectOrientation(handles['joint'],  [0, 0, theta0], -1)
        sim.setJointPosition(handles['joint'], psi1)

        # Obstáculo 1: persona caminando en línea recta (ida y vuelta)
        if handles['person1']:
            try:
                px = 1.5 * math.sin(0.35 * t)
                py = 1.8
                sim.setObjectPosition(handles['person1'], [px, py, 0.0], -1)
            except: pass

        # Obstáculo 2: caja cruzando perpendicularmente
        if handles['box1']:
            try:
                bx = 0.0
                by = 2.2 * math.sin(0.22 * t + math.pi/4)
                sim.setObjectPosition(handles['box1'], [bx, by, 0.0], -1)
            except: pass

        if int(t) % 5 == 0 and t % 1.0 < 0.05:
            w = 0.28
            tau = w * t
            lobulo = "Curva izq (+)" if math.sin(tau) > 0 else "Contracurva der (-)"
            print(f"[G2T] t={t:.1f}s [{lobulo}] "
                  f"pos=({x0:.2f},{y0:.2f}) "
                  f"psi1={math.degrees(psi1):+.1f}deg")

        t += 0.05

    except Exception as e:
        print(f"[G2T] ERROR: {e}")
        t += 0.05


def sysCall_sensing():
    global sim, handles
    sim.handleVisionSensor(handles['lidar'])


def sysCall_cleanup():
    global t
    print(f"[G2T] Fin. t={t:.1f}s")
