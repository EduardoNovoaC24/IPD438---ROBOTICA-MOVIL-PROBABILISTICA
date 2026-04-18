# Guia 1 P2 Novoa IPD482
# Robot Omnidireccional — 3 Ruedas Suecas
# IPD-482 Robotica Movil Probabilistica — UTFSM 2026

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import json

# PARAMETROS DEL ROBOT OMNIDIRECCIONAL
R_robot  = 0.25          # Distancia del centro del chasis a cada rueda [m]
r        = 0.05          # Radio de las ruedas suecas [m]
theta1   = 0.0           # Angulo de montaje rueda 1 [rad]
theta2   = np.deg2rad(120.0)
theta3   = np.deg2rad(240.0)
offsets  = np.array([theta1, theta2, theta3])
dt       = 0.01          # Paso de tiempo de simulacion [s]
T_ramp   = 0.5           # Tiempo de rampa del perfil trapezoidal [s]

# PARAMETROS DE LA TRAYECTORIA
v0       = 0.2           # Velocidad lineal de avance [m/s]
R_curva  = 0.5           # Radio de la semicircunferencia [m]
omega0   = v0 / R_curva  # Velocidad angular para la curva [rad/s]
t_recta  = 2.0 / v0      # Duracion de cada tramo recto [s]
t_curva  = np.pi / omega0 # Duracion de cada semicircunferencia [s]

# CINEMATICA
def cinematica_inversa(Xd, Yd, pd, phi):
    # Dado el vector de velocidad deseado del chasis en el marco global,
    # calcula las velocidades de cada rueda usando la restriccion de rodadura:
    # v_i = -Xd*sin(theta_i + phi) + Yd*cos(theta_i + phi) + R_robot*pd
    angulos = phi + offsets
    J = np.column_stack([-np.sin(angulos), np.cos(angulos),
                          np.full(3, R_robot)])
    return J @ np.array([Xd, Yd, pd])


def cinematica_directa(vw, phi):
    # Dado el vector de velocidades de ruedas, recupera la velocidad del chasis.
    # Es la inversa de J, valida por la simetria 120 grados de la configuracion.
    t = phi + offsets
    v1, v2, v3 = vw
    Xd = (2/3)*(-np.sin(t[0])*v1 - np.sin(t[1])*v2 - np.sin(t[2])*v3)
    Yd = (2/3)*( np.cos(t[0])*v1 + np.cos(t[1])*v2 + np.cos(t[2])*v3)
    pd = (2/3)*(1/(2*R_robot))*(v1 + v2 + v3)
    return Xd, Yd, pd

# PERFIL TRAPEZOIDAL

def trapezoid(t_total, v_cruise, t_ramp, dt):
    # Genera un perfil de velocidad trapezoidal: rampa subida, crucero, rampa bajada.
    # Evita discontinuidades que causarian picos de corriente en los motores.
    n   = int(round(t_total / dt))
    n_r = int(round(min(t_ramp, t_total/2) / dt))
    v   = np.zeros(n)
    for i in range(n):
        if   i < n_r:      v[i] = v_cruise * i / n_r
        elif i >= n - n_r: v[i] = v_cruise * (n - i) / n_r
        else:              v[i] = v_cruise
    return v

# MISION
def build_mission():
    return [
        dict(label='Fase 1: Recta 2 m (+X)',           Vl=v0, om=0,       t=t_recta, color='royalblue'),
        dict(label='Fase 2: Semicirculo R=0.5 m (izq)', Vl=v0, om=+omega0, t=t_curva, color='darkorange'),
        dict(label='Fase 3: Recta 2 m (+X)',           Vl=v0, om=0,       t=t_recta, color='seagreen'),
        dict(label='Fase 4: Semicirculo R=0.5 m (der)', Vl=v0, om=-omega0, t=t_curva, color='firebrick'),
    ]

# SIMULACION

def simulate():
    segs     = build_mission()
    X, Y, phi = 0., 0., 0.
    rec      = {k: [] for k in ['X','Y','phi','v1','v2','v3','Xd','Yd','pd','t','sid']}
    tg       = 0.

    for si, seg in enumerate(segs):
        n   = int(round(seg['t'] / dt))
        pvl = trapezoid(seg['t'], seg['Vl'], T_ramp, dt)
        pom = trapezoid(seg['t'], seg['om'], T_ramp, dt)

        for i in range(n):
            vl, om = pvl[i], pom[i]

            # Velocidad deseada del chasis en marco global
            Xd_des = vl * np.cos(phi)
            Yd_des = vl * np.sin(phi)
            pd_des = om

            # Velocidades de ruedas por cinematica inversa
            vw = cinematica_inversa(Xd_des, Yd_des, pd_des, phi)

            # Verificacion por cinematica directa
            Xdr, Ydr, pdr = cinematica_directa(vw, phi)

            # Integracion Euler
            X  += Xdr * dt
            Y  += Ydr * dt
            phi = ((phi + pdr * dt) + np.pi) % (2*np.pi) - np.pi
            tg += dt

            for k, val in zip(
                ['X','Y','phi','v1','v2','v3','Xd','Yd','pd','t','sid'],
                [X, Y, phi, vw[0], vw[1], vw[2], Xdr, Ydr, pdr, tg, si]
            ):
                rec[k].append(val)

    return {k: np.array(v) for k, v in rec.items()}, segs

# DIBUJO DEL ROBOT

def draw_robot(ax, x, y, phi, rb=0.10):
    # Cuerpo circular + 3 ruedas coloreadas + flecha de heading
    arts  = []
    body  = plt.Circle((x, y), rb, color='steelblue', alpha=0.75, zorder=4)
    ax.add_patch(body)
    arts.append(body)

    wc = ['royalblue', 'darkorange', 'seagreen']
    for off, col in zip(offsets, wc):
        ang = phi + off
        wx  = x + rb * np.cos(ang)
        wy  = y + rb * np.sin(ang)
        rect = mpatches.Rectangle(
            (wx - 0.013, wy - 0.032), 0.026, 0.064,
            angle=np.degrees(ang + np.pi/2),
            rotation_point=(wx, wy),
            fc=col, ec='white', lw=0.6, zorder=5)
        ax.add_patch(rect)
        arts.append(rect)

    arr = ax.annotate('',
        xy=(x + rb*1.1*np.cos(phi), y + rb*1.1*np.sin(phi)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle='->', color='white', lw=2), zorder=6)
    arts.append(arr)
    return arts


def plot_static(h, segs):
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    ax_xy  = fig.add_subplot(gs[:, 0])
    ax_vw  = fig.add_subplot(gs[0, 1])
    ax_vg  = fig.add_subplot(gs[1, 1])
    ax_phi = fig.add_subplot(gs[2, 1])

    t, sid = h['t'], h['sid']

    for si, seg in enumerate(segs):
        m = sid == si
        ax_xy.plot(h['X'][m], h['Y'][m], color=seg['color'], lw=2.5)

    step_r = max(1, len(h['X']) // 28)
    for i in range(0, len(h['X']), step_r):
        draw_robot(ax_xy, h['X'][i], h['Y'][i], h['phi'][i])

    ax_xy.plot(h['X'][0],  h['Y'][0],  'go', ms=11, zorder=10)
    ax_xy.plot(h['X'][-1], h['Y'][-1], 'rs', ms=11, zorder=10)

    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('$X_g$ [m]', fontsize=12)
    ax_xy.set_ylabel('$Y_g$ [m]', fontsize=12)
    ax_xy.set_title('Trayectoria en el plano $X_g$-$Y_g$\n'
                    r'($R_{robot}=0.25$ m,  $v_0=0.2$ m/s)', fontsize=12)

    legend_handles = (
        [Line2D([0],[0], color='g', marker='o', ms=9, lw=0, label='Inicio'),
         Line2D([0],[0], color='r', marker='s', ms=9, lw=0, label='Fin')] +
        [mpatches.Patch(color=s['color'], label=s['label']) for s in segs]
    )
    ax_xy.legend(handles=legend_handles, fontsize=9, loc='upper right', framealpha=0.90)

    for ax_ in [ax_vw, ax_vg, ax_phi]:
        for si, seg in enumerate(segs):
            m = sid == si
            if m.any():
                ax_.axvspan(t[m][0], t[m][-1], alpha=0.08, color=seg['color'])

    ax_vw.plot(t, h['v1'], color='royalblue',  lw=1.8, label='$v_1$')
    ax_vw.plot(t, h['v2'], color='darkorange', lw=1.8, label='$v_2$')
    ax_vw.plot(t, h['v3'], color='seagreen',   lw=1.8, label='$v_3$')
    ax_vw.axhline(0, color='k', lw=0.7, ls='--')
    ax_vw.set_ylabel('Vel. rueda [m/s]', fontsize=10)
    ax_vw.set_title('Velocidades de ruedas $v_1, v_2, v_3$', fontsize=10)
    ax_vw.legend(fontsize=10)
    ax_vw.grid(True, alpha=0.3)

    ax_vg.plot(t, h['Xd'], color='royalblue',  lw=1.8, label=r'$\dot{X}_g$')
    ax_vg.plot(t, h['Yd'], color='darkorange', lw=1.8, label=r'$\dot{Y}_g$')
    ax_vg.plot(t, h['pd'], color='purple',     lw=1.8, label=r'$\dot{\phi}$')
    ax_vg.axhline(0, color='k', lw=0.7, ls='--')
    ax_vg.set_ylabel('Vel. global', fontsize=10)
    ax_vg.set_title(r'Velocidades globales $\dot{X}_g,\dot{Y}_g,\dot{\phi}$', fontsize=10)
    ax_vg.legend(fontsize=10)
    ax_vg.grid(True, alpha=0.3)

    ax_phi.plot(t, np.degrees(h['phi']), color='purple', lw=2.0)
    ax_phi.set_xlabel('Tiempo [s]', fontsize=11)
    ax_phi.set_ylabel(r'$\phi$ [°]', fontsize=11)
    ax_phi.set_title(r'Orientacion $\phi(t)$', fontsize=10)
    ax_phi.grid(True, alpha=0.3)

    fig.suptitle('Guia 1 P2 Novoa IPD482 — Robot Omnidireccional 3 Ruedas Suecas',
                 fontsize=13, fontweight='bold')
    fig.savefig('p2_static.png', dpi=150, bbox_inches='tight')
    print('OK  p2_static.png')
    plt.close(fig)

def make_gif(h, segs):
    SKIP = 10
    FPS  = 24
    idx  = np.arange(0, len(h['X']), SKIP)
    t, sid = h['t'], h['sid']
    pad    = 0.45

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    ax_xy  = fig.add_subplot(gs[:, 0])
    ax_vw  = fig.add_subplot(gs[0, 1])
    ax_vg  = fig.add_subplot(gs[1, 1])
    ax_phi = fig.add_subplot(gs[2, 1])

    ax_xy.set_xlim(h['X'].min()-pad, h['X'].max()+pad)
    ax_xy.set_ylim(h['Y'].min()-pad, h['Y'].max()+pad)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.25)
    ax_xy.set_xlabel('$X_g$ [m]')
    ax_xy.set_ylabel('$Y_g$ [m]')
    ax_xy.set_title('Trayectoria + Orientacion del robot', fontsize=11)
    ax_xy.plot(h['X'], h['Y'], color='#d0d0d0', lw=1.2, zorder=1)
    ax_xy.plot(h['X'][0], h['Y'][0], 'go', ms=9, zorder=10)

    legend_handles = (
        [Line2D([0],[0], color='g', marker='o', ms=8, lw=0, label='Inicio')] +
        [mpatches.Patch(color=s['color'], label=s['label']) for s in segs]
    )
    ax_xy.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.90)

    trail,    = ax_xy.plot([], [], 'steelblue', lw=2.0, zorder=2)
    time_txt   = ax_xy.text(0.02, 0.96, '', transform=ax_xy.transAxes,
                            fontsize=10, va='top',
                            bbox=dict(fc='white', alpha=0.7, ec='none'))

    for ax_ in [ax_vw, ax_vg, ax_phi]:
        for si, seg in enumerate(segs):
            m = sid == si
            if m.any():
                ax_.axvspan(t[m][0], t[m][-1], alpha=0.08, color=seg['color'])
        ax_.set_xlim(0, t[-1])
        ax_.grid(True, alpha=0.25)

    for ax_, ys, cols in [
        (ax_vw,  [h['v1'], h['v2'], h['v3']],
                 ['royalblue','darkorange','seagreen']),
        (ax_vg,  [h['Xd'], h['Yd'], h['pd']],
                 ['royalblue','darkorange','purple']),
        (ax_phi, [np.degrees(h['phi'])], ['purple']),
    ]:
        for y_, col in zip(ys, cols):
            ax_.plot(t, y_, color=col, lw=0.6, alpha=0.20)

    ax_vw.set_ylabel('Vel. rueda [m/s]', fontsize=9)
    ax_vw.set_title('Velocidades de ruedas $v_1,v_2,v_3$', fontsize=9)
    ax_vg.set_ylabel('Vel. global', fontsize=9)
    ax_vg.set_title(r'Velocidades globales', fontsize=9)
    ax_phi.set_ylabel(r'$\phi$ [°]', fontsize=9)
    ax_phi.set_xlabel('Tiempo [s]', fontsize=9)
    ax_phi.set_title(r'Orientacion $\phi(t)$', fontsize=9)

    lv1, = ax_vw.plot([], [], color='royalblue',  lw=1.8, label='$v_1$')
    lv2, = ax_vw.plot([], [], color='darkorange', lw=1.8, label='$v_2$')
    lv3, = ax_vw.plot([], [], color='seagreen',   lw=1.8, label='$v_3$')
    ax_vw.legend(fontsize=8, loc='upper right')

    lXd, = ax_vg.plot([], [], color='royalblue',  lw=1.8, label=r'$\dot{X}_g$')
    lYd, = ax_vg.plot([], [], color='darkorange', lw=1.8, label=r'$\dot{Y}_g$')
    lpd, = ax_vg.plot([], [], color='purple',     lw=1.8, label=r'$\dot{\phi}$')
    ax_vg.legend(fontsize=8, loc='upper right')

    lphi, = ax_phi.plot([], [], color='purple', lw=2.0)

    vlines = [ax_.axvline(0, color='red', lw=1.3, ls='--', alpha=0.8)
              for ax_ in [ax_vw, ax_vg, ax_phi]]

    fig.suptitle('Guia 1 P2 Novoa IPD482 — Robot Omnidireccional 3 Ruedas Suecas',
                 fontsize=12, fontweight='bold')

    robot_arts = []

    def init():
        trail.set_data([], [])
        for l in [lv1,lv2,lv3,lXd,lYd,lpd,lphi]:
            l.set_data([], [])
        return [trail, time_txt] + [lv1,lv2,lv3,lXd,lYd,lpd,lphi] + vlines

    def update(frame):
        nonlocal robot_arts
        for a in robot_arts:
            try: a.remove()
            except: pass
        robot_arts = []

        i = idx[frame]
        trail.set_data(h['X'][:i+1], h['Y'][:i+1])
        robot_arts = draw_robot(ax_xy, h['X'][i], h['Y'][i], h['phi'][i])
        time_txt.set_text(f"t = {t[i]:.2f} s   phi = {np.degrees(h['phi'][i]):.1f} deg")

        ti = t[:i+1]
        lv1.set_data(ti, h['v1'][:i+1])
        lv2.set_data(ti, h['v2'][:i+1])
        lv3.set_data(ti, h['v3'][:i+1])
        lXd.set_data(ti, h['Xd'][:i+1])
        lYd.set_data(ti, h['Yd'][:i+1])
        lpd.set_data(ti, h['pd'][:i+1])
        lphi.set_data(ti, np.degrees(h['phi'][:i+1]))

        for vl in vlines:
            vl.set_xdata([t[i], t[i]])

        return [trail, time_txt] + robot_arts + [lv1,lv2,lv3,lXd,lYd,lpd,lphi] + vlines

    for ax_, arr in [
        (ax_vw,  np.concatenate([h['v1'],h['v2'],h['v3']])),
        (ax_vg,  np.concatenate([h['Xd'],h['Yd'],h['pd']])),
        (ax_phi, np.degrees(h['phi'])),
    ]:
        margin = (arr.max() - arr.min()) * 0.15 + 0.02
        ax_.set_ylim(arr.min() - margin, arr.max() + margin)

    anim = FuncAnimation(fig, update, frames=len(idx),
                         init_func=init, interval=int(1000/FPS), blit=False)
    anim.save('p2_realtime.gif', writer='pillow', fps=FPS, dpi=100)
    print('OK  p2_realtime.gif')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    h, segs = simulate()
    print(f'Pasos: {len(h["t"])}   t_total: {h["t"][-1]:.2f} s')
    print(f'Pose final: ({h["X"][-1]:.3f}, {h["Y"][-1]:.3f}) m   '
          f'phi = {np.degrees(h["phi"][-1]):.2f} deg')
    plot_static(h, segs)
    make_gif(h, segs)
    generate_notebook(h, segs)
