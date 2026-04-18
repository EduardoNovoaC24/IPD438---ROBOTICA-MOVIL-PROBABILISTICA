# Guia 1 P3 Novoa IPD482
# Sistema Car-like + Trailer Pasivo (TTWR — off-axle hitching)
# IPD-482 Robotica Movil Probabilistica — UTFSM 2026

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import json

# =============================================================================
# 1. PARAMETROS DEL TRACTOR  (Clearpath Husky A200 — biciclo traccion trasera)
# =============================================================================
d        = 0.50   # Distancia entre ejes del tractor [m]  (nomenclatura Jorquera p.12)
W_trac   = 0.40   # Ancho visual del tractor [m]
L_trac   = 0.65   # Largo visual del tractor [m]

# =============================================================================
# 2. PARAMETROS DEL ENGANCHE Y TRAILER  (Gorilla Cart GOR1001 — mitad)
# =============================================================================
d_h      = 0.10   # Offset del enganche detras del eje trasero del tractor [m]
L1       = 0.60   # Distancia enganche -> eje trasero del trailer [m]
W_trail  = 0.50   # Ancho visual del trailer [m]
L_trail  = 0.80   # Largo visual del trailer [m]

# =============================================================================
# 3. PARAMETROS DE SIMULACION Y TRAYECTORIA
# =============================================================================
v0       = 0.30   # Velocidad lineal de avance [m/s]
T_ramp   = 0.60   # Tiempo de rampa del perfil trapezoidal [s]
dt       = 0.02   # Paso de tiempo de simulacion [s]
d_recta  = 3.0    # Distancia de la fase recta [m]
R_curva  = 1.0    # Radio de la circunferencia [m]

# =============================================================================
# 4. PERFIL TRAPEZOIDAL
# =============================================================================

def trapezoid(t_total, v_max, t_ramp, dt):
    # Perfil de velocidad continuo: evita discontinuidades en las entradas.
    t = np.arange(0, t_total + dt, dt)
    v = np.zeros_like(t)
    for i, ti in enumerate(t):
        if   ti < t_ramp:             v[i] = v_max * ti / t_ramp
        elif ti < t_total - t_ramp:   v[i] = v_max
        elif ti < t_total:            v[i] = v_max * (t_total - ti) / t_ramp
    return t, v

# =============================================================================
# 5. MISION
# =============================================================================

def build_mission():
    # Fase 1: recta — alpha = 0, sin rotacion del tractor
    # Fase 2: circunferencia — alpha obtenido invirtiendo phi0_dot = (v/d)*tan(alpha)
    alpha_c = np.arctan(d / R_curva)
    t1 = d_recta / v0 + 2 * T_ramp
    t2 = 2 * np.pi * R_curva / v0 + 2 * T_ramp
    return [
        {'type': 'straight', 't_total': t1, 'alpha': 0.0,     'color': 'royalblue', 'label': 'Fase 1: Recta 3 m'},
        {'type': 'circle',   't_total': t2, 'alpha': alpha_c, 'color': 'darkorange', 'label': f'Fase 2: Circunferencia R={R_curva} m'},
    ]

# =============================================================================
# 6. CINEMATICA DIRECTA (un paso Euler)
# =============================================================================

def step_forward(state, v, alpha, dt):
    # Tractor — biciclo traccion trasera (Jorquera p.12):
    #   Xdot    = v * cos(phi0)
    #   Ydot    = v * sin(phi0)
    #   phi0dot = (v/d) * tan(alpha)
    #
    # Trailer — off-axle hitching (Guevara et al. ec.2 / Michalek ec.4):
    #   La restriccion de no deslizamiento del trailer impone:
    #   v_H     = v - d_h * omega0 * sin(beta)   velocidad en el punto de enganche
    #   phi1dot = (v_H / L1)*sin(beta) - (d_h/L1)*cos(beta)*omega0
    X, Y, phi0, phi1 = state

    omega0  = (v / d) * np.tan(alpha)
    beta    = phi0 - phi1
    v_H     = v - d_h * omega0 * np.sin(beta)
    phi1dot = (v_H / L1) * np.sin(beta) - (d_h / L1) * np.cos(beta) * omega0

    return np.array([
        X    + v     * np.cos(phi0) * dt,
        Y    + v     * np.sin(phi0) * dt,
        phi0 + omega0               * dt,
        phi1 + phi1dot              * dt,
    ])


def trailer_axle_pos(state):
    # Posicion geometrica del enganche H y del eje trasero del trailer T
    X, Y, phi0, phi1 = state
    Hx = X  - d_h * np.cos(phi0)
    Hy = Y  - d_h * np.sin(phi0)
    Tx = Hx - L1  * np.cos(phi1)
    Ty = Hy - L1  * np.sin(phi1)
    return Hx, Hy, Tx, Ty

# =============================================================================
# 7. SIMULACION
# =============================================================================

def simulate():
    mission  = build_mission()
    state    = np.zeros(4)
    states   = [state.copy()]
    controls = []
    times    = [0.0]
    sid      = [0]
    t_offset = 0.0

    for si, seg in enumerate(mission):
        t_seg, v_seg = trapezoid(seg['t_total'], v0, T_ramp, dt)
        alpha_seg    = seg['alpha']
        for i in range(len(t_seg) - 1):
            state = step_forward(state, v_seg[i], alpha_seg, dt)
            states.append(state.copy())
            controls.append([v_seg[i], alpha_seg])
            times.append(t_offset + t_seg[i + 1])
            sid.append(si)
        t_offset += t_seg[-1]

    controls.append(controls[-1])

    return (np.array(states), np.array(controls),
            np.array(times),  np.array(sid))

# =============================================================================
# 8. DIBUJO DEL VEHICULO
# =============================================================================

def draw_vehicle(ax, state, alpha_val=1.0):
    X, Y, phi0, phi1 = state
    Hx, Hy, Tx, Ty   = trailer_axle_pos(state)

    def make_rect(cx, cy, angle, length, width, fc):
        ca, sa = np.cos(angle), np.sin(angle)
        dx, dy = length/2, width/2
        corners = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])
        rot     = np.array([[ca,-sa],[sa,ca]])
        pts     = (rot @ corners.T).T + np.array([cx, cy])
        return patches.Polygon(pts, closed=True, facecolor=fc,
                               edgecolor='#333333', linewidth=1.2,
                               alpha=alpha_val, zorder=4)

    # Trailer primero (zorder menor para quedar detras del tractor)
    cx_tr = (Hx + Tx) / 2
    cy_tr = (Hy + Ty) / 2
    ax.add_patch(make_rect(cx_tr, cy_tr, phi1, L_trail, W_trail, '#E65100'))

    # Tractor
    cx_t = X - (d/2) * np.cos(phi0)
    cy_t = Y - (d/2) * np.sin(phi0)
    ax.add_patch(make_rect(cx_t, cy_t, phi0, L_trac, W_trac, '#1565C0'))

    # Flecha heading del tractor
    ax.annotate('', xy=(X + 0.20*np.cos(phi0), Y + 0.20*np.sin(phi0)),
                xytext=(X, Y),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.8), zorder=6)

    # Barra de enganche
    ax.plot([X, Hx], [Y, Hy], '-', color='#555555', lw=2.0, zorder=5)
    ax.plot([Hx, Tx], [Hy, Ty], '-', color='#888888', lw=1.8, zorder=4)
    ax.plot(Hx, Hy, 'o', color='#FFC107', ms=6, zorder=7,
            markeredgecolor='#555', markeredgewidth=0.8)

# =============================================================================
# 9. FIGURA ESTATICA
# Layout: GridSpec(3,2) — XY izquierda (3 filas), 3 graficas apiladas derecha
# =============================================================================

def plot_static(states, controls, times, sid, segs):
    Tx_arr, Ty_arr = [], []
    for s in states:
        _, _, Tx, Ty = trailer_axle_pos(s)
        Tx_arr.append(Tx); Ty_arr.append(Ty)
    betas = np.degrees(states[:, 2] - states[:, 3])

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             left=0.06, right=0.97, top=0.93, bottom=0.08,
                             hspace=0.50, wspace=0.38)

    ax_xy = fig.add_subplot(gs[:, 0])
    ax1   = fig.add_subplot(gs[0, 1])
    ax2   = fig.add_subplot(gs[1, 1])
    ax3   = fig.add_subplot(gs[2, 1])

    # Trayectorias coloreadas por fase
    for si, seg in enumerate(segs):
        m = sid == si
        ax_xy.plot(states[m, 0], states[m, 1],
                   color=seg['color'], lw=2.0, label=f'Tractor — {seg["label"]}')
        ax_xy.plot(np.array(Tx_arr)[m], np.array(Ty_arr)[m],
                   color=seg['color'], lw=1.8, ls='--')

    n = len(states)
    for idx, av in [(0, 0.40), (n//3, 0.55), (2*n//3, 0.70), (n-1, 1.0)]:
        draw_vehicle(ax_xy, states[idx], alpha_val=av)

    ax_xy.plot(states[0, 0], states[0, 1], 'go', ms=10, zorder=10, label='Inicio')
    ax_xy.plot(states[-1,0], states[-1,1], 'rs', ms=10, zorder=10, label='Fin')

    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('X [m]', fontsize=11)
    ax_xy.set_ylabel('Y [m]', fontsize=11)
    ax_xy.set_title('Trayectoria XY — Tractor (azul) + Trailer (naranja)', fontsize=11)

    legend_handles = (
        [Line2D([0],[0], color='g', marker='o', ms=9, lw=0, label='Inicio'),
         Line2D([0],[0], color='r', marker='s', ms=9, lw=0, label='Fin'),
         Line2D([0],[0], color='#1565C0', lw=2.5, label='Tractor'),
         Line2D([0],[0], color='#E65100', lw=2.5, ls='--', label='Trailer')] +
        [patches.Patch(color=s['color'], label=s['label']) for s in segs]
    )
    ax_xy.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.90)

    # Bandas de fase en graficas laterales
    for ax_ in [ax1, ax2, ax3]:
        for si, seg in enumerate(segs):
            m = sid == si
            if m.any():
                ax_.axvspan(times[m][0], times[m][-1], alpha=0.08, color=seg['color'])
        ax_.set_facecolor('#F8F9FA')
        ax_.grid(True, color='#DDDDDD', lw=0.6, alpha=0.8)
        ax_.tick_params(colors='#212121', labelsize=8)
        for sp in ax_.spines.values(): sp.set_edgecolor('#AAAAAA')

    ax1.plot(times, controls[:, 0], color='#1565C0', lw=1.6, label='$v$ [m/s]')
    ax1r = ax1.twinx()
    ax1r.plot(times, np.degrees(controls[:, 1]), color='#C62828', lw=1.6, ls='--', label='$\\alpha$ [deg]')
    ax1.set_ylabel('$v$ [m/s]', color='#1565C0', fontsize=9)
    ax1r.set_ylabel('$\\alpha$ [deg]', color='#C62828', fontsize=9)
    ax1r.tick_params(colors='#212121', labelsize=8)
    ax1r.spines['right'].set_edgecolor('#AAAAAA')
    ax1.set_title('Velocidad lineal $v$ y steering $\\alpha$', fontsize=10, fontweight='bold')
    lines = ax1.get_lines() + ax1r.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='upper right')

    ax2.plot(times, np.degrees(states[:, 2]), color='#1565C0', lw=1.6, label='$\\phi_0$ tractor')
    ax2.plot(times, np.degrees(states[:, 3]), color='#E65100', lw=1.6, ls='--', label='$\\phi_1$ trailer')
    ax2.set_ylabel('[deg]', fontsize=9)
    ax2.set_title('Orientaciones $\\phi_0$ y $\\phi_1$', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')

    ax3.plot(times, betas, color='#2E7D32', lw=1.6, label='$\\beta = \\phi_0 - \\phi_1$')
    ax3.axhline(0, color='#AAAAAA', lw=0.9, ls=':')
    idx_max = np.argmax(np.abs(betas))
    ax3.annotate(f'$\\beta_{{max}}={betas[idx_max]:.1f}$ deg',
                 xy=(times[idx_max], betas[idx_max]),
                 xytext=(times[idx_max]-3, betas[idx_max]+4),
                 fontsize=8, color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))
    ax3.set_ylabel('[deg]', fontsize=9)
    ax3.set_xlabel('Tiempo [s]', fontsize=9)
    ax3.set_title('Angulo de articulacion $\\beta$', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper left')

    fig.suptitle('Guia 1 P3 Novoa IPD482 — TTWR off-axle  |  '
                 'Tractor: Husky A200  |  Trailer: Gorilla Cart GOR1001 (1/2)',
                 fontsize=10, fontweight='bold')

    fig.savefig('/mnt/user-data/outputs/p3_static.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('OK  p3_static.png')
    plt.close(fig)

# =============================================================================
# 10. GIF EN TIEMPO REAL
# =============================================================================

def make_gif(states, controls, times, sid, segs):
    N      = len(states)
    stride = max(1, N // 300)

    Tx_arr, Ty_arr = [], []
    for s in states:
        _, _, Tx, Ty = trailer_axle_pos(s)
        Tx_arr.append(Tx); Ty_arr.append(Ty)
    betas = np.degrees(states[:, 2] - states[:, 3])

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             left=0.06, right=0.97, top=0.92, bottom=0.08,
                             hspace=0.50, wspace=0.38)

    ax_xy = fig.add_subplot(gs[:, 0])
    ax1   = fig.add_subplot(gs[0, 1]); ax1r = ax1.twinx()
    ax2   = fig.add_subplot(gs[1, 1])
    ax3   = fig.add_subplot(gs[2, 1])

    margin = 1.2
    ax_xy.set_xlim(min(list(states[:,0])+Tx_arr)-margin, max(list(states[:,0])+Tx_arr)+margin)
    ax_xy.set_ylim(min(list(states[:,1])+Ty_arr)-margin, max(list(states[:,1])+Ty_arr)+margin)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Trayectoria XY', fontsize=11, fontweight='bold')

    # Trazas de fondo tenues
    for si, seg in enumerate(segs):
        m = sid == si
        ax_xy.plot(states[m, 0], states[m, 1], color=seg['color'], lw=0.8, alpha=0.15)
        ax_xy.plot(np.array(Tx_arr)[m], np.array(Ty_arr)[m],
                   color=seg['color'], lw=0.8, alpha=0.15, ls='--')

    legend_handles = (
        [Line2D([0],[0], color='#1565C0', lw=2.5, label='Tractor'),
         Line2D([0],[0], color='#E65100', lw=2.5, ls='--', label='Trailer')] +
        [patches.Patch(color=s['color'], label=s['label']) for s in segs]
    )
    ax_xy.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.90)

    lt,  = ax_xy.plot([], [], color='#1565C0', lw=1.8)
    ltr, = ax_xy.plot([], [], color='#E65100', lw=1.8, ls='--')

    for ax_ in [ax1, ax2, ax3]:
        for si, seg in enumerate(segs):
            m = sid == si
            if m.any():
                ax_.axvspan(times[m][0], times[m][-1], alpha=0.08, color=seg['color'])
        ax_.set_facecolor('#F8F9FA')
        ax_.grid(True, color='#DDDDDD', lw=0.6, alpha=0.8)
        ax_.tick_params(colors='#212121', labelsize=8)
        for sp in ax_.spines.values(): sp.set_edgecolor('#AAAAAA')

    ax1.set_xlim(times[0], times[-1]); ax1.set_ylim(-0.05, v0*1.2)
    ax1r.set_ylim(-5, np.degrees(np.arctan(d/R_curva))*1.6+5)
    ax1.set_ylabel('$v$ [m/s]', color='#1565C0', fontsize=9)
    ax1r.set_ylabel('$\\alpha$ [deg]', color='#C62828', fontsize=9)
    ax1r.tick_params(colors='#212121', labelsize=8)
    ax1r.spines['right'].set_edgecolor('#AAAAAA')
    ax1.set_title('$v$ y $\\alpha$', fontsize=10, fontweight='bold')

    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(np.degrees(states[:,2:4]).min()-10, np.degrees(states[:,2:4]).max()+10)
    ax2.set_ylabel('[deg]', fontsize=9)
    ax2.set_title('Orientaciones $\\phi_0$ y $\\phi_1$', fontsize=10, fontweight='bold')

    ax3.set_xlim(times[0], times[-1])
    ax3.set_ylim(betas.min()-5, betas.max()+5)
    ax3.axhline(0, color='#AAAAAA', lw=0.8, ls=':')
    ax3.set_ylabel('[deg]', fontsize=9)
    ax3.set_xlabel('Tiempo [s]', fontsize=9)
    ax3.set_title('Articulacion $\\beta$', fontsize=10, fontweight='bold')

    lv,  = ax1.plot([], [], color='#1565C0', lw=1.6, label='$v$')
    la,  = ax1r.plot([], [], color='#C62828', lw=1.6, ls='--', label='$\\alpha$')
    lp0, = ax2.plot([], [], color='#1565C0', lw=1.6, label='$\\phi_0$')
    lp1, = ax2.plot([], [], color='#E65100', lw=1.6, ls='--', label='$\\phi_1$')
    lb,  = ax3.plot([], [], color='#2E7D32', lw=1.6, label='$\\beta$')

    ax1.legend(fontsize=8, loc='upper right')
    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)

    cur1 = ax1.axvline(0, color='red', lw=1.0, alpha=0.7)
    cur2 = ax2.axvline(0, color='red', lw=1.0, alpha=0.7)
    cur3 = ax3.axvline(0, color='red', lw=1.0, alpha=0.7)

    fig.suptitle('Guia 1 P3 Novoa IPD482 — TTWR off-axle (animacion)',
                 fontsize=10, fontweight='bold')

    dyn_patches = []

    def init():
        lt.set_data([], []); ltr.set_data([], [])
        for l in [lv, la, lp0, lp1, lb]: l.set_data([], [])
        return lt, ltr, lv, la, lp0, lp1, lb

    def update(frame):
        k = min(frame * stride, N - 1)
        for p in dyn_patches:
            try: p.remove()
            except: pass
        dyn_patches.clear()

        lt.set_data(states[:k+1, 0], states[:k+1, 1])
        ltr.set_data(Tx_arr[:k+1],   Ty_arr[:k+1])

        # Robot actual
        s = states[k]
        X, Y, phi0, phi1 = s
        Hx, Hy, Tx_k, Ty_k = trailer_axle_pos(s)

        def mp(cx, cy, ang, l, w, fc):
            ca, sa = np.cos(ang), np.sin(ang)
            dx, dy = l/2, w/2
            c = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])
            r = np.array([[ca,-sa],[sa,ca]])
            pts = (r @ c.T).T + np.array([cx, cy])
            return patches.Polygon(pts, closed=True, facecolor=fc,
                                   edgecolor='#333', lw=1.2, zorder=4)

        cx_tr = (Hx+Tx_k)/2; cy_tr = (Hy+Ty_k)/2
        p2 = mp(cx_tr, cy_tr, phi1, L_trail, W_trail, '#E65100')
        cx_t = X-(d/2)*np.cos(phi0); cy_t = Y-(d/2)*np.sin(phi0)
        p1 = mp(cx_t, cy_t, phi0, L_trac, W_trac, '#1565C0')
        ax_xy.add_patch(p2); ax_xy.add_patch(p1)
        bar, = ax_xy.plot([X,Hx,Tx_k],[Y,Hy,Ty_k], color='#555', lw=1.8, zorder=3)
        dot, = ax_xy.plot([Hx],[Hy], 'o', color='#FFC107', ms=6, zorder=6)
        dyn_patches.extend([p1, p2, bar, dot])

        t_k = times[:k+1]
        lv.set_data(t_k, controls[:k+1, 0])
        la.set_data(t_k, np.degrees(controls[:k+1, 1]))
        lp0.set_data(t_k, np.degrees(states[:k+1, 2]))
        lp1.set_data(t_k, np.degrees(states[:k+1, 3]))
        lb.set_data(t_k, betas[:k+1])

        for cur in [cur1, cur2, cur3]:
            cur.set_xdata([times[k]])

        return lt, ltr, lv, la, lp0, lp1, lb

    ani = animation.FuncAnimation(fig, update, frames=N//stride,
                                  init_func=init, interval=40, blit=False)
    ani.save('/mnt/user-data/outputs/p3_realtime.gif', writer='pillow', fps=25, dpi=100)
    print('OK  p3_realtime.gif')
    plt.close(fig)

# =============================================================================
# 11. NOTEBOOK INTERACTIVO
# =============================================================================

def generate_notebook(states, controls, times, sid, segs):
    Tx_arr, Ty_arr = [], []
    for s in states:
        _, _, Tx, Ty = trailer_axle_pos(s)
        Tx_arr.append(Tx); Ty_arr.append(Ty)
    betas = np.degrees(states[:, 2] - states[:, 3])

    nb_code = f"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import ipywidgets as widgets
from IPython.display import display

d={d}; d_h={d_h}; L1={L1}; L_trail={L_trail}; W_trail={W_trail}; L_trac={L_trac}; W_trac={W_trac}
states  = np.{repr(states)}
controls= np.{repr(controls)}
times   = np.{repr(times)}
sid     = np.{repr(sid)}
betas   = np.{repr(betas)}
Tx_arr  = {Tx_arr}
Ty_arr  = {Ty_arr}
segs    = {segs}
N       = len(states)

def trailer_axle_pos(state):
    X,Y,phi0,phi1=state
    Hx=X-d_h*np.cos(phi0); Hy=Y-d_h*np.sin(phi0)
    Tx=Hx-L1*np.cos(phi1); Ty=Hy-L1*np.sin(phi1)
    return Hx,Hy,Tx,Ty

def draw_frame(k):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    gs=gridspec.GridSpec(3,2,figure=fig,left=0.06,right=0.97,top=0.92,bottom=0.08,hspace=0.50,wspace=0.38)
    ax_xy=fig.add_subplot(gs[:,0]); ax1=fig.add_subplot(gs[0,1]); ax1r=ax1.twinx()
    ax2=fig.add_subplot(gs[1,1]); ax3=fig.add_subplot(gs[2,1])

    for si,seg in enumerate(segs):
        m=sid==si
        ax_xy.plot(states[m,0],states[m,1],color=seg['color'],lw=0.8,alpha=0.2)
    ax_xy.plot(states[:k+1,0],states[:k+1,1],color='#1565C0',lw=1.8)
    ax_xy.plot(Tx_arr[:k+1],Ty_arr[:k+1],color='#E65100',lw=1.8,ls='--')

    s=states[k]; X,Y,phi0,phi1=s; Hx,Hy,Tx_k,Ty_k=trailer_axle_pos(s)
    def mp(cx,cy,ang,l,w,fc):
        ca,sa=np.cos(ang),np.sin(ang); dx,dy=l/2,w/2
        c=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])
        r=np.array([[ca,-sa],[sa,ca]])
        pts=(r@c.T).T+np.array([cx,cy])
        return patches.Polygon(pts,closed=True,facecolor=fc,edgecolor='#333',lw=1.2,zorder=4)
    cx_tr=(Hx+Tx_k)/2; cy_tr=(Hy+Ty_k)/2
    ax_xy.add_patch(mp(cx_tr,cy_tr,phi1,L_trail,W_trail,'#E65100'))
    cx_t=X-(d/2)*np.cos(phi0); cy_t=Y-(d/2)*np.sin(phi0)
    ax_xy.add_patch(mp(cx_t,cy_t,phi0,L_trac,W_trac,'#1565C0'))
    ax_xy.plot([X,Hx,Tx_k],[Y,Hy,Ty_k],color='#555',lw=1.8,zorder=3)
    ax_xy.plot(Hx,Hy,'o',color='#FFC107',ms=6,zorder=6)

    margin=1.2
    all_x=list(states[:,0])+Tx_arr; all_y=list(states[:,1])+Ty_arr
    ax_xy.set_xlim(min(all_x)-margin,max(all_x)+margin)
    ax_xy.set_ylim(min(all_y)-margin,max(all_y)+margin)
    ax_xy.set_aspect('equal'); ax_xy.grid(True,alpha=0.3)
    ax_xy.set_title(f't = {{times[k]:.2f}} s',fontsize=11,fontweight='bold')
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')

    for ax_ in [ax1,ax2,ax3]:
        for si,seg in enumerate(segs):
            m=sid==si
            if m.any(): ax_.axvspan(times[m][0],times[m][-1],alpha=0.08,color=seg['color'])
        ax_.set_facecolor('#F8F9FA'); ax_.grid(True,color='#DDDDDD',lw=0.6,alpha=0.8)
        ax_.set_xlim(times[0],times[-1])
        ax_.axvline(times[k],color='red',lw=1.0,alpha=0.7)

    t_k=times[:k+1]
    ax1.plot(t_k,controls[:k+1,0],color='#1565C0',lw=1.6,label='v')
    ax1r.plot(t_k,np.degrees(controls[:k+1,1]),color='#C62828',lw=1.6,ls='--',label='alpha')
    ax1.set_title('v y alpha',fontsize=10,fontweight='bold')
    ax1.set_ylabel('v [m/s]',color='#1565C0',fontsize=9)
    ax1r.set_ylabel('alpha [deg]',color='#C62828',fontsize=9)

    ax2.plot(t_k,np.degrees(states[:k+1,2]),color='#1565C0',lw=1.6,label='phi0')
    ax2.plot(t_k,np.degrees(states[:k+1,3]),color='#E65100',lw=1.6,ls='--',label='phi1')
    ax2.set_title('Orientaciones',fontsize=10,fontweight='bold')
    ax2.set_ylabel('[deg]',fontsize=9); ax2.legend(fontsize=8)

    ax3.plot(t_k,betas[:k+1],color='#2E7D32',lw=1.6,label='beta')
    ax3.axhline(0,color='#AAAAAA',lw=0.8,ls=':')
    ax3.set_title('Articulacion beta',fontsize=10,fontweight='bold')
    ax3.set_ylabel('[deg]',fontsize=9); ax3.set_xlabel('Tiempo [s]',fontsize=9)
    ax3.legend(fontsize=8)

    fig.suptitle('Guia 1 P3 Novoa IPD482 — TTWR off-axle',fontsize=10,fontweight='bold')
    plt.show(); plt.close(fig)

slider=widgets.IntSlider(value=0,min=0,max=N-1,step=max(1,N//500),
                         description='Frame:',layout=widgets.Layout(width='60%'))
play=widgets.Play(value=0,min=0,max=N-1,step=max(1,N//500),interval=40)
widgets.jslink((play,'value'),(slider,'value'))
out_w=widgets.interactive_output(draw_frame,{{'k':slider}})
display(widgets.VBox([widgets.HBox([play,slider]),out_w]))
"""

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Guia 1 P3 Novoa IPD482\n",
                "Sistema TTWR — Car-like + Trailer pasivo (off-axle hitching)\n\n",
                "Usa el slider o Play para navegar la simulacion."
            ]},
            {"cell_type": "code", "execution_count": None,
             "metadata": {}, "outputs": [], "source": [nb_code]}
        ]
    }
    with open('/mnt/user-data/outputs/p3_interactivo.ipynb', 'w') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print('OK  p3_interactivo.ipynb')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    states, controls, times, sid = simulate()
    segs = build_mission()
    print(f'Pasos: {len(states)}   t_total: {times[-1]:.2f} s')
    print(f'beta_max: {np.degrees(np.abs(states[:,2]-states[:,3])).max():.2f} deg')
    plot_static(states, controls, times, sid, segs)
    make_gif(states, controls, times, sid, segs)
    generate_notebook(states, controls, times, sid, segs)
