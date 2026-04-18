# Guia 1 P4 Novoa IPD482
# Dinamica directa del robot uniciclo — modelo Newton-Euler
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
# 1. PARAMETROS FISICOS DEL ROBOT
# =============================================================================
m    = 4.0      # Masa total [kg]
I    = 0.08     # Momento de inercia rotacional alrededor del eje vertical [kg*m^2]
r    = 0.07     # Radio de cada rueda [m]
L    = 0.28     # Distancia entre ruedas [m]
cv   = 2.6667   # Coeficiente de friccion viscosa traslacional [N*s/m]
cw   = 0.1244   # Coeficiente de friccion viscosa rotacional [N*m*s]

# =============================================================================
# 2. PARAMETROS DE SIMULACION
# =============================================================================
tau  = 0.0467   # Torque de actuacion aplicado por maniobra [N*m]
dt   = 0.01     # Paso de tiempo de simulacion [s]
T_man = 6.0     # Duracion de cada maniobra [s]

# =============================================================================
# 3. VELOCIDADES TERMINALES TEORICAS
# Provienen de las ecuaciones en estado estacionario (v_dot = 0, omega_dot = 0)
# =============================================================================
v_term   = 2 * tau / (r * cv)            # 0.500 m/s — maniobras M1 y M2
tau_v    = m / cv                         # 1.500 s   — constante de tiempo de v
tau_om   = I / cw                         # 0.645 s   — constante de tiempo de omega
om_term3 = L * tau / (2 * r * cw)        # 0.750 rad/s — maniobra M3
om_term4 = L * 2 * tau / (r * cw)        # 1.502 rad/s — maniobra M4

# =============================================================================
# 4. MANIOBRAS
# =============================================================================
MANIOBRAS = [
    {'tau_R': +tau, 'tau_L': +tau,  'color': 'royalblue',  'label': 'M1: Adelante'},
    {'tau_R': -tau, 'tau_L': -tau,  'color': 'firebrick',  'label': 'M2: Atras'},
    {'tau_R': +tau, 'tau_L':  0.0,  'color': 'seagreen',   'label': 'M3: Una rueda'},
    {'tau_R': +tau, 'tau_L': -tau,  'color': 'darkorchid', 'label': 'M4: Contrapuestas'},
]

# =============================================================================
# 5. DINAMICA  (Newton-Euler)
# =============================================================================

def dynamics(state, tau_R, tau_L):
    # Ecuaciones de movimiento en espacio de estados X = [x, y, theta, v, omega]:
    #
    # Cinematica (igual que el uniciclo de P1):
    #   x_dot     = v * cos(theta)
    #   y_dot     = v * sin(theta)
    #   theta_dot = omega
    #
    # Dinamica traslacional (2da ley de Newton en direccion longitudinal):
    #   v_dot = (tau_R + tau_L)/(m*r) - (cv/m)*v
    #   El primer termino es la fuerza neta de traccion dividida por la masa.
    #   El segundo termino es la friccion viscosa que limita la velocidad terminal.
    #
    # Dinamica rotacional (ecuacion de Euler alrededor del COM):
    #   omega_dot = L*(tau_R - tau_L)/(2*I*r) - (cw/I)*omega
    #   El par diferencial entre ruedas genera el momento de giro.
    #   La friccion viscosa rotacional limita la velocidad angular terminal.
    x, y, theta, v, omega = state
    return np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        omega,
        (tau_R + tau_L) / (m * r) - (cv / m) * v,
        L * (tau_R - tau_L) / (2 * I * r) - (cw / I) * omega,
    ])


def step_rk4(state, tau_R, tau_L, dt):
    # Integracion Runge-Kutta de orden 4.
    # Se prefiere sobre Euler porque la dinamica tiene constantes de tiempo
    # cortas (tau_om ~ 0.6 s) y RK4 garantiza mejor precision con dt = 0.01 s.
    k1 = dynamics(state,           tau_R, tau_L)
    k2 = dynamics(state + dt/2*k1, tau_R, tau_L)
    k3 = dynamics(state + dt/2*k2, tau_R, tau_L)
    k4 = dynamics(state + dt*k3,   tau_R, tau_L)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# =============================================================================
# 6. SIMULACION — 4 maniobras concatenadas
# =============================================================================

def simulate():
    # Cada maniobra parte con v=0, omega=0 para visualizar claramente
    # la respuesta transitoria desde reposo (efecto de la inercia).
    # La posicion y orientacion se heredan de la maniobra anterior.
    n_steps  = int(T_man / dt)
    t_seg    = np.arange(0, T_man + dt, dt)[:n_steps + 1]

    states_all = []
    tauR_all   = []
    tauL_all   = []
    times_all  = []
    sid_all    = []

    state    = np.zeros(5)
    t_offset = 0.0

    for mi, man in enumerate(MANIOBRAS):
        tR = man['tau_R']
        tL = man['tau_L']
        state[3] = 0.0   # reiniciar v
        state[4] = 0.0   # reiniciar omega

        for k in range(n_steps):
            states_all.append(state.copy())
            tauR_all.append(tR)
            tauL_all.append(tL)
            times_all.append(t_offset + t_seg[k])
            sid_all.append(mi)
            state = step_rk4(state, tR, tL, dt)

        states_all.append(state.copy())
        tauR_all.append(tR)
        tauL_all.append(tL)
        times_all.append(t_offset + t_seg[-1])
        sid_all.append(mi)
        t_offset += T_man

    return (np.array(states_all), np.array(tauR_all),
            np.array(tauL_all),   np.array(times_all),
            np.array(sid_all))

# =============================================================================
# 7. DIBUJO DEL ROBOT
# =============================================================================

def draw_robot(ax, x, y, theta, color, alpha=1.0):
    circle = patches.Circle((x, y), 0.05,
                             facecolor=color, edgecolor='#333333',
                             linewidth=0.9, alpha=alpha, zorder=4)
    ax.add_patch(circle)
    dx = 0.09 * np.cos(theta)
    dy = 0.09 * np.sin(theta)
    ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.4), zorder=5)

# =============================================================================
# 8. FIGURA ESTATICA
# Layout: GridSpec(3,2) — XY izquierda (3 filas), 3 graficas apiladas derecha
# Grafica 1: torques tau_R y tau_L vs tiempo
# Grafica 2: velocidad lineal v(t) con valor terminal anotado
# Grafica 3: velocidad angular omega(t) con valores terminales anotados
# =============================================================================

def plot_static(states, tauR, tauL, times, sid):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             left=0.06, right=0.97, top=0.93, bottom=0.08,
                             hspace=0.50, wspace=0.38)

    ax_xy = fig.add_subplot(gs[:, 0])
    ax1   = fig.add_subplot(gs[0, 1])
    ax2   = fig.add_subplot(gs[1, 1])
    ax3   = fig.add_subplot(gs[2, 1])

    # Trayectorias coloreadas por maniobra
    for mi, man in enumerate(MANIOBRAS):
        mk = sid == mi
        ax_xy.plot(states[mk, 0], states[mk, 1], color=man['color'], lw=2.0)

    # Snapshots del robot en puntos clave de cada maniobra
    for mi, man in enumerate(MANIOBRAS):
        mk  = np.where(sid == mi)[0]
        for idx, alp in [(mk[0], 0.35), (mk[len(mk)//2], 0.60), (mk[-1], 1.00)]:
            draw_robot(ax_xy,
                       states[idx,0], states[idx,1], states[idx,2],
                       man['color'], alpha=alp)

    ax_xy.plot(states[0, 0],  states[0, 1],  'go', ms=10, zorder=10, label='Inicio M1')
    ax_xy.plot(states[-1, 0], states[-1, 1], 'rs', ms=10, zorder=10, label='Fin M4')

    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('X [m]', fontsize=11)
    ax_xy.set_ylabel('Y [m]', fontsize=11)
    ax_xy.set_title('Trayectoria XY — 4 Maniobras Dinamicas', fontsize=11, fontweight='bold')

    legend_handles = (
        [Line2D([0],[0], color='g', marker='o', ms=9, lw=0, label='Inicio M1'),
         Line2D([0],[0], color='r', marker='s', ms=9, lw=0, label='Fin M4')] +
        [Line2D([0],[0], color=man['color'], lw=2.5, label=man['label']) for man in MANIOBRAS]
    )
    ax_xy.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.90)

    # Estilo comun de ejes laterales
    for ax_ in [ax1, ax2, ax3]:
        for mi, man in enumerate(MANIOBRAS):
            mk = sid == mi
            if mk.any():
                ax_.axvspan(times[mk][0], times[mk][-1], alpha=0.07, color=man['color'])
        ax_.set_facecolor('#F8F9FA')
        ax_.grid(True, color='#DDDDDD', lw=0.6, alpha=0.8)
        ax_.tick_params(colors='#212121', labelsize=8)
        for sp in ax_.spines.values(): sp.set_edgecolor('#AAAAAA')

    # Grafica 1 — torques
    ax1.plot(times, tauR * 1000, color='#2E7D32', lw=1.6, label='$\\tau_R$ [mN·m]')
    ax1.plot(times, tauL * 1000, color='#E65100', lw=1.6, ls='--', label='$\\tau_L$ [mN·m]')
    ax1.axhline(0, color='#AAAAAA', lw=0.8, ls=':')
    ax1.set_ylabel('Torque [mN·m]', fontsize=9)
    ax1.set_title('Torques $\\tau_R$ y $\\tau_L$', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')

    # Grafica 2 — velocidad lineal
    for mi, man in enumerate(MANIOBRAS):
        mk = sid == mi
        ax2.plot(times[mk], states[mk, 3], color=man['color'], lw=1.8, label=f'M{mi+1}')
    ax2.axhline( v_term, color=MANIOBRAS[0]['color'], lw=0.9, ls='--', alpha=0.6)
    ax2.axhline(-v_term, color=MANIOBRAS[1]['color'], lw=0.9, ls='--', alpha=0.6)

    mk0 = np.where(sid==0)[0]
    ax2.annotate(f'$v_{{term}}={v_term:.3f}$ m/s',
                 xy=(times[mk0[int(len(mk0)*0.6)]], v_term),
                 xytext=(times[mk0[int(len(mk0)*0.6)]], v_term + 0.04),
                 fontsize=7.5, color=MANIOBRAS[0]['color'])

    ax2.axhline(0, color='#AAAAAA', lw=0.7, ls=':')
    ax2.set_ylabel('$v$ [m/s]', fontsize=9)
    ax2.set_title('Velocidad lineal $v(t)$', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')

    # Grafica 3 — velocidad angular
    for mi, man in enumerate(MANIOBRAS):
        mk = sid == mi
        ax3.plot(times[mk], states[mk, 4], color=man['color'], lw=1.8, label=f'M{mi+1}')

    mk2 = np.where(sid==2)[0]; mk3 = np.where(sid==3)[0]
    ax3.axhline(om_term3, color=MANIOBRAS[2]['color'], lw=0.9, ls='--', alpha=0.6)
    ax3.axhline(om_term4, color=MANIOBRAS[3]['color'], lw=0.9, ls='--', alpha=0.6)
    ax3.annotate(f'$\\omega_{{term,3}}={om_term3:.3f}$ rad/s',
                 xy=(times[mk2[int(len(mk2)*0.6)]], om_term3),
                 xytext=(times[mk2[int(len(mk2)*0.6)]], om_term3 + 0.06),
                 fontsize=7.5, color=MANIOBRAS[2]['color'])
    ax3.annotate(f'$\\omega_{{term,4}}={om_term4:.3f}$ rad/s',
                 xy=(times[mk3[int(len(mk3)*0.6)]], om_term4),
                 xytext=(times[mk3[int(len(mk3)*0.6)]], om_term4 + 0.06),
                 fontsize=7.5, color=MANIOBRAS[3]['color'])

    ax3.axhline(0, color='#AAAAAA', lw=0.7, ls=':')
    ax3.set_ylabel('$\\omega$ [rad/s]', fontsize=9)
    ax3.set_xlabel('Tiempo [s]', fontsize=9)
    ax3.set_title('Velocidad angular $\\omega(t)$', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')

    fig.suptitle(
        f'Guia 1 P4 Novoa IPD482 — Dinamica uniciclo  |  '
        f'$m={m}$ kg, $I={I}$ kg·m$^2$, $r={r}$ m, $L={L}$ m',
        fontsize=9, fontweight='bold')

    fig.savefig('/mnt/user-data/outputs/p4_static.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('OK  p4_static.png')
    plt.close(fig)

# =============================================================================
# 9. GIF EN TIEMPO REAL
# =============================================================================

def make_gif(states, tauR, tauL, times, sid):
    N      = len(states)
    stride = max(1, N // 300)

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             left=0.06, right=0.97, top=0.92, bottom=0.08,
                             hspace=0.50, wspace=0.38)

    ax_xy = fig.add_subplot(gs[:, 0])
    ax1   = fig.add_subplot(gs[0, 1])
    ax2   = fig.add_subplot(gs[1, 1])
    ax3   = fig.add_subplot(gs[2, 1])

    margin = 0.25
    ax_xy.set_xlim(states[:,0].min()-margin, states[:,0].max()+margin)
    ax_xy.set_ylim(states[:,1].min()-margin, states[:,1].max()+margin)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Trayectoria XY — 4 Maniobras', fontsize=11, fontweight='bold')

    # Trazas de fondo tenues
    for mi, man in enumerate(MANIOBRAS):
        mk = sid == mi
        ax_xy.plot(states[mk, 0], states[mk, 1], color=man['color'], lw=0.8, alpha=0.15)

    ax_xy.legend(
        handles=[Line2D([0],[0], color=man['color'], lw=2.5, label=man['label']) for man in MANIOBRAS],
        fontsize=8, loc='upper right', framealpha=0.90)

    for ax_ in [ax1, ax2, ax3]:
        for mi, man in enumerate(MANIOBRAS):
            mk = sid == mi
            if mk.any():
                ax_.axvspan(times[mk][0], times[mk][-1], alpha=0.07, color=man['color'])
        ax_.set_facecolor('#F8F9FA')
        ax_.grid(True, color='#DDDDDD', lw=0.6, alpha=0.8)
        ax_.tick_params(colors='#212121', labelsize=8)
        for sp in ax_.spines.values(): sp.set_edgecolor('#AAAAAA')

    tau_max = max(abs(tauR).max(), abs(tauL).max()) * 1000 * 1.2
    ax1.set_xlim(times[0], times[-1]); ax1.set_ylim(-tau_max, tau_max)
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(states[:,3].min()*1.2-0.05, states[:,3].max()*1.2+0.05)
    ax3.set_xlim(times[0], times[-1])
    ax3.set_ylim(states[:,4].min()*1.2-0.05, states[:,4].max()*1.2+0.05)

    ax1.axhline(0, color='#AAAAAA', lw=0.7, ls=':')
    ax2.axhline(0, color='#AAAAAA', lw=0.7, ls=':')
    ax3.axhline(0, color='#AAAAAA', lw=0.7, ls=':')

    ax1.set_ylabel('Torque [mN·m]', fontsize=9)
    ax1.set_title('Torques $\\tau_R$, $\\tau_L$', fontsize=10, fontweight='bold')
    ax2.set_ylabel('$v$ [m/s]', fontsize=9)
    ax2.set_title('Velocidad lineal $v(t)$', fontsize=10, fontweight='bold')
    ax3.set_ylabel('$\\omega$ [rad/s]', fontsize=9)
    ax3.set_xlabel('Tiempo [s]', fontsize=9)
    ax3.set_title('Velocidad angular $\\omega(t)$', fontsize=10, fontweight='bold')

    tray_lines = [ax_xy.plot([], [], color=man['color'], lw=1.8)[0] for man in MANIOBRAS]
    ltr,  = ax1.plot([], [], color='#2E7D32', lw=1.6, label='$\\tau_R$')
    ltl,  = ax1.plot([], [], color='#E65100', lw=1.6, ls='--', label='$\\tau_L$')
    ax1.legend(fontsize=8, loc='upper right')
    lv_lines = [ax2.plot([], [], color=man['color'], lw=1.8)[0] for man in MANIOBRAS]
    lw_lines = [ax3.plot([], [], color=man['color'], lw=1.8)[0] for man in MANIOBRAS]

    cur1 = ax1.axvline(0, color='red', lw=1.0, alpha=0.7)
    cur2 = ax2.axvline(0, color='red', lw=1.0, alpha=0.7)
    cur3 = ax3.axvline(0, color='red', lw=1.0, alpha=0.7)

    fig.suptitle('Guia 1 P4 Novoa IPD482 — Dinamica uniciclo (animacion)',
                 fontsize=10, fontweight='bold')

    dyn_patches = []

    def init():
        for l in tray_lines + lv_lines + lw_lines:
            l.set_data([], [])
        ltr.set_data([], []); ltl.set_data([], [])
        return tray_lines + lv_lines + lw_lines + [ltr, ltl]

    def update(frame):
        k = min(frame * stride, N - 1)
        for p in dyn_patches:
            try: p.remove()
            except: pass
        dyn_patches.clear()

        for mi, man in enumerate(MANIOBRAS):
            m_all = np.where(sid == mi)[0]
            m_k   = m_all[m_all <= k]
            if len(m_k):
                tray_lines[mi].set_data(states[m_k, 0], states[m_k, 1])
                idx_last = m_k[-1]
                circ = patches.Circle(
                    (states[idx_last, 0], states[idx_last, 1]), 0.05,
                    facecolor=man['color'], edgecolor='#333',
                    linewidth=0.8, alpha=0.9, zorder=4)
                ax_xy.add_patch(circ)
                dyn_patches.append(circ)
                dx = 0.09 * np.cos(states[idx_last, 2])
                dy = 0.09 * np.sin(states[idx_last, 2])
                arr = ax_xy.annotate('',
                    xy=(states[idx_last,0]+dx, states[idx_last,1]+dy),
                    xytext=(states[idx_last,0], states[idx_last,1]),
                    arrowprops=dict(arrowstyle='->', color=man['color'], lw=1.3), zorder=5)
                dyn_patches.append(arr)

                lv_lines[mi].set_data(times[m_k], states[m_k, 3])
                lw_lines[mi].set_data(times[m_k], states[m_k, 4])

        t_k = times[:k+1]
        ltr.set_data(t_k, tauR[:k+1] * 1000)
        ltl.set_data(t_k, tauL[:k+1] * 1000)

        for cur in [cur1, cur2, cur3]:
            cur.set_xdata([times[k]])

        return tray_lines + lv_lines + lw_lines + [ltr, ltl, cur1, cur2, cur3]

    ani = animation.FuncAnimation(fig, update, frames=N//stride,
                                  init_func=init, interval=40, blit=False)
    ani.save('/mnt/user-data/outputs/p4_realtime.gif', writer='pillow', fps=25, dpi=100)
    print('OK  p4_realtime.gif')
    plt.close(fig)

# =============================================================================
# 10. NOTEBOOK INTERACTIVO
# =============================================================================

def generate_notebook(states, tauR, tauL, times, sid):
    nb_code = f"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import ipywidgets as widgets
from IPython.display import display

m={m}; I={I}; r={r}; L={L}; cv={cv}; cw={cw}; tau={tau}
MANIOBRAS = {MANIOBRAS}
states = np.{repr(states)}
tauR   = np.{repr(tauR)}
tauL   = np.{repr(tauL)}
times  = np.{repr(times)}
sid    = np.{repr(sid)}
N      = len(states)

def draw_frame(k):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    gs=gridspec.GridSpec(3,2,figure=fig,left=0.06,right=0.97,top=0.92,bottom=0.08,hspace=0.50,wspace=0.38)
    ax_xy=fig.add_subplot(gs[:,0]); ax1=fig.add_subplot(gs[0,1])
    ax2=fig.add_subplot(gs[1,1]); ax3=fig.add_subplot(gs[2,1])

    for mi,man in enumerate(MANIOBRAS):
        mk=sid==mi
        ax_xy.plot(states[mk,0],states[mk,1],color=man['color'],lw=0.8,alpha=0.2)
        m_all=np.where(mk)[0]; m_k=m_all[m_all<=k]
        if len(m_k):
            ax_xy.plot(states[m_k,0],states[m_k,1],color=man['color'],lw=1.8)
            idx=m_k[-1]
            circ=patches.Circle((states[idx,0],states[idx,1]),0.05,
                                  facecolor=man['color'],edgecolor='#333',lw=0.8,alpha=0.9,zorder=4)
            ax_xy.add_patch(circ)
            dx=0.09*np.cos(states[idx,2]); dy=0.09*np.sin(states[idx,2])
            ax_xy.annotate('',xy=(states[idx,0]+dx,states[idx,1]+dy),
                xytext=(states[idx,0],states[idx,1]),
                arrowprops=dict(arrowstyle='->',color=man['color'],lw=1.3),zorder=5)

    margin=0.25
    ax_xy.set_xlim(states[:,0].min()-margin,states[:,0].max()+margin)
    ax_xy.set_ylim(states[:,1].min()-margin,states[:,1].max()+margin)
    ax_xy.set_aspect('equal'); ax_xy.grid(True,alpha=0.3)
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title(f't = {{times[k]:.2f}} s',fontsize=11,fontweight='bold')
    ax_xy.legend(handles=[Line2D([0],[0],color=man['color'],lw=2.5,label=man['label'])
                           for man in MANIOBRAS],fontsize=8,loc='upper right')

    for ax_ in [ax1,ax2,ax3]:
        for mi,man in enumerate(MANIOBRAS):
            mk=sid==mi
            if mk.any(): ax_.axvspan(times[mk][0],times[mk][-1],alpha=0.07,color=man['color'])
        ax_.set_facecolor('#F8F9FA'); ax_.grid(True,color='#DDDDDD',lw=0.6,alpha=0.8)
        ax_.set_xlim(times[0],times[-1])
        ax_.axvline(times[k],color='red',lw=1.0,alpha=0.7)

    t_k=times[:k+1]
    ax1.plot(t_k,tauR[:k+1]*1000,color='#2E7D32',lw=1.6,label='tau_R')
    ax1.plot(t_k,tauL[:k+1]*1000,color='#E65100',lw=1.6,ls='--',label='tau_L')
    ax1.axhline(0,color='#AAAAAA',lw=0.8,ls=':')
    ax1.set_title('Torques',fontsize=10,fontweight='bold')
    ax1.set_ylabel('Torque [mN·m]',fontsize=9); ax1.legend(fontsize=8)

    for mi,man in enumerate(MANIOBRAS):
        mk=np.where(sid==mi)[0]; m_k=mk[mk<=k]
        if len(m_k):
            ax2.plot(times[m_k],states[m_k,3],color=man['color'],lw=1.8,label=f'M{{mi+1}}')
            ax3.plot(times[m_k],states[m_k,4],color=man['color'],lw=1.8,label=f'M{{mi+1}}')
    ax2.axhline(0,color='#AAAAAA',lw=0.7,ls=':')
    ax2.set_title('Velocidad lineal v(t)',fontsize=10,fontweight='bold')
    ax2.set_ylabel('v [m/s]',fontsize=9); ax2.legend(fontsize=8)
    ax3.axhline(0,color='#AAAAAA',lw=0.7,ls=':')
    ax3.set_title('Velocidad angular omega(t)',fontsize=10,fontweight='bold')
    ax3.set_ylabel('omega [rad/s]',fontsize=9); ax3.set_xlabel('Tiempo [s]',fontsize=9); ax3.legend(fontsize=8)

    fig.suptitle('Guia 1 P4 Novoa IPD482 — Dinamica uniciclo',fontsize=10,fontweight='bold')
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
                "# Guia 1 P4 Novoa IPD482\n",
                "Dinamica directa del robot uniciclo — Newton-Euler\n\n",
                "4 maniobras: adelante, atras, una rueda, contrapuestas.\n",
                "Usa el slider o Play para navegar la simulacion."
            ]},
            {"cell_type": "code", "execution_count": None,
             "metadata": {}, "outputs": [], "source": [nb_code]}
        ]
    }
    with open('/mnt/user-data/outputs/p4_interactivo.ipynb', 'w') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print('OK  p4_interactivo.ipynb')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print(f'v_term    = {v_term:.4f} m/s')
    print(f'tau_v     = {tau_v:.4f} s')
    print(f'om_term3  = {om_term3:.4f} rad/s')
    print(f'om_term4  = {om_term4:.4f} rad/s')

    states, tauR, tauL, times, sid = simulate()
    print(f'Pasos: {len(states)}   t_total: {times[-1]:.1f} s')

    plot_static(states, tauR, tauL, times, sid)
    make_gif(states, tauR, tauL, times, sid)
    generate_notebook(states, tauR, tauL, times, sid)
