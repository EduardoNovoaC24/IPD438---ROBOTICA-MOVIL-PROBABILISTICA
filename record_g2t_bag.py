"""
Grabador de datos G2T v2 — IPD-482
Correcciones:
  - psi1 normalizado con atan2 (evita valores >180°)
  - psi2 eliminado (solo 1 trailer)
  - hitch_pin como referencia del ángulo de articulación
  - Mejor detección de hits LiDAR
"""

import time
import math
import struct
import sqlite3
import os
import yaml

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("ERROR: pip install coppeliasim-zmqremoteapi-client")
    exit(1)

# ── Configuración ─────────────────────────────────────────────
OUTPUT_DIR      = "g2t_dataset"
RATE_HZ         = 10
LIDAR_FOV       = 270.0
LIDAR_RANGE_MAX = 30.0
LIDAR_RANGE_MIN = 0.01
LIDAR_RES       = 1040

# ── Conexión ──────────────────────────────────────────────────
print("Conectando a CoppeliaSim...")
client = RemoteAPIClient()
sim    = client.require('sim')
print(f"✓ Conectado — v{sim.getInt32Param(sim.intparam_program_version)}")

# ── Handles ───────────────────────────────────────────────────
def get_handle(name):
    try:
        h = sim.getObject(f'/{name}')
        print(f"✓ {name} (handle={h})")
        return h
    except:
        print(f"⚠ {name} no encontrado")
        return None

lidar_h    = get_handle('visionSensor')
tractor_h  = get_handle('tractor')
trailer1_h = get_handle('trailer1')
hitch_pin_h= get_handle('hitch_pin')   # referencia angular LiDAR

# ── Crear .bag ────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
db_path = os.path.join(OUTPUT_DIR, "g2t_dataset_0.db3")

# Eliminar db anterior si existe
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cur  = conn.cursor()
cur.executescript("""
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    serialization_format TEXT NOT NULL,
    offered_qos_profiles TEXT NOT NULL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    data BLOB NOT NULL
);
""")

topics = [
    ('/scan',        'sensor_msgs/msg/LaserScan'),
    ('/odom',        'nav_msgs/msg/Odometry'),
    ('/articulation','std_msgs/msg/Float64MultiArray'),
]
topic_ids = {}
for name, msg_type in topics:
    cur.execute(
        "INSERT INTO topics VALUES (NULL,?,?,?,?)",
        (name, msg_type, 'cdr', '[]')
    )
    topic_ids[name] = cur.lastrowid
conn.commit()

# ── Serialización CDR ─────────────────────────────────────────
CDR_HEADER = b'\x00\x01\x00\x00'

def pack_u32(v): return struct.pack('<I', int(v))
def pack_f32(v): return struct.pack('<f', float(v))
def pack_f64(v): return struct.pack('<d', float(v))
def pack_str(s):
    enc = s.encode('utf-8') + b'\x00'
    pad = (4 - len(enc) % 4) % 4
    return pack_u32(len(enc)) + enc + b'\x00' * pad

def ser_laserscan(sec, nsec, ranges):
    d  = CDR_HEADER
    d += pack_u32(sec) + pack_u32(nsec)
    d += pack_str('lidar_link')
    d += pack_f32(math.radians(-LIDAR_FOV/2))
    d += pack_f32(math.radians( LIDAR_FOV/2))
    d += pack_f32(math.radians(LIDAR_FOV) / LIDAR_RES)
    d += pack_f32(0.0)
    d += pack_f32(1.0/40.0)
    d += pack_f32(LIDAR_RANGE_MIN)
    d += pack_f32(LIDAR_RANGE_MAX)
    d += pack_u32(len(ranges))
    for r in ranges:
        d += pack_f32(r)
    d += pack_u32(0)  # intensities vacío
    return d

def ser_odometry(sec, nsec, pos, yaw):
    qz = math.sin(yaw/2)
    qw = math.cos(yaw/2)
    d  = CDR_HEADER
    d += pack_u32(sec) + pack_u32(nsec)
    d += pack_str('odom')
    d += pack_str('base_link')
    d += pack_f64(pos[0]) + pack_f64(pos[1]) + pack_f64(pos[2])
    d += pack_f64(0.0) + pack_f64(0.0) + pack_f64(qz) + pack_f64(qw)
    d += b'\x00' * (36*8)   # pose covariance
    d += b'\x00' * (6*8 + 36*8)  # twist
    return d

def ser_articulation(sec, nsec, psi1):
    d  = CDR_HEADER
    d += pack_u32(0) + pack_u32(0)  # layout vacío
    d += pack_u32(1)                # 1 valor
    d += pack_f64(psi1)
    return d

# ── Lectura LiDAR ─────────────────────────────────────────────
def read_lidar():
    if not lidar_h:
        return None
    try:
        img, res = sim.getVisionSensorDepth(lidar_h, 0)
        if not img:
            return None
        ranges = []
        for i in range(res[0]):
            idx = i * 4
            if idx + 4 <= len(img):
                val = struct.unpack('f', bytes(img[idx:idx+4]))[0]
                d   = val * LIDAR_RANGE_MAX
                ranges.append(float('inf') if d < LIDAR_RANGE_MIN
                               or d >= LIDAR_RANGE_MAX * 0.999 else d)
        return ranges
    except:
        return None

# ── Lectura pose ──────────────────────────────────────────────
def read_pose(h):
    if not h:
        return [0,0,0], [0,0,0]
    try:
        return sim.getObjectPosition(h,-1), sim.getObjectOrientation(h,-1)
    except:
        return [0,0,0], [0,0,0]

# ── Cálculo ψ1 normalizado ────────────────────────────────────
def calc_psi1(tractor_h, trailer1_h, hitch_pin_h):
    """
    Calcula el ángulo de articulación ψ1 correctamente normalizado.
    Usa hitch_pin si disponible (referencia directa), sino diferencia
    de orientaciones normalizada con atan2.
    """
    _, ori_t  = read_pose(tractor_h)
    _, ori_tr = read_pose(trailer1_h)

    # FIX: normalizar con atan2 para evitar valores >180°
    diff = ori_tr[2] - ori_t[2]
    psi1 = math.atan2(math.sin(diff), math.cos(diff))

    # Si tenemos hitch_pin, usar su orientación relativa al tractor
    # (más preciso porque es la referencia geométrica real)
    if hitch_pin_h:
        try:
            _, ori_pin = read_pose(hitch_pin_h)
            diff_pin   = ori_pin[2] - ori_t[2]
            psi1       = math.atan2(math.sin(diff_pin), math.cos(diff_pin))
        except:
            pass

    return psi1

# ── Loop principal ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Grabando G2T → {db_path}")
print(f"  Frecuencia: {RATE_HZ} Hz")
print(f"  Ctrl+C para detener")
print(f"{'='*50}\n")

period    = 1.0 / RATE_HZ
msg_count = 0
t0        = time.time()

try:
    while True:
        t_loop = time.time()
        t_ns   = int((t_loop - t0) * 1e9)
        sec    = int(t_ns // 1_000_000_000)
        nsec   = int(t_ns %  1_000_000_000)

        # LiDAR
        ranges = read_lidar()
        if ranges:
            cur.execute(
                "INSERT INTO messages VALUES (NULL,?,?,?)",
                (topic_ids['/scan'], t_ns, ser_laserscan(sec, nsec, ranges))
            )
            msg_count += 1

        # Odometría (ground truth tractor)
        pos, ori = read_pose(tractor_h)
        cur.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?)",
            (topic_ids['/odom'], t_ns,
             ser_odometry(sec, nsec, pos, ori[2]))
        )
        msg_count += 1

        # Ángulo de articulación ψ1 (normalizado)
        psi1 = calc_psi1(tractor_h, trailer1_h, hitch_pin_h)
        cur.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?)",
            (topic_ids['/articulation'], t_ns,
             ser_articulation(sec, nsec, psi1))
        )
        msg_count += 1

        conn.commit()

        # Log cada 2 segundos
        if sec % 2 == 0 and nsec < period * 1e9:
            hits = sum(1 for r in (ranges or []) if r != float('inf'))
            print(f"  t={sec:4d}s | msgs={msg_count:5d} | "
                  f"pos=({pos[0]:.2f},{pos[1]:.2f}) | "
                  f"ψ1={math.degrees(psi1):+6.1f}° | "
                  f"LiDAR hits={hits}/{LIDAR_RES}")

        elapsed = time.time() - t_loop
        time.sleep(max(0, period - elapsed))

except KeyboardInterrupt:
    conn.commit()
    dur = time.time() - t0
    print(f"\n{'='*50}")
    print(f"  Grabación terminada")
    print(f"  Duración : {dur:.1f} s")
    print(f"  Mensajes : {msg_count}")
    print(f"  Archivo  : {db_path}")
    print(f"{'='*50}")

finally:
    metadata = {
        'rosbag2_bagfile_information': {
            'version': 6,
            'storage_identifier': 'sqlite3',
            'relative_file_paths': ['g2t_dataset_0.db3'],
            'duration': {'nanoseconds': int((time.time()-t0)*1e9)},
            'starting_time': {'nanoseconds_since_epoch': int(t0*1e9)},
            'message_count': msg_count,
            'topics_with_message_count': [
                {'topic_metadata': {
                    'name': n, 'type': tp,
                    'serialization_format': 'cdr',
                    'offered_qos_profiles': '[]'},
                 'message_count': msg_count // 3}
                for n, tp in topics
            ],
            'compression_format': '',
            'compression_mode': '',
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)
    print(f"  metadata.yaml escrito")
    conn.close()
