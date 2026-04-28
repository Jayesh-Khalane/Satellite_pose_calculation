"""
Live Satellite Point Cloud Simulator
-------------------------------------
Controls:
  W/S        - move forward/backward
  A/D        - strafe left/right
  Q/E        - move up/down
  Up/Down    - pitch camera
  Left/Right - yaw camera
  R          - reset camera
  ESC        - quit
"""

import open3d as o3d
import numpy as np
import threading
import time
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STL_PATH       = r"D:/ring.stl"  # Path to your STL file here

# --- Camera FOV ---
FOV_H_DEG      = 70.0       # horizontal field of view in degrees

# --- Ray density (more rays = more points, slower) ---
RAY_W          = 150       
RAY_H          = 100       

# --- Depth range ---
NEAR_CLIP      = 0.3        # ignore hits closer than this (metres)
FAR_CLIP       = 20.0       # ignore hits farther than this (metres)

# --- Camera movement ---
MOVE_SPEED     = 0.01       # metres per frame while key held
LOOK_SPEED_DEG = 1.0        # degrees per frame while key held

# --- Visuals ---
POINT_SIZE     = 2.0        # point cloud dot size (pixels)
BG_COLOR       = [0.05, 0.05, 0.05]
POINT_COLOR    = [0.0, 0.2, 0.9]
# ──────────────────────────────────────────────────────────────────────────────

KEY_W = 87; KEY_S = 83; KEY_A = 65; KEY_D = 68
KEY_Q = 81; KEY_E = 69; KEY_R = 82; KEY_ESC = 256
KEY_UP = 265; KEY_DOWN = 264; KEY_LEFT = 263; KEY_RIGHT = 262
MOVE_KEYS = [KEY_W, KEY_S, KEY_A, KEY_D, KEY_Q, KEY_E,
             KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]
KEY_HOLD_WINDOW = 0.15


def build_ray_grid(fov_h_deg, ray_w, ray_h):
    fov_h = np.radians(fov_h_deg)
    fov_v = fov_h * (ray_h / ray_w)
    u = np.linspace(-np.tan(fov_h / 2), np.tan(fov_h / 2), ray_w)
    v = np.linspace( np.tan(fov_v / 2), -np.tan(fov_v / 2), ray_h)
    uu, vv = np.meshgrid(u, v)
    dirs = np.stack([uu.ravel(), vv.ravel(), np.ones(ray_w * ray_h)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs.astype(np.float32)


def rotation_matrix(yaw_deg, pitch_deg):
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [ 0,         1, 0        ],
                   [-np.sin(y), 0, np.cos(y)]], dtype=np.float64)
    Rx = np.array([[1, 0,          0         ],
                   [0, np.cos(p), -np.sin(p) ],
                   [0, np.sin(p),  np.cos(p) ]], dtype=np.float64)
    return Ry @ Rx


def cast_points(scene, cam_pos, R, ray_dirs_cam):
    ray_dirs_world = (R @ ray_dirs_cam.T).T.astype(np.float32)
    n = len(ray_dirs_cam)
    origins = np.tile(cam_pos.astype(np.float32), (n, 1))
    rays = np.concatenate([origins, ray_dirs_world], axis=1)
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_t)
    t_hit = ans['t_hit'].numpy()
    valid = (t_hit > NEAR_CLIP) & (t_hit < FAR_CLIP)
    return origins[valid] + ray_dirs_world[valid] * t_hit[valid, np.newaxis]


def make_camera_marker(size=0.15):
    s = size
    f = s;  b = -s * 0.5;  hw = s * 0.5;  hh = s * 0.35
    box_pts = np.array([
        [-hw, -hh, b], [ hw, -hh, b], [ hw,  hh, b], [-hw,  hh, b],
        [-hw, -hh, f], [ hw, -hh, f], [ hw,  hh, f], [-hw,  hh, f],
    ])
    box_lines  = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    box_colors = [[0.8, 0.8, 0.2]] * len(box_lines)

    ax_len = s * 1.5
    axis_pts   = np.array([[0,0,0],[ax_len,0,0],[0,0,0],[0,ax_len,0],[0,0,0],[0,0,ax_len]], dtype=np.float64)
    axis_lines  = [[0,1],[2,3],[4,5]]
    axis_colors = [[1,0,0],[0,1,0],[0,0,1]]

    fov_h = np.radians(FOV_H_DEG)
    fov_v = fov_h * (RAY_H / RAY_W)
    fd = s * 2.0
    tx = np.tan(fov_h/2) * fd;  ty = np.tan(fov_v/2) * fd
    frust_pts   = np.array([[0,0,0],[tx,ty,fd],[-tx,ty,fd],[-tx,-ty,fd],[tx,-ty,fd]])
    frust_lines  = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    frust_colors = [[0.0, 0.8, 1.0]] * len(frust_lines)

    all_pts = np.vstack([box_pts, axis_pts, frust_pts])
    ob = 0;  oa = len(box_pts);  of_ = oa + len(axis_pts)

    lines  = ([[l[0]+ob,  l[1]+ob]  for l in box_lines]  +
              [[l[0]+oa,  l[1]+oa]  for l in axis_lines]  +
              [[l[0]+of_, l[1]+of_] for l in frust_lines])
    colors = box_colors + axis_colors + frust_colors

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls, all_pts


def update_camera_marker(ls, base_pts, cam_pos, R):
    transformed = (R @ base_pts.T).T + cam_pos
    ls.points = o3d.utility.Vector3dVector(transformed)


def make_world_axes(length=0.5):
    pts    = np.array([[0,0,0],[length,0,0],[0,0,0],[0,length,0],[0,0,0],[0,0,length]], dtype=np.float64)
    lines  = [[0,1],[2,3],[4,5]]
    colors = [[1,0,0],[0,1,0],[0,0,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


class CameraState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos   = np.array([0.0, 0.0, -3.0])
        self.yaw   = 0.0
        self.pitch = 0.0          # unbounded — full sphere allowed

    @property
    def R(self):       return rotation_matrix(self.yaw, self.pitch)
    @property
    def forward(self): return self.R @ np.array([0, 0, 1])
    @property
    def right(self):   return self.R @ np.array([1, 0, 0])
    @property
    def up(self):      return self.R @ np.array([0, 1, 0])


def main():
    print(f"Loading STL: {STL_PATH}")
    try:
        mesh = o3d.io.read_triangle_mesh(STL_PATH)
    except Exception as e:
        print(f"ERROR: {e}"); sys.exit(1)

    if not mesh.has_triangles():
        print("ERROR: STL has no triangles."); sys.exit(1)

    mesh.compute_vertex_normals()
    bb = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bb.get_center())
    mesh.scale(2.0 / max(bb.get_extent()), center=[0, 0, 0])
    print(f"Mesh loaded. Verts={len(mesh.vertices)}, Tris={len(mesh.triangles)}")
    mesh.paint_uniform_color([0.45, 0.45, 0.5])

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    print("Raycasting scene built.")

    ray_dirs_cam = build_ray_grid(FOV_H_DEG, RAY_W, RAY_H)
    print(f"Ray grid: {RAY_W}x{RAY_H} = {RAY_W*RAY_H} rays per frame")

    cam     = CameraState()
    lock    = threading.Lock()
    dirty   = threading.Event()
    running = [True]
    new_pts = [np.zeros((0, 3))]

    def raycast_loop():
        while running[0]:
            with lock:
                pos = cam.pos.copy()
                R   = cam.R.copy()
            pts = cast_points(scene, pos, R, ray_dirs_cam)
            with lock:
                new_pts[0] = pts
            dirty.set()
            time.sleep(0.033)

    threading.Thread(target=raycast_loop, daemon=True).start()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Satellite Live Sim", width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color    = np.array(BG_COLOR)
    opt.point_size          = POINT_SIZE
    opt.mesh_show_back_face = True
    opt.light_on            = True

    vis.add_geometry(mesh)
    vis.add_geometry(make_world_axes(0.4))

    cam_marker, base_pts = make_camera_marker(size=0.12)
    vis.add_geometry(cam_marker)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    last_press_time = {}

    def make_cb(key):
        def cb(v): last_press_time[key] = time.time(); return False
        return cb

    for k in MOVE_KEYS:
        vis.register_key_callback(k, make_cb(k))

    def cb_r(v):
        with lock: cam.reset(); return False

    def cb_esc(v):
        running[0] = False; v.close(); return False

    vis.register_key_callback(KEY_R,   cb_r)
    vis.register_key_callback(KEY_ESC, cb_esc)

    def on_animation(v):
        now = time.time()
        def active(k): return (now - last_press_time.get(k, 0)) < KEY_HOLD_WINDOW

        with lock:
            if active(KEY_W):     cam.pos += MOVE_SPEED * cam.forward
            if active(KEY_S):     cam.pos -= MOVE_SPEED * cam.forward
            if active(KEY_A):     cam.pos -= MOVE_SPEED * cam.right
            if active(KEY_D):     cam.pos += MOVE_SPEED * cam.right
            if active(KEY_Q):     cam.pos -= MOVE_SPEED * cam.up
            if active(KEY_E):     cam.pos += MOVE_SPEED * cam.up
            if active(KEY_UP):    cam.pitch -= LOOK_SPEED_DEG   # no clamp — full sphere
            if active(KEY_DOWN):  cam.pitch += LOOK_SPEED_DEG
            if active(KEY_LEFT):  cam.yaw   -= LOOK_SPEED_DEG
            if active(KEY_RIGHT): cam.yaw   += LOOK_SPEED_DEG

            pos = cam.pos.copy()
            R   = cam.R.copy()

        update_camera_marker(cam_marker, base_pts, pos, R)
        v.update_geometry(cam_marker)

        if dirty.is_set():
            dirty.clear()
            with lock:
                pts = new_pts[0].copy()
            if len(pts) > 0:
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.paint_uniform_color(POINT_COLOR)
            else:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            v.update_geometry(pcd)

        return False

    vis.register_animation_callback(on_animation)

    print("\n=== Controls ===")
    print("  W/S        : forward / back")
    print("  A/D        : strafe left / right")
    print("  Q/E        : up / down")
    print("  Arrow keys : look (Up/Down now full 360)")
    print("  R          : reset camera")
    print("  ESC        : quit")
    print("World axes: Red=X  Green=Y  Blue=Z")
    print("\n>>> Click inside the window first to capture keyboard input <<<\n")

    vis.run()
    vis.destroy_window()
    running[0] = False


if __name__ == "__main__":
    main()