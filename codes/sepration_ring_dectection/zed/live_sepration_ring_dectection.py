"""
ZED2i Live Point Cloud — Plane Detection + Annulus Circle Fitting
-----------------------------------------------------------------
- Captures live point cloud from ZED2i
- Filters points within 150cm (1.5m) from camera
- Extracts up to 5 planes via RANSAC (shown as colored rectangles)
- Fits inner/outer circles on each plane via angular binning
- Draws detected annulus rings as cyan (inner) and orange (outer) overlays

No trigger key — detection runs continuously from start.

Controls:
  ESC / Q  - quit
"""

import pyzed.sl as sl
import open3d as o3d
import numpy as np
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# --- ZED camera ---
ZED_RESOLUTION   = sl.RESOLUTION.HD720   # 720p — good balance of speed/density
ZED_FPS          = 30
ZED_DEPTH_MODE   = sl.DEPTH_MODE.PERFORMANCE
ZED_COORD_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
ZED_UNIT         = sl.UNIT.METER

MAX_DEPTH_M      = 1    # only keep points within 150cm

# --- Plane detection ---
MAX_PLANES         = 5
PLANE_DIST_THRESH  = 0.015   # metres — tight for flat surfaces
PLANE_RANSAC_N     = 3
PLANE_RANSAC_ITER  = 300
MIN_INLIERS        = 30

# --- Angular binning ---
NUM_ANGLE_BINS     = 72
MIN_BINS_FILLED    = 10

# --- Annulus validation ---
MIN_ANNULUS_RATIO  = 1.05
MAX_ANNULUS_RATIO  = 3.00
OUTER_OUTLIER_FRAC = 0.10
INNER_OUTLIER_FRAC = 0.10

# --- Visuals ---
POINT_SIZE         = 2.0
LINE_WIDTH         = 3.0
BG_COLOR           = [0.05, 0.05, 0.05]
POINT_COLOR        = [0.0, 0.9, 1.0]
PLANE_VIZ_SIZE     = 0.3     # half-size of plane debug rectangle (metres)
OVERLAY_SEGMENTS   = 64

PLANE_COLORS = [
    [1.0, 1.0, 0.0],   # yellow
    [1.0, 0.0, 1.0],   # magenta
    [0.0, 1.0, 0.0],   # green
    [1.0, 1.0, 1.0],   # white
    [1.0, 0.5, 0.0],   # orange
]
INNER_COLOR  = [0.0, 1.0, 1.0]   # cyan
OUTER_COLOR  = [1.0, 0.3, 0.0]   # deep orange

# --- Timing ---
DETECT_INTERVAL_SEC = 0.2
LOG_INTERVAL_SEC    = 1.0
# ──────────────────────────────────────────────────────────────────────────────


# ─── ZED CAPTURE ──────────────────────────────────────────────────────────────

def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution  = ZED_RESOLUTION
    init_params.camera_fps         = ZED_FPS
    init_params.depth_mode         = ZED_DEPTH_MODE
    init_params.coordinate_system  = ZED_COORD_SYSTEM
    init_params.coordinate_units   = ZED_UNIT
    init_params.depth_minimum_distance = 0.2   # 20cm min
    init_params.depth_maximum_distance = MAX_DEPTH_M

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open failed: {status}")
        sys.exit(1)

    print(f"ZED2i opened — {ZED_RESOLUTION} @ {ZED_FPS}fps")
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold    = 50
    runtime_params.texture_confidence_threshold = 100
    return zed, runtime_params


def grab_point_cloud(zed, runtime_params):
    """
    Grabs one frame, returns Nx3 numpy array of valid 3D points
    within MAX_DEPTH_M from camera.
    """
    point_cloud_zed = sl.Mat()

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return np.zeros((0, 3))

    zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA)
    pc_data = point_cloud_zed.get_data()   # shape: (H, W, 4) — X,Y,Z,RGBA

    xyz = pc_data[:, :, :3].reshape(-1, 3).astype(np.float32)

    # remove NaN / Inf
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]

    # distance filter: keep only points within MAX_DEPTH_M
    dist = np.linalg.norm(xyz, axis=1)
    xyz = xyz[dist <= MAX_DEPTH_M]
    xyz = xyz[dist[dist <= MAX_DEPTH_M] >= 0.2]   # also drop < 20cm

    return xyz


# ─── PLANE EXTRACTION ─────────────────────────────────────────────────────────

def extract_planes(pts_3d):
    if len(pts_3d) < MIN_INLIERS:
        return []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    planes = []; remaining = pcd
    for _ in range(MAX_PLANES):
        if len(remaining.points) < MIN_INLIERS:
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=PLANE_DIST_THRESH,
                ransac_n=PLANE_RANSAC_N,
                num_iterations=PLANE_RANSAC_ITER)
        except Exception:
            break
        if len(inliers) < MIN_INLIERS:
            break
        inlier_pts = np.asarray(remaining.select_by_index(inliers).points)
        planes.append((plane_model, inlier_pts))
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


# ─── 2D PROJECTION ────────────────────────────────────────────────────────────

def project_to_plane_2d(pts_3d, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c]); normal /= np.linalg.norm(normal)
    ref = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    axis_u = np.cross(normal, ref); axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    origin = pts_3d.mean(axis=0)
    shifted = pts_3d - origin
    pts_2d = np.stack([shifted @ axis_u, shifted @ axis_v], axis=1)
    return pts_2d, origin, axis_u, axis_v


# ─── CIRCLE FIT ───────────────────────────────────────────────────────────────

def fit_circle_least_squares(xy):
    if len(xy) < 3: return None
    x = xy[:,0]; y = xy[:,1]
    A = np.c_[x, y, np.ones(len(x))]; B = x**2 + y**2
    try: C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    except Exception: return None
    cx = C[0]/2; cy = C[1]/2; r2 = C[2] + cx**2 + cy**2
    if r2 <= 0: return None
    return cx, cy, np.sqrt(r2)


# ─── ANGULAR BINNING ──────────────────────────────────────────────────────────

def extract_boundaries_angular(pts_2d):
    center_guess = pts_2d.mean(axis=0)
    shifted = pts_2d - center_guess
    r = np.linalg.norm(shifted, axis=1)
    theta = np.arctan2(shifted[:,1], shifted[:,0])
    bins = np.linspace(-np.pi, np.pi, NUM_ANGLE_BINS+1)
    indices = np.digitize(theta, bins)
    inner_boundary = []; outer_boundary = []
    for i in range(1, NUM_ANGLE_BINS+1):
        bin_mask = (indices == i)
        if not np.any(bin_mask): continue
        rel_idx = np.where(bin_mask)[0]
        bin_r = r[bin_mask]
        inner_boundary.append(pts_2d[rel_idx[np.argmin(bin_r)]])
        outer_boundary.append(pts_2d[rel_idx[np.argmax(bin_r)]])
    if len(inner_boundary) < MIN_BINS_FILLED:
        return None, None
    return np.array(inner_boundary), np.array(outer_boundary)


# ─── ANNULUS DETECTION ON ONE PLANE ───────────────────────────────────────────

def detect_annulus_on_plane(plane_model, inlier_pts):
    if len(inlier_pts) < MIN_INLIERS: return None
    pts_2d, origin, axis_u, axis_v = project_to_plane_2d(inlier_pts, plane_model)
    inner_pts, outer_pts = extract_boundaries_angular(pts_2d)
    if inner_pts is None: return None
    inner_fit = fit_circle_least_squares(inner_pts)
    outer_fit = fit_circle_least_squares(outer_pts)
    if inner_fit is None or outer_fit is None: return None
    cx_i, cy_i, r_i = inner_fit
    cx_o, cy_o, r_o = outer_fit
    if r_i > r_o:
        cx_i,cy_i,r_i,cx_o,cy_o,r_o = cx_o,cy_o,r_o,cx_i,cy_i,r_i
    ratio = r_o/r_i if r_i > 1e-6 else 999
    if ratio < MIN_ANNULUS_RATIO or ratio > MAX_ANNULUS_RATIO: return None
    center_dist = np.sqrt((cx_i-cx_o)**2 + (cy_i-cy_o)**2)
    if center_dist + r_i > r_o * 1.08: return None
    cx = (cx_i+cx_o)/2; cy = (cy_i+cy_o)/2
    r_from_center = np.linalg.norm(pts_2d - np.array([cx,cy]), axis=1)
    frac_outside = np.mean(r_from_center > r_o)
    frac_inside  = np.mean(r_from_center < r_i)
    if frac_outside > OUTER_OUTLIER_FRAC: return None
    if frac_inside  > INNER_OUTLIER_FRAC: return None
    return {
        'cx_2d': cx, 'cy_2d': cy,
        'r_inner': r_i, 'r_outer': r_o,
        'score': 1.0 - frac_outside - frac_inside,
        'ratio': ratio,
        'origin': origin, 'axis_u': axis_u, 'axis_v': axis_v,
        'plane_model': plane_model,
        'n_inliers': len(inlier_pts),
        'frac_outside': frac_outside,
        'frac_inside': frac_inside,
    }


def detect_annulus_parallel(planes):
    if not planes: return None
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PLANES) as ex:
        futures = {ex.submit(detect_annulus_on_plane, pm, ip): i
                   for i, (pm, ip) in enumerate(planes)}
        for f in as_completed(futures):
            r = f.result()
            if r is not None: results.append(r)
    if not results: return None
    return max(results, key=lambda r: r['score'])


# ─── 3D RING BACK-PROJECTION ──────────────────────────────────────────────────

def make_ring_3d_pts(cx, cy, radius, origin, axis_u, axis_v, n_seg=OVERLAY_SEGMENTS):
    angles = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
    return np.array([origin + (cx + radius*np.cos(a))*axis_u
                              + (cy + radius*np.sin(a))*axis_v
                     for a in angles])


# ─── LINESET HELPERS ──────────────────────────────────────────────────────────

def make_empty_lineset():
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.zeros((2,3)))
    ls.lines  = o3d.utility.Vector2iVector([[0,1]])
    ls.colors = o3d.utility.Vector3dVector([[0,0,0]])
    return ls

def hide_lineset(ls):
    ls.points = o3d.utility.Vector3dVector(np.zeros((2,3)))
    ls.lines  = o3d.utility.Vector2iVector([[0,1]])
    ls.colors = o3d.utility.Vector3dVector([[0,0,0]])

def update_ring_lineset(ls, world_pts, color):
    n = len(world_pts)
    ls.points = o3d.utility.Vector3dVector(world_pts)
    ls.lines  = o3d.utility.Vector2iVector([[i,(i+1)%n] for i in range(n)])
    ls.colors = o3d.utility.Vector3dVector([color]*n)

def make_plane_rect_lineset(plane_model, inlier_pts, color, half_size=PLANE_VIZ_SIZE):
    a,b,c,d = plane_model
    normal = np.array([a,b,c]); normal /= np.linalg.norm(normal)
    ref = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    axis_u = np.cross(normal, ref); axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    centroid = inlier_pts.mean(axis=0); s = half_size
    corners = np.array([
        centroid+s*axis_u+s*axis_v, centroid-s*axis_u+s*axis_v,
        centroid-s*axis_u-s*axis_v, centroid+s*axis_u-s*axis_v])
    arrow_end = centroid + normal*s*0.5
    all_pts = np.vstack([corners, centroid, arrow_end])
    lines  = [[0,1],[1,2],[2,3],[3,0],[0,2],[1,3],[4,5]]
    colors = [color]*6 + [[1,1,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def make_world_axes(length=0.3):
    pts = np.array([[0,0,0],[length,0,0],[0,0,0],[0,length,0],[0,0,0],[0,0,length]])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector([[0,1],[2,3],[4,5]])
    ls.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return ls


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # init ZED
    zed, runtime_params = init_zed()

    running      = [True]
    lock         = threading.Lock()
    latest_pts   = [np.zeros((0,3))]
    frame_dirty  = threading.Event()
    detect_dirty = threading.Event()

    overlay_lock       = threading.Lock()
    plane_overlay_data = [None]*MAX_PLANES
    ring_inner_pts     = [None]
    ring_outer_pts     = [None]
    overlay_dirty      = [False]

    # ── ZED capture thread ────────────────────────────────────────────────────
    def capture_loop():
        while running[0]:
            pts = grab_point_cloud(zed, runtime_params)
            with lock:
                latest_pts[0] = pts
            frame_dirty.set()
            detect_dirty.set()
            # ZED already runs at camera FPS — no extra sleep needed

    threading.Thread(target=capture_loop, daemon=True).start()

    # ── Detection thread ──────────────────────────────────────────────────────
    last_log = [0.0]

    def detection_loop():
        while running[0]:
            detect_dirty.wait(timeout=0.5)
            detect_dirty.clear()

            with lock:
                pts = latest_pts[0].copy()

            if len(pts) < MIN_INLIERS:
                time.sleep(DETECT_INTERVAL_SEC)
                continue

            planes  = extract_planes(pts)
            result  = detect_annulus_parallel(planes)

            # build plane rect overlays
            new_plane_data = [None]*MAX_PLANES
            for i, (pm, ip) in enumerate(planes):
                src = make_plane_rect_lineset(pm, ip, PLANE_COLORS[i%5])
                new_plane_data[i] = (
                    np.asarray(src.points).copy(),
                    np.asarray(src.lines).tolist(),
                    np.asarray(src.colors).tolist(),
                )

            with overlay_lock:
                for i in range(MAX_PLANES):
                    plane_overlay_data[i] = new_plane_data[i]
                if result:
                    ring_inner_pts[0] = make_ring_3d_pts(
                        result['cx_2d'], result['cy_2d'], result['r_inner'],
                        result['origin'], result['axis_u'], result['axis_v'])
                    ring_outer_pts[0] = make_ring_3d_pts(
                        result['cx_2d'], result['cy_2d'], result['r_outer'],
                        result['origin'], result['axis_u'], result['axis_v'])
                else:
                    ring_inner_pts[0] = None
                    ring_outer_pts[0] = None
                overlay_dirty[0] = True

            # terminal log
            now = time.time()
            if now - last_log[0] > LOG_INTERVAL_SEC:
                last_log[0] = now
                print("\n" + "─"*55)
                print(f"  Points in frame : {len(pts)}  (within {MAX_DEPTH_M*100:.0f}cm)")
                print(f"  Planes found    : {len(planes)}")
                for i, (pm, ip) in enumerate(planes):
                    a,b,c,d = pm
                    cname = ['Yellow','Magenta','Green','White','Orange'][i]
                    print(f"    Plane {i} [{cname:7s}]: n={len(ip):4d}  "
                          f"normal=({a:+.2f},{b:+.2f},{c:+.2f})")
                if result:
                    print(f"  ANNULUS : DETECTED")
                    print(f"    Center  : ({result['cx_2d']:+.4f}, {result['cy_2d']:+.4f})")
                    print(f"    R inner : {result['r_inner']:.4f} m")
                    print(f"    R outer : {result['r_outer']:.4f} m")
                    print(f"    Ratio   : {result['ratio']:.3f}x")
                    print(f"    Score   : {result['score']*100:.1f}%")
                else:
                    print(f"  ANNULUS : NO DETECTION")
                print("─"*55)

            time.sleep(DETECT_INTERVAL_SEC)

    threading.Thread(target=detection_loop, daemon=True).start()

    # ── Open3D viewer ─────────────────────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="ZED2i Live — Annulus Detection", width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array(BG_COLOR)
    opt.point_size       = POINT_SIZE
    opt.line_width       = LINE_WIDTH

    vis.add_geometry(make_world_axes(0.3))

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    plane_ls = [make_empty_lineset() for _ in range(MAX_PLANES)]
    for ls in plane_ls: vis.add_geometry(ls)

    ring_inner_ls = make_empty_lineset()
    ring_outer_ls = make_empty_lineset()
    vis.add_geometry(ring_inner_ls)
    vis.add_geometry(ring_outer_ls)

    # ── Key callbacks ─────────────────────────────────────────────────────────
    KEY_ESC = 256; KEY_Q = 81

    def cb_quit(v):
        running[0] = False; v.close(); return False

    vis.register_key_callback(KEY_ESC, cb_quit)
    vis.register_key_callback(KEY_Q,   cb_quit)

    # ── Animation callback ────────────────────────────────────────────────────
    def on_animation(v):
        if frame_dirty.is_set():
            frame_dirty.clear()
            with lock:
                pts = latest_pts[0].copy()
            if len(pts) > 0:
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.paint_uniform_color(POINT_COLOR)
            else:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((0,3)))
            v.update_geometry(pcd)

        with overlay_lock:
            dirty = overlay_dirty[0]
            if dirty:
                pd = [plane_overlay_data[i] for i in range(MAX_PLANES)]
                ip = ring_inner_pts[0]
                op = ring_outer_pts[0]
                overlay_dirty[0] = False

        if dirty:
            for i, ls in enumerate(plane_ls):
                if pd[i] is not None:
                    p, l, c = pd[i]
                    ls.points = o3d.utility.Vector3dVector(p)
                    ls.lines  = o3d.utility.Vector2iVector(l)
                    ls.colors = o3d.utility.Vector3dVector(c)
                else:
                    hide_lineset(ls)
                v.update_geometry(ls)

            if ip is not None and len(ip) > 1:
                update_ring_lineset(ring_inner_ls, ip, INNER_COLOR)
                update_ring_lineset(ring_outer_ls, op, OUTER_COLOR)
            else:
                hide_lineset(ring_inner_ls)
                hide_lineset(ring_outer_ls)
            v.update_geometry(ring_inner_ls)
            v.update_geometry(ring_outer_ls)

        return False

    vis.register_animation_callback(on_animation)

    print("\n=== ZED2i Live Annulus Detection ===")
    print(f"  Depth filter : within {MAX_DEPTH_M*100:.0f}cm")
    print(f"  Max planes   : {MAX_PLANES}")
    print(f"  Plane thresh : {PLANE_DIST_THRESH}m")
    print(f"  Ratio range  : [{MIN_ANNULUS_RATIO}, {MAX_ANNULUS_RATIO}]")
    print("\nPlane colors: Yellow | Magenta | Green | White | Orange")
    print("Rings: CYAN=inner  ORANGE=outer")
    print("Press ESC or Q to quit\n")

    vis.run()
    vis.destroy_window()
    running[0] = False
    zed.close()
    print("ZED closed. Done.")


if __name__ == "__main__":
    main()