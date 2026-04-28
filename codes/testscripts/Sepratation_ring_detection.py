"""
Live Satellite Point Cloud Simulator + Annulus Detection
Angular binning method for inner/outer boundary extraction.

Controls:
  W/S/A/D/Q/E - move camera
  Arrow keys   - look
  R            - reset camera
  J            - toggle detection ON/OFF
  ESC          - quit

Plane colors:
  Plane 0 : Yellow
  Plane 1 : Magenta
  Plane 2 : Green
  Plane 3 : White
  Plane 4 : Orange
  Annulus : CYAN inner + deep ORANGE outer
"""

import open3d as o3d
import numpy as np
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STL_PATH       = r"D:/ring.stl"

FOV_H_DEG      = 50.0
RAY_W          = 250
RAY_H          = 200
NEAR_CLIP      = 0.1
FAR_CLIP       = 20.0

MOVE_SPEED     = 0.01
LOOK_SPEED_DEG = 1.0
POINT_SIZE     = 2.0
BG_COLOR       = [0.05, 0.05, 0.05]
POINT_COLOR    = [0.0, 0.9, 1.0]

# --- Plane detection ---
MAX_PLANES         = 5
PLANE_DIST_THRESH  = 0.015
PLANE_RANSAC_N     = 3
PLANE_RANSAC_ITER  = 300
MIN_INLIERS        = 30

# --- Angular binning ---
NUM_ANGLE_BINS     = 72          # 5° per bin
MIN_BINS_FILLED    = 10          # need at least this many bins with points

# --- Annulus validation ---
MIN_ANNULUS_RATIO  = 1.05
MAX_ANNULUS_RATIO  = 3.00
OUTER_OUTLIER_FRAC = 0.10
INNER_OUTLIER_FRAC = 0.10

# --- Plane viz ---
PLANE_VIZ_SIZE     = 0.3

# --- Timing ---
DETECT_INTERVAL_SEC = 0.2
LOG_INTERVAL_SEC    = 1.0

# --- Colors ---
PLANE_COLORS = [
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],
]
INNER_COLOR     = [1, 0.0, 0.2]
OUTER_COLOR     = [1.0, 0.0, 0.2]
OVERLAY_SEGMENTS= 64
# ──────────────────────────────────────────────────────────────────────────────

KEY_W=87; KEY_S=83; KEY_A=65; KEY_D=68
KEY_Q=81; KEY_E=69; KEY_R=82; KEY_ESC=256; KEY_J=74
KEY_UP=265; KEY_DOWN=264; KEY_LEFT=263; KEY_RIGHT=262
MOVE_KEYS=[KEY_W,KEY_S,KEY_A,KEY_D,KEY_Q,KEY_E,KEY_UP,KEY_DOWN,KEY_LEFT,KEY_RIGHT]
KEY_HOLD_WINDOW=0.15


# ─── RAY GRID / ROTATION / CAST ───────────────────────────────────────────────

def build_ray_grid(fov_h_deg, ray_w, ray_h):
    fov_h=np.radians(fov_h_deg); fov_v=fov_h*(ray_h/ray_w)
    u=np.linspace(-np.tan(fov_h/2), np.tan(fov_h/2), ray_w)
    v=np.linspace( np.tan(fov_v/2),-np.tan(fov_v/2), ray_h)
    uu,vv=np.meshgrid(u,v)
    dirs=np.stack([uu.ravel(),vv.ravel(),np.ones(ray_w*ray_h)],axis=1)
    dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    return dirs.astype(np.float32)


def rotation_matrix(yaw_deg, pitch_deg):
    y=np.radians(yaw_deg); p=np.radians(pitch_deg)
    Ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    return (Ry@Rx).astype(np.float64)


def cast_frame(scene, cam_pos, R, ray_dirs_cam):
    ray_dirs_world=(R@ray_dirs_cam.T).T.astype(np.float32)
    n=len(ray_dirs_cam)
    origins=np.tile(cam_pos.astype(np.float32),(n,1))
    rays=np.concatenate([origins,ray_dirs_world],axis=1)
    rays_t=o3d.core.Tensor(rays,dtype=o3d.core.Dtype.Float32)
    ans=scene.cast_rays(rays_t)
    t_hit=ans['t_hit'].numpy()
    valid=(t_hit>NEAR_CLIP)&(t_hit<FAR_CLIP)
    pts=origins[valid]+ray_dirs_world[valid]*t_hit[valid,np.newaxis]
    return pts


# ─── PLANE EXTRACTION ─────────────────────────────────────────────────────────

def extract_planes(pts_3d):
    if len(pts_3d)<MIN_INLIERS: return []
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pts_3d)
    planes=[]; remaining=pcd
    for _ in range(MAX_PLANES):
        if len(remaining.points)<MIN_INLIERS: break
        try:
            plane_model,inliers=remaining.segment_plane(
                distance_threshold=PLANE_DIST_THRESH,
                ransac_n=PLANE_RANSAC_N,
                num_iterations=PLANE_RANSAC_ITER)
        except Exception: break
        if len(inliers)<MIN_INLIERS: break
        inlier_pts=np.asarray(remaining.select_by_index(inliers).points)
        planes.append((plane_model,inlier_pts))
        remaining=remaining.select_by_index(inliers,invert=True)
    return planes


# ─── PLANE RECT VISUALIZER ────────────────────────────────────────────────────

def make_plane_rect_lineset(plane_model, inlier_pts, color, half_size=PLANE_VIZ_SIZE):
    a,b,c,d=plane_model
    normal=np.array([a,b,c]); normal/=np.linalg.norm(normal)
    ref=np.array([1,0,0]) if abs(normal[0])<0.9 else np.array([0,1,0])
    axis_u=np.cross(normal,ref); axis_u/=np.linalg.norm(axis_u)
    axis_v=np.cross(normal,axis_u)
    centroid=inlier_pts.mean(axis=0)
    s=half_size
    corners=np.array([
        centroid+s*axis_u+s*axis_v,
        centroid-s*axis_u+s*axis_v,
        centroid-s*axis_u-s*axis_v,
        centroid+s*axis_u-s*axis_v,
    ])
    arrow_end=centroid+normal*s*0.5
    all_pts=np.vstack([corners,centroid,arrow_end])
    lines=[[0,1],[1,2],[2,3],[3,0],[0,2],[1,3],[4,5]]
    colors=[color]*6+[[1,1,1]]
    ls=o3d.geometry.LineSet()
    ls.points=o3d.utility.Vector3dVector(all_pts)
    ls.lines=o3d.utility.Vector2iVector(lines)
    ls.colors=o3d.utility.Vector3dVector(colors)
    return ls


# ─── 2D PROJECTION ────────────────────────────────────────────────────────────

def project_to_plane_2d(pts_3d, plane_model):
    a,b,c,d=plane_model
    normal=np.array([a,b,c]); normal/=np.linalg.norm(normal)
    ref=np.array([1,0,0]) if abs(normal[0])<0.9 else np.array([0,1,0])
    axis_u=np.cross(normal,ref); axis_u/=np.linalg.norm(axis_u)
    axis_v=np.cross(normal,axis_u)
    origin=pts_3d.mean(axis=0)
    shifted=pts_3d-origin
    pts_2d=np.stack([shifted@axis_u, shifted@axis_v],axis=1)
    return pts_2d, origin, axis_u, axis_v


# ─── CIRCLE FIT ───────────────────────────────────────────────────────────────

def fit_circle_least_squares(xy):
    """Algebraic least squares circle fit."""
    if len(xy)<3: return None
    x=xy[:,0]; y=xy[:,1]
    A=np.c_[x,y,np.ones(len(x))]; B=x**2+y**2
    try: C,_,_,_=np.linalg.lstsq(A,B,rcond=None)
    except Exception: return None
    cx=C[0]/2; cy=C[1]/2; r2=C[2]+cx**2+cy**2
    if r2<=0: return None
    return cx,cy,np.sqrt(r2)


# ─── ANGULAR BINNING ──────────────────────────────────────────────────────────

def extract_boundaries_angular(pts_2d, num_bins=NUM_ANGLE_BINS):
    """
    For each angular bin, take the min-radius point (inner boundary)
    and max-radius point (outer boundary).
    Returns (inner_pts, outer_pts) or (None, None) if too few bins filled.
    """
    center_guess=pts_2d.mean(axis=0)
    shifted=pts_2d-center_guess
    r=np.linalg.norm(shifted,axis=1)
    theta=np.arctan2(shifted[:,1],shifted[:,0])

    bins=np.linspace(-np.pi,np.pi,num_bins+1)
    indices=np.digitize(theta,bins)

    inner_boundary=[]; outer_boundary=[]

    for i in range(1,num_bins+1):
        bin_mask=(indices==i)
        if not np.any(bin_mask): continue
        rel_idx=np.where(bin_mask)[0]
        bin_r=r[bin_mask]
        inner_boundary.append(pts_2d[rel_idx[np.argmin(bin_r)]])
        outer_boundary.append(pts_2d[rel_idx[np.argmax(bin_r)]])

    if len(inner_boundary)<MIN_BINS_FILLED:
        return None, None

    return np.array(inner_boundary), np.array(outer_boundary)


# ─── ANNULUS DETECTION ON ONE PLANE ───────────────────────────────────────────

def detect_annulus_on_plane(plane_model, inlier_pts):
    if len(inlier_pts)<MIN_INLIERS: return None

    pts_2d, origin, axis_u, axis_v = project_to_plane_2d(inlier_pts, plane_model)

    # angular binning to extract inner/outer boundaries
    inner_pts, outer_pts = extract_boundaries_angular(pts_2d)
    if inner_pts is None: return None

    # fit circle to each boundary
    inner_fit=fit_circle_least_squares(inner_pts)
    outer_fit=fit_circle_least_squares(outer_pts)
    if inner_fit is None or outer_fit is None: return None

    cx_i,cy_i,r_i=inner_fit
    cx_o,cy_o,r_o=outer_fit

    # ensure inner < outer
    if r_i>r_o:
        cx_i,cy_i,r_i,cx_o,cy_o,r_o=cx_o,cy_o,r_o,cx_i,cy_i,r_i

    # validation 1: ratio
    ratio=r_o/r_i if r_i>1e-6 else 999
    if ratio<MIN_ANNULUS_RATIO or ratio>MAX_ANNULUS_RATIO:
        return None

    # validation 2: inner fully inside outer
    center_dist=np.sqrt((cx_i-cx_o)**2+(cy_i-cy_o)**2)
    if center_dist+r_i>r_o*1.08:
        return None

    # use mean center
    cx=(cx_i+cx_o)/2; cy=(cy_i+cy_o)/2
    r_from_center=np.linalg.norm(pts_2d-np.array([cx,cy]),axis=1)

    # validation 3: no points outside outer
    frac_outside=np.mean(r_from_center>r_o)
    if frac_outside>OUTER_OUTLIER_FRAC: return None

    # validation 4: no points inside inner
    frac_inside=np.mean(r_from_center<r_i)
    if frac_inside>INNER_OUTLIER_FRAC: return None

    score=1.0-frac_outside-frac_inside

    return {
        'cx_2d':cx,'cy_2d':cy,
        'r_inner':r_i,'r_outer':r_o,
        'score':score,'ratio':ratio,
        'origin':origin,'axis_u':axis_u,'axis_v':axis_v,
        'plane_model':plane_model,
        'n_inliers':len(inlier_pts),
        'frac_outside':frac_outside,
        'frac_inside':frac_inside,
        'n_bins':len(inner_pts),
    }


# ─── PARALLEL DETECTION ───────────────────────────────────────────────────────

def detect_annulus_parallel(planes):
    if not planes: return None
    results=[]
    with ThreadPoolExecutor(max_workers=MAX_PLANES) as ex:
        futures={ex.submit(detect_annulus_on_plane,pm,ip):i
                 for i,(pm,ip) in enumerate(planes)}
        for f in as_completed(futures):
            r=f.result()
            if r is not None: results.append(r)
    if not results: return None
    return max(results,key=lambda r:r['score'])


# ─── 3D RING BACK-PROJECTION ──────────────────────────────────────────────────

def make_ring_3d_pts(cx,cy,radius,origin,axis_u,axis_v,n_seg=OVERLAY_SEGMENTS):
    angles=np.linspace(0,2*np.pi,n_seg,endpoint=False)
    return np.array([origin+(cx+radius*np.cos(a))*axis_u+(cy+radius*np.sin(a))*axis_v
                     for a in angles])


# ─── LINESET HELPERS ──────────────────────────────────────────────────────────

def make_empty_lineset():
    ls=o3d.geometry.LineSet()
    ls.points=o3d.utility.Vector3dVector(np.zeros((2,3)))
    ls.lines=o3d.utility.Vector2iVector([[0,1]])
    ls.colors=o3d.utility.Vector3dVector([[0,0,0]])
    return ls

def hide_lineset(ls):
    ls.points=o3d.utility.Vector3dVector(np.zeros((2,3)))
    ls.lines=o3d.utility.Vector2iVector([[0,1]])
    ls.colors=o3d.utility.Vector3dVector([[0,0,0]])

def update_ring_lineset(ls, world_pts, color):
    n=len(world_pts)
    ls.points=o3d.utility.Vector3dVector(world_pts)
    ls.lines=o3d.utility.Vector2iVector([[i,(i+1)%n] for i in range(n)])
    ls.colors=o3d.utility.Vector3dVector([color]*n)

def apply_lineset_data(ls, src_ls):
    ls.points=src_ls.points
    ls.lines=src_ls.lines
    ls.colors=src_ls.colors


# ─── CAMERA MARKER ────────────────────────────────────────────────────────────

def make_camera_marker(size=0.08):
    s=size; f=s; b=-s*0.5; hw=s*0.5; hh=s*0.35
    box_pts=np.array([[-hw,-hh,b],[hw,-hh,b],[hw,hh,b],[-hw,hh,b],
                       [-hw,-hh,f],[hw,-hh,f],[hw,hh,f],[-hw,hh,f]])
    box_lines=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    box_colors=[[0.8,0.8,0.2]]*len(box_lines)
    ax_len=s*1.5
    axis_pts=np.array([[0,0,0],[ax_len,0,0],[0,0,0],[0,ax_len,0],[0,0,0],[0,0,ax_len]])
    axis_lines=[[0,1],[2,3],[4,5]]; axis_colors=[[1,0,0],[0,1,0],[0,0,1]]
    fov_h=np.radians(FOV_H_DEG); fov_v=fov_h*(RAY_H/RAY_W); fd=s*2.0
    tx=np.tan(fov_h/2)*fd; ty=np.tan(fov_v/2)*fd
    frust_pts=np.array([[0,0,0],[tx,ty,fd],[-tx,ty,fd],[-tx,-ty,fd],[tx,-ty,fd]])
    frust_lines=[[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    frust_colors=[[0.0,0.8,1.0]]*len(frust_lines)
    all_pts=np.vstack([box_pts,axis_pts,frust_pts])
    ob=0; oa=len(box_pts); of_=oa+len(axis_pts)
    lines=([[l[0]+ob,l[1]+ob] for l in box_lines]+
           [[l[0]+oa,l[1]+oa] for l in axis_lines]+
           [[l[0]+of_,l[1]+of_] for l in frust_lines])
    colors=box_colors+axis_colors+frust_colors
    ls=o3d.geometry.LineSet()
    ls.points=o3d.utility.Vector3dVector(all_pts)
    ls.lines=o3d.utility.Vector2iVector(lines)
    ls.colors=o3d.utility.Vector3dVector(colors)
    return ls, all_pts

def update_camera_marker(ls,base_pts,cam_pos,R):
    ls.points=o3d.utility.Vector3dVector((R@base_pts.T).T+cam_pos)

def make_world_axes(length=0.3):
    pts=np.array([[0,0,0],[length,0,0],[0,0,0],[0,length,0],[0,0,0],[0,0,length]])
    ls=o3d.geometry.LineSet()
    ls.points=o3d.utility.Vector3dVector(pts)
    ls.lines=o3d.utility.Vector2iVector([[0,1],[2,3],[4,5]])
    ls.colors=o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return ls


# ─── CAMERA STATE ─────────────────────────────────────────────────────────────

class CameraState:
    def __init__(self): self.reset()
    def reset(self):
        self.pos=np.array([0.0,-2.0,0.0])
        self.yaw=0.0; self.pitch=-90.0
    @property
    def R(self):       return rotation_matrix(self.yaw,self.pitch)
    @property
    def forward(self): return self.R@np.array([0,0,1])
    @property
    def right(self):   return self.R@np.array([1,0,0])
    @property
    def up(self):      return self.R@np.array([0,1,0])


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading STL: {STL_PATH}")
    try:
        mesh=o3d.io.read_triangle_mesh(STL_PATH)
    except Exception as e:
        print(f"ERROR: {e}"); sys.exit(1)
    if not mesh.has_triangles():
        print("ERROR: no triangles."); sys.exit(1)

    mesh.compute_vertex_normals()
    bb=mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bb.get_center())
    mesh.scale(2.0/max(bb.get_extent()),center=[0,0,0])
    mesh.paint_uniform_color([0.45,0.45,0.5])
    print(f"Mesh loaded. Verts={len(mesh.vertices)}, Tris={len(mesh.triangles)}")
    print(f"Ring face: Y~-0.171 | Inner r~0.070 | Outer r~0.099 | Ratio~1.41x")

    mesh_t=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    print(f"Scene built. Ray grid: {RAY_W}x{RAY_H}")

    ray_dirs_cam=build_ray_grid(FOV_H_DEG,RAY_W,RAY_H)
    cam=CameraState(); lock=threading.Lock(); running=[True]; detecting=[False]
    latest_pts=[np.zeros((0,3))]
    frame_dirty=threading.Event(); detect_dirty=threading.Event()

    overlay_lock=threading.Lock()
    plane_overlay_data=[None]*MAX_PLANES
    ring_inner_pts=[None]; ring_outer_pts=[None]
    overlay_dirty=[False]

    # ── Raycast thread ────────────────────────────────────────────────────────
    def raycast_loop():
        while running[0]:
            with lock: pos=cam.pos.copy(); R=cam.R.copy()
            pts=cast_frame(scene,pos,R,ray_dirs_cam)
            with lock: latest_pts[0]=pts
            frame_dirty.set()
            if detecting[0]: detect_dirty.set()
            time.sleep(0.033)

    threading.Thread(target=raycast_loop,daemon=True).start()

    # ── Detection thread ──────────────────────────────────────────────────────
    last_log=[0.0]

    def detection_loop():
        while running[0]:
            detect_dirty.wait(timeout=0.5)
            detect_dirty.clear()
            if not detecting[0]: time.sleep(0.05); continue

            with lock: pts=latest_pts[0].copy()
            if len(pts)<MIN_INLIERS: continue

            planes=extract_planes(pts)

            # build plane rect data
            new_plane_data=[None]*MAX_PLANES
            for i,(pm,ip) in enumerate(planes):
                src=make_plane_rect_lineset(pm,ip,PLANE_COLORS[i%len(PLANE_COLORS)])
                new_plane_data[i]=(
                    np.asarray(src.points).copy(),
                    np.asarray(src.lines).tolist(),
                    np.asarray(src.colors).tolist(),
                )

            result=detect_annulus_parallel(planes)

            with overlay_lock:
                for i in range(MAX_PLANES): plane_overlay_data[i]=new_plane_data[i]
                if result:
                    ring_inner_pts[0]=make_ring_3d_pts(
                        result['cx_2d'],result['cy_2d'],result['r_inner'],
                        result['origin'],result['axis_u'],result['axis_v'])
                    ring_outer_pts[0]=make_ring_3d_pts(
                        result['cx_2d'],result['cy_2d'],result['r_outer'],
                        result['origin'],result['axis_u'],result['axis_v'])
                else:
                    ring_inner_pts[0]=None; ring_outer_pts[0]=None
                overlay_dirty[0]=True

            now=time.time()
            if now-last_log[0]>LOG_INTERVAL_SEC:
                last_log[0]=now
                print("\n"+"─"*60)
                print(f"  Planes found : {len(planes)}")
                for i,(pm,ip) in enumerate(planes):
                    a,b,c,d=pm
                    cname=['Yellow','Magenta','Green','White','Orange'][i]
                    print(f"  Plane {i} [{cname:7s}]: n={len(ip):4d}  "
                          f"normal=({a:+.2f},{b:+.2f},{c:+.2f})")
                if result:
                    print(f"  ANNULUS : DETECTED")
                    print(f"  Center  : ({result['cx_2d']:+.4f}, {result['cy_2d']:+.4f})")
                    print(f"  R inner : {result['r_inner']:.4f}  (expected ~0.070)")
                    print(f"  R outer : {result['r_outer']:.4f}  (expected ~0.099)")
                    print(f"  Ratio   : {result['ratio']:.3f}x  (expected ~1.41x)")
                    print(f"  Score   : {result['score']*100:.1f}%  bins={result['n_bins']}")
                    print(f"  Frac out: {result['frac_outside']*100:.1f}%  in: {result['frac_inside']*100:.1f}%")
                else:
                    print(f"  ANNULUS : NO DETECTION")
                print("─"*60)

            time.sleep(DETECT_INTERVAL_SEC)

    threading.Thread(target=detection_loop,daemon=True).start()

    # ── Open3D viewer ─────────────────────────────────────────────────────────
    vis=o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Ring Sim — Angular Bin Detection",width=1280,height=720)
    opt=vis.get_render_option()
    opt.background_color=np.array(BG_COLOR); opt.point_size=POINT_SIZE
    opt.line_width = 30.0 
    opt.mesh_show_back_face=True; opt.light_on=True

    vis.add_geometry(mesh)
    vis.add_geometry(make_world_axes(0.3))
    cam_marker,base_pts=make_camera_marker(size=0.08)
    vis.add_geometry(cam_marker)
    pcd=o3d.geometry.PointCloud(); vis.add_geometry(pcd)

    plane_ls=[make_empty_lineset() for _ in range(MAX_PLANES)]
    for ls in plane_ls: vis.add_geometry(ls)

    ring_inner_ls=make_empty_lineset(); ring_outer_ls=make_empty_lineset()
    vis.add_geometry(ring_inner_ls); vis.add_geometry(ring_outer_ls)

    last_press_time={}
    def make_cb(key):
        def cb(v): last_press_time[key]=time.time(); return False
        return cb
    for k in MOVE_KEYS: vis.register_key_callback(k,make_cb(k))

    def cb_r(v):
        with lock: cam.reset(); return False

    def cb_j(v):
        detecting[0]=not detecting[0]
        if detecting[0]:
            print(f"\n[J] Detection ENABLED — angular binning method")
        else:
            print("\n[J] Detection DISABLED.")
            with overlay_lock:
                for i in range(MAX_PLANES): plane_overlay_data[i]=None
                ring_inner_pts[0]=None; ring_outer_pts[0]=None
                overlay_dirty[0]=True
        return False

    def cb_esc(v):
        running[0]=False; v.close(); return False

    vis.register_key_callback(KEY_R,cb_r)
    vis.register_key_callback(KEY_J,cb_j)
    vis.register_key_callback(KEY_ESC,cb_esc)

    def on_animation(v):
        now=time.time()
        def active(k): return (now-last_press_time.get(k,0))<KEY_HOLD_WINDOW
        with lock:
            if active(KEY_W):    cam.pos+=MOVE_SPEED*cam.forward
            if active(KEY_S):    cam.pos-=MOVE_SPEED*cam.forward
            if active(KEY_A):    cam.pos-=MOVE_SPEED*cam.right
            if active(KEY_D):    cam.pos+=MOVE_SPEED*cam.right
            if active(KEY_Q):    cam.pos-=MOVE_SPEED*cam.up
            if active(KEY_E):    cam.pos+=MOVE_SPEED*cam.up
            if active(KEY_UP):   cam.pitch-=LOOK_SPEED_DEG
            if active(KEY_DOWN): cam.pitch+=LOOK_SPEED_DEG
            if active(KEY_LEFT): cam.yaw-=LOOK_SPEED_DEG
            if active(KEY_RIGHT):cam.yaw+=LOOK_SPEED_DEG
            pos=cam.pos.copy(); R=cam.R.copy()

        update_camera_marker(cam_marker,base_pts,pos,R)
        v.update_geometry(cam_marker)

        if frame_dirty.is_set():
            frame_dirty.clear()
            with lock: pts=latest_pts[0].copy()
            if len(pts)>0:
                pcd.points=o3d.utility.Vector3dVector(pts)
                pcd.paint_uniform_color(POINT_COLOR)
            else:
                pcd.points=o3d.utility.Vector3dVector(np.zeros((0,3)))
            v.update_geometry(pcd)

        with overlay_lock:
            dirty=overlay_dirty[0]
            if dirty:
                pd=[plane_overlay_data[i] for i in range(MAX_PLANES)]
                ip=ring_inner_pts[0]; op=ring_outer_pts[0]
                overlay_dirty[0]=False

        if dirty:
            for i,ls in enumerate(plane_ls):
                if pd[i] is not None:
                    pts_,lines_,colors_=pd[i]
                    ls.points=o3d.utility.Vector3dVector(pts_)
                    ls.lines=o3d.utility.Vector2iVector(lines_)
                    ls.colors=o3d.utility.Vector3dVector(colors_)
                else:
                    hide_lineset(ls)
                v.update_geometry(ls)

            if ip is not None and len(ip)>1:
                update_ring_lineset(ring_inner_ls,ip,INNER_COLOR)
                update_ring_lineset(ring_outer_ls,op,OUTER_COLOR)
            else:
                hide_lineset(ring_inner_ls)
                hide_lineset(ring_outer_ls)
            v.update_geometry(ring_inner_ls)
            v.update_geometry(ring_outer_ls)

        return False

    vis.register_animation_callback(on_animation)

    print("\n=== Controls ===")
    print("  W/S/A/D/Q/E : move")
    print("  Arrow keys  : look")
    print("  R           : reset (below ring looking up)")
    print("  J           : toggle detection")
    print("  ESC         : quit")
    print("\nPlane colors: Yellow | Magenta | Green | White | Orange")
    print("White arrow = plane normal direction")
    print("CYAN=inner ring  ORANGE=outer ring")
    print("\n>>> Click inside the window first <<<\n")

    vis.run()
    vis.destroy_window()
    running[0]=False


if __name__=="__main__":
    main()