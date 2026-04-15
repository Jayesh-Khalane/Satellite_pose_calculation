import pyzed.sl as sl
import open3d as o3d
import numpy as np
import time
import sys
import os
from datetime import datetime
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from collections import deque

# ─── Settings ────────────────────────────────────────────────────────────────
ZED0_SN    = 33140394
MAX_DEPTH  = 100       # cm
VOXEL      = 0.3       # cm — registration resolution
REF_PATH   = "logs/data/reference_stitched_satellite.ply"
DELAY      = 0         # seconds between pose updates
HISTORY_LN = 100       # Number of frames to keep in the live charts

def log(tag, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}")

# ─── Capture ─────────────────────────────────────────────────────────────────

def grab(zed, mat):
    zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
    d = mat.get_data()
    xyz  = d[:, :, :3].reshape(-1, 3)
    rgba = d[:, :, 3].reshape(-1)

    ok = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) < MAX_DEPTH)
    xyz, rgba = xyz[ok], rgba[ok]
    if len(xyz) < 500:
        return None

    ri = rgba.view(np.uint32)
    rgb = np.stack([((ri >> 0) & 255), ((ri >> 8) & 255), ((ri >> 16) & 255)], axis=1) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

# ─── Registration ────────────────────────────────────────────────────────────

def prep(pcd):
    d = pcd.voxel_down_sample(VOXEL)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 3, max_nn=30))
    d.orient_normals_towards_camera_location(np.zeros(3))
    return d

def align(source_raw, target_raw):
    """RANSAC + point-to-plane ICP. Returns (4x4 transform, fitness, rmse)."""
    src = prep(source_raw)
    tgt = prep(target_raw)

    r = VOXEL * 5
    sf = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    tf = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, sf, tf,
        mutual_filter=True,
        max_correspondence_distance=VOXEL * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL * 2),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, VOXEL * 1.5, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    return icp.transformation, icp.fitness, icp.inlier_rmse

# ─── Pose extraction ─────────────────────────────────────────────────────────

def extract_pose(T):
    R = T[:3, :3]
    t = T[:3, 3]
    rot = Rotation.from_matrix(R)
    roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
    return t[0], t[1], t[2], roll, pitch, yaw

def print_pose(T, fitness, rmse, centroid, frame):
    x, y, z, roll, pitch, yaw = extract_pose(T)
    cx, cy, cz = centroid

    print(f"\n{'━'*60}")
    print(f"  FRAME {frame:04d}                    fitness={fitness:.3f}  rmse={rmse:.4f}")
    print(f"{'━'*60}")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  6-DOF POSE (live → reference)                      │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Translation                                        │")
    print(f"  │    X : {x:+10.3f} cm                                │")
    print(f"  │    Y : {y:+10.3f} cm                                │")
    print(f"  │    Z : {z:+10.3f} cm                                │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Orientation (Euler XYZ)                            │")
    print(f"  │    Roll  (X) : {roll:+8.2f}°                            │")
    print(f"  │    Pitch (Y) : {pitch:+8.2f}°                            │")
    print(f"  │    Yaw   (Z) : {yaw:+8.2f}°                            │")
    print(f"  └─────────────────────────────────────────────────────┘")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(REF_PATH):
        log("ERR", f"Reference not found: {REF_PATH}"); return

    # Load reference and paint it RED
    ref = o3d.io.read_point_cloud(REF_PATH)
    ref.paint_uniform_color([0.8, 0.1, 0.1]) 
    log("REF", f"Loaded {REF_PATH} — {len(ref.points):,} pts")

    # Open camera
    zed  = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(ZED0_SN)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode        = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units  = sl.UNIT.CENTIMETER

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        log("ERR", "Cannot open ZED"); return

    # --- Dashboard Setup: Matplotlib Charts ---
    plt.ion() # Interactive mode on
    fig, (ax_fit, ax_rmse) = plt.subplots(2, 1, figsize=(6, 5))
    fig.canvas.manager.set_window_title('Optimization Metrics')
    
    frames_history = deque(maxlen=HISTORY_LN)
    fitness_history = deque(maxlen=HISTORY_LN)
    rmse_history = deque(maxlen=HISTORY_LN)
    
    line_fit, = ax_fit.plot([], [], 'g-', label='Fitness (Higher is better)')
    line_rmse, = ax_rmse.plot([], [], 'r-', label='RMSE (Lower is better)')
    
    ax_fit.set_xlim(0, HISTORY_LN); ax_fit.set_ylim(0, 1.0)
    ax_rmse.set_xlim(0, HISTORY_LN); ax_rmse.set_ylim(0, VOXEL * 3)
    ax_fit.legend(); ax_fit.grid(True)
    ax_rmse.legend(); ax_rmse.grid(True)
    plt.tight_layout()

    # --- Dashboard Setup: Open3D Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Pose Optimization", 1280, 720)
    
    vis.add_geometry(ref) # Add RED reference cloud permanently
    
    # Placeholder for aligned live data
    aligned_render = o3d.geometry.PointCloud() # GREEN (Live snapped to Reference)
    geo_added = False

    mat    = sl.Mat()
    last_t = time.time()
    frame  = 0

    print(f"\n{'='*60}")
    print(f"  LIVE POSE DASHBOARD STARTING")
    print(f"  RED   = Reference Cloud")
    print(f"  GREEN = Optimized Alignment")
    print(f"{'='*60}\n")

    try:
        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            elapsed = time.time() - last_t
            vis.poll_events(); vis.update_renderer()
            
            # Keep matplotlib GUI responsive
            fig.canvas.flush_events() 

            if elapsed <= DELAY:
                continue

            pcd = grab(zed, mat)
            last_t = time.time()

            if pcd is None:
                sys.stdout.write(f"\r  Frame {frame} — too few points...   ")
                sys.stdout.flush()
                continue

            frame += 1

            # Compute centroid of live data
            pts = np.asarray(pcd.points)
            centroid = pts.mean(axis=0)

            # 1. Align live (source) → reference (target)
            T, fitness, rmse = align(pcd, ref)
            print_pose(T, fitness, rmse, centroid, frame)

            # 2. Update Matplotlib Charts
            frames_history.append(frame)
            fitness_history.append(fitness)
            rmse_history.append(rmse)
            
            line_fit.set_data(range(len(frames_history)), fitness_history)
            line_rmse.set_data(range(len(frames_history)), rmse_history)
            
            ax_fit.set_xlim(max(0, len(frames_history) - HISTORY_LN), max(HISTORY_LN, len(frames_history)))
            ax_rmse.set_xlim(max(0, len(frames_history) - HISTORY_LN), max(HISTORY_LN, len(frames_history)))
            fig.canvas.draw()

            # 3. Update Open3D Visualization
            d_live = pcd.voxel_down_sample(VOXEL)
            
            # Update GREEN cloud (Aligned Live)
            d_aligned = o3d.geometry.PointCloud(d_live) # copy
            d_aligned.transform(T) # Apply ICP matrix
            aligned_render.points = d_aligned.points
            aligned_render.paint_uniform_color([0.2, 0.8, 0.2])

            if geo_added:
                vis.update_geometry(aligned_render)
            else:
                vis.add_geometry(aligned_render)
                geo_added = True

    except KeyboardInterrupt:
        print(f"\n\nStopped after {frame} frames.\n")

    finally:
        zed.close()
        vis.destroy_window()
        plt.close(fig)

if __name__ == "__main__":
    main()