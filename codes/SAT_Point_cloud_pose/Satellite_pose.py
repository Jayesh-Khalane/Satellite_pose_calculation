import pyzed.sl as sl
import open3d as o3d
import numpy as np
import copy
import time
import sys
import os
from datetime import datetime

# ─── Settings ────────────────────────────────────────────────────────────────
ZED0_SN    = 33140394
MAX_DEPTH  = 100       # cm
VOXEL      = 0.3       # cm — registration resolution
VOXEL_FINE = 0.2       # cm — export resolution
DELAY      = 10         # seconds between captures

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
    """RANSAC global + point-to-plane ICP. Returns 4x4 transform."""
    src = prep(source_raw)
    tgt = prep(target_raw)

    # FPFH features
    r = VOXEL * 5
    sf = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    tf = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))

    # RANSAC — robust global alignment
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

    T = ransac.transformation
    log("REG", f"RANSAC fitness={ransac.fitness:.3f}")

    # Point-to-plane ICP refinement
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, VOXEL * 1.5, T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    log("REG", f"ICP    fitness={icp.fitness:.3f}  rmse={icp.inlier_rmse:.4f}")
    return icp.transformation

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    zed  = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(ZED0_SN)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode        = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units  = sl.UNIT.CENTIMETER

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        log("ERR", "Cannot open ZED"); return

    vis = o3d.visualization.Visualizer()
    vis.create_window("Scanner", 1280, 720)
    render    = o3d.geometry.PointCloud()
    geo_added = False

    model = o3d.geometry.PointCloud()  # accumulated
    mat   = sl.Mat()
    last_t = time.time()
    n = 0

    print(f"\n  SCANNER — depth < {MAX_DEPTH} cm | Ctrl+C to stop & save\n")

    try:
        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            elapsed = time.time() - last_t
            sys.stdout.write(f"\r  Scans: {n} | next in {max(0, int(DELAY - elapsed))}s   ")
            sys.stdout.flush()
            vis.poll_events(); vis.update_renderer()

            if elapsed <= DELAY:
                continue

            pcd = grab(zed, mat)
            last_t = time.time()

            if pcd is None:
                log("SKIP", "too few points"); continue

            log("CAP", f"{len(pcd.points):,} pts")

            if n == 0:
                model += copy.deepcopy(pcd)
                n += 1
                d = model.voxel_down_sample(VOXEL)
                render.points = d.points; render.colors = d.colors
                vis.add_geometry(render); geo_added = True
                log("OK", "Anchor set — rotate object ~30° and wait")
                continue

            # Register new scan against full accumulated model
            T = align(pcd, model)

            tmp = copy.deepcopy(pcd)
            tmp.transform(T)
            model += tmp
            model = model.voxel_down_sample(VOXEL)  # keep model lean
            n += 1

            render.points = model.points; render.colors = model.colors
            if geo_added:
                vis.update_geometry(render)
            else:
                vis.add_geometry(render); geo_added = True

            log("OK", f"Scan {n} merged — model {len(model.points):,} pts")

    except KeyboardInterrupt:
        print(f"\n\nDone — {n} scans.\n")

    finally:
        zed.close()
        if geo_added:
            vis.destroy_window()
        if n < 2:
            log("ERR", "Need >= 2 scans"); return

        # Final cleanup
        model = model.voxel_down_sample(VOXEL_FINE)
        model, _ = model.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        model.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_FINE * 3, max_nn=30))
        model.orient_normals_towards_camera_location(np.zeros(3))

        os.makedirs("logs/data", exist_ok=True)
        out = "logs/data/stitched_satellite.ply"
        o3d.io.write_point_cloud(out, model, write_ascii=False)
        log("DONE", f"{out} — {len(model.points):,} pts from {n} scans")

if __name__ == "__main__":
    main()