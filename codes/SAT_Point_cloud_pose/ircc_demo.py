import os, sys

# ── Suppress Open3D / ZED C++ warnings ───────────────────────────────────────
_nul  = os.open(os.devnull, os.O_WRONLY)
_serr = os.dup(2)
os.dup2(_nul, 2)
os.close(_nul)

import pyzed.sl as sl
import open3d as o3d

os.dup2(_serr, 2)
os.close(_serr)

import numpy as np
import time
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation

# ─── Settings ────────────────────────────────────────────────────────────────
ZED0_SN   = 33140394
MAX_DEPTH = 130      # cm
VOXEL     = 0.3     # cm
REF_PATH  = "logs/data/reference_stitched_satellite.ply"
DELAY     = 0       # seconds between pose updates
EMA_ALPHA = 0.25    # EMA smoothing: 0.1=very smooth/laggy  0.4=snappier  1.0=raw

# ─── ANSI helpers ────────────────────────────────────────────────────────────
# Enable VT100 on Windows
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

HOME    = "\033[H"           # cursor to top-left
CLEAR   = "\033[2J"         # clear screen
HIDE    = "\033[?25l"       # hide cursor
SHOW    = "\033[?25h"       # show cursor

CY  = "\033[96m"            # cyan
GR  = "\033[92m"            # green
YL  = "\033[93m"            # yellow
WH  = "\033[97m"            # white
BL  = "\033[94m"            # blue
MG  = "\033[95m"            # magenta
DIM = "\033[2m"             # dim
BLD = "\033[1m"             # bold
RST = "\033[0m"             # reset

def clr():
    sys.stdout.write(CLEAR + HOME)

def draw(frame, fps, fitness, rmse, x, y, z, roll, pitch, yaw, status):
    fit_c  = GR if fitness > 0.4 else YL
    rmse_c = GR if rmse < VOXEL  else YL

    lines = [
        f"{CY}{BLD}  ◈  LIVE 6-DOF POSE DASHBOARD  ◈{RST}",
        f"{BL}  {'─'*52}{RST}",
        f"  {WH}Frame{RST}  {GR}{BLD}{frame:06d}{RST}    {WH}FPS{RST}  {GR}{BLD}{fps:5.1f}{RST}",
        f"{BL}  {'─'*52}{RST}",
        f"  {CY}REGISTRATION QUALITY{RST}",
        f"  {WH}{'Fitness':<14}{RST}{fit_c}{BLD}{fitness:8.4f}{RST}  {DIM}(↑ higher=better){RST}",
        f"  {WH}{'RMSE':<14}{RST}{rmse_c}{BLD}{rmse:8.4f}{RST}  {DIM}cm  (↓ lower=better){RST}",
        f"{BL}  {'─'*52}{RST}",
        f"  {CY}TRANSLATION{RST}",
        f"  {WH}{'X':<14}{RST}{GR}{BLD}{x:+12.4f}{RST}  {DIM}cm{RST}",
        f"  {WH}{'Y':<14}{RST}{GR}{BLD}{y:+12.4f}{RST}  {DIM}cm{RST}",
        f"  {WH}{'Z':<14}{RST}{GR}{BLD}{z:+12.4f}{RST}  {DIM}cm{RST}",
        f"{BL}  {'─'*52}{RST}",
        f"  {CY}ORIENTATION  (Euler XYZ){RST}",
        f"  {WH}{'Roll  (X)':<14}{RST}{GR}{BLD}{roll:+11.4f}{RST}  {DIM}deg{RST}",
        f"  {WH}{'Pitch (Y)':<14}{RST}{GR}{BLD}{pitch:+11.4f}{RST}  {DIM}deg{RST}",
        f"  {WH}{'Yaw   (Z)':<14}{RST}{GR}{BLD}{yaw:+11.4f}{RST}  {DIM}deg{RST}",
        f"{BL}  {'─'*52}{RST}",
        f"  {MG}{status}{RST}",
        f"{BL}  {'─'*52}{RST}",
        f"  {DIM}RED=Reference   BLUE=Live   GREEN=Aligned   [Q] quit{RST}",
    ]

    sys.stdout.write(HOME)
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

# ─── Capture ─────────────────────────────────────────────────────────────────

def grab_pcd(zed, mat):
    zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
    d    = mat.get_data()
    xyz  = d[:, :, :3].reshape(-1, 3)
    rgba = d[:, :, 3].reshape(-1)

    ok = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) < MAX_DEPTH)
    xyz, rgba = xyz[ok], rgba[ok]
    if len(xyz) < 500:
        return None

    ri  = rgba.view(np.uint32)
    rgb = np.stack([((ri >> 0) & 255), ((ri >> 8) & 255), ((ri >> 16) & 255)], axis=1) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def grab_image(zed, img_mat):
    zed.retrieve_image(img_mat, sl.VIEW.LEFT)
    return cv2.cvtColor(img_mat.get_data(), cv2.COLOR_BGRA2BGR)

# ─── Registration ────────────────────────────────────────────────────────────

def prep(pcd):
    d = pcd.voxel_down_sample(VOXEL)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 3, max_nn=30))
    d.orient_normals_towards_camera_location(np.zeros(3))
    return d

def align(source_raw, target_raw):
    src = prep(source_raw)
    tgt = prep(target_raw)

    r  = VOXEL * 5
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

# ─── Pose ────────────────────────────────────────────────────────────────────

def extract_pose(T):
    rot = Rotation.from_matrix(T[:3, :3])
    roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
    return T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw

# ─── CV2 overlay ─────────────────────────────────────────────────────────────

def overlay_pose(img, T, fitness, rmse, frame):
    x, y, z, roll, pitch, yaw = extract_pose(T)
    lines = [
        f"Frame {frame:05d}   Fit={fitness:.3f}   RMSE={rmse:.4f}",
        f"X:{x:+8.2f}  Y:{y:+8.2f}  Z:{z:+8.2f} cm",
        f"R:{roll:+7.2f}  P:{pitch:+7.2f}  Yaw:{yaw:+7.2f} deg",
    ]
    font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 8
    for i, ln in enumerate(lines):
        yt = 28 + i * 24
        cv2.putText(img, ln, (pad+1, yt+1), font, scale, (0, 0, 0),     thick+1, cv2.LINE_AA)
        cv2.putText(img, ln, (pad,   yt),   font, scale, (0, 255, 120), thick,   cv2.LINE_AA)

def overlay_status(img, frame, msg):
    font, scale, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 8
    ln = f"Frame {frame:05d}  -  {msg}"
    cv2.putText(img, ln, (pad+1, 29), font, scale, (0,  0,  0),    2, cv2.LINE_AA)
    cv2.putText(img, ln, (pad,   28), font, scale, (80, 200, 255), 1, cv2.LINE_AA)

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(REF_PATH):
        print(f"[ERR] Reference not found: {REF_PATH}"); return

    ref = o3d.io.read_point_cloud(REF_PATH)
    ref.paint_uniform_color([0.8, 0.1, 0.1])

    zed  = sl.Camera()
    ip   = sl.InitParameters()
    ip.set_from_serial_number(ZED0_SN)
    ip.camera_resolution = sl.RESOLUTION.HD1080
    ip.depth_mode        = sl.DEPTH_MODE.NEURAL_PLUS
    ip.coordinate_units  = sl.UNIT.CENTIMETER

    if zed.open(ip) != sl.ERROR_CODE.SUCCESS:
        print("[ERR] Cannot open ZED camera"); return

    img_mat = sl.Mat()
    pc_mat  = sl.Mat()

    vis = o3d.visualization.Visualizer()
    vis.create_window("RED=Reference  BLUE=Live  GREEN=Aligned", 1280, 720)
    vis.add_geometry(ref)

    live_render    = o3d.geometry.PointCloud()
    aligned_render = o3d.geometry.PointCloud()
    live_added     = False
    aligned_added  = False

    cv2.namedWindow("ZED 720p", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED 720p", 1280, 720)

    frame      = 0
    last_t     = time.time()
    fps        = 0.0
    last_T     = None
    last_fit   = 0.0
    last_rmse  = 0.0
    last_pose  = (0.0,) * 6
    ema_pose   = None          # EMA state: np.array of 6 values, init on first frame

    sys.stdout.write(CLEAR + HIDE)
    draw(0, 0.0, 0.0, 0.0, *last_pose, "Initialising...")

    try:
        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            bgr = grab_image(zed, img_mat)
            vis.poll_events()
            vis.update_renderer()

            elapsed = time.time() - last_t

            if elapsed <= DELAY:
                if last_T is not None:
                    overlay_pose(bgr, last_T, last_fit, last_rmse, frame)
                else:
                    overlay_status(bgr, frame, "Waiting...")
                cv2.imshow("ZED 720p", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            pcd = grab_pcd(zed, pc_mat)
            now = time.time()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now

            if pcd is None:
                draw(frame, fps, last_fit, last_rmse, *last_pose,
                     "WARNING  Too few points - move camera closer")
                overlay_status(bgr, frame, "Too few points")
                cv2.imshow("ZED 720p", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            frame += 1
            draw(frame, fps, last_fit, last_rmse, *last_pose,
                 "Computing RANSAC + ICP...")

            T, fitness, rmse = align(pcd, ref)

            last_T    = T
            last_fit  = fitness
            last_rmse = rmse

            # Reject low-quality frames entirely
            if fitness < 0.6:
                draw(frame, fps, last_fit, last_rmse, *last_pose,
                     f"SKIP  fitness={fitness:.3f} < 0.6  (frame rejected)")
                overlay_status(bgr, frame, f"Low fitness {fitness:.3f} — skipped")
                cv2.imshow("ZED 720p", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            raw_pose = np.array(extract_pose(T))
            if ema_pose is None:
                ema_pose = raw_pose.copy()
            else:
                ema_pose = EMA_ALPHA * raw_pose + (1.0 - EMA_ALPHA) * ema_pose
            last_pose = tuple(ema_pose)

            draw(frame, fps, fitness, rmse, *last_pose,
                 f"OK   {datetime.now().strftime('%H:%M:%S')}")

            # BLUE: raw live
            d_live = pcd.voxel_down_sample(VOXEL)
            live_render.points = d_live.points
            live_render.paint_uniform_color([0.1, 0.3, 0.9])
            if live_added:
                vis.update_geometry(live_render)
            else:
                vis.add_geometry(live_render)
                live_added = True

            # GREEN: aligned
            d_aligned = o3d.geometry.PointCloud(d_live)
            d_aligned.transform(T)
            aligned_render.points = d_aligned.points
            aligned_render.paint_uniform_color([0.2, 0.85, 0.2])
            if aligned_added:
                vis.update_geometry(aligned_render)
            else:
                vis.add_geometry(aligned_render)
                aligned_added = True

            overlay_pose(bgr, T, fitness, rmse, frame)
            cv2.imshow("ZED 720p", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        sys.stdout.write(SHOW)
        zed.close()
        vis.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()