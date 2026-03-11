import pyzed.sl as sl
import open3d as o3d
import numpy as np
import time
import csv
import itertools
from scipy.spatial.transform import Rotation as R_tool

# --- CONFIGURATION ---
CAPTURE_COUNT = 3
COUNTDOWN_SEC = 3.0
MAX_DIST_CM = 150.0  # 1.5 meters

# --- MATH HELPERS ---
def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def get_zed0_pose_from_zed1(pc_data):
    """ LED tracking logic to find the Pose of ZED 0 as seen by ZED 1 """
    mask = (np.sum(pc_data[:, :, :3], axis=2) > 750) & (~np.isnan(pc_data[:, :, 0]))
    xyz_all = pc_data[mask][:, :3]
    if len(xyz_all) < 10: return None, None

    found_leds = []
    temp_xyz = xyz_all.copy()
    for _ in range(3):
        if len(temp_xyz) == 0: break
        seed_pt = temp_xyz[0]
        dists = np.linalg.norm(temp_xyz - seed_pt, axis=1)
        cluster_mask = dists <= 10.0 
        found_leds.append(np.mean(temp_xyz[cluster_mask], axis=0))
        temp_xyz = temp_xyz[~cluster_mask]
    
    if len(found_leds) != 3: return None, None

    idx_pairs = list(itertools.combinations(range(3), 2))
    d_list = [{'pair': (i, j), 'dist': get_dist(found_leds[i], found_leds[j])} for i, j in idx_pairs]
    d_list.sort(key=lambda x: x['dist'], reverse=True)
    L1, L2 = d_list[0]['pair'], d_list[1]['pair']
    
    common = set(L1).intersection(set(L2))
    if not common: return None, None
    p1_idx = list(common)[0] 
    p2_idx = L2[0] if L2[0] != p1_idx else L2[1] 
    p3_idx = L1[0] if L1[0] != p1_idx else L1[1] 
    
    p1, p2, p3 = found_leds[p1_idx], found_leds[p2_idx], found_leds[p3_idx]

    x_axis = (p2 - p1) / np.linalg.norm(p2 - p1)
    z_axis = np.cross(p2 - p1, p3 - p1)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    r_matrix = np.column_stack((x_axis, y_axis, z_axis))
    centroid = np.mean(found_leds, axis=0) 
    
    return r_matrix, centroid

def main():
    # --- 1. INITIALIZE CAMERAS ---
    cam0, cam1 = sl.Camera(), sl.Camera()
    
    # Handheld Camera (Scanner) - ZED 0
    init0 = sl.InitParameters()
    init0.set_from_camera_id(0) 
    init0.camera_resolution = sl.RESOLUTION.HD1080
    init0.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init0.coordinate_units = sl.UNIT.CENTIMETER
    init0.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    # Ceiling Camera (Tracker) - ZED 1
    init1 = sl.InitParameters()
    init1.set_from_camera_id(1)
    init1.camera_resolution = sl.RESOLUTION.HD1080
    init1.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init1.coordinate_units = sl.UNIT.CENTIMETER
    init1.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    print("Opening ZED 0 and ZED 1 at 1080p...")
    if cam0.open(init0) != sl.ERROR_CODE.SUCCESS or cam1.open(init1) != sl.ERROR_CODE.SUCCESS:
        print("Initialization failed.")
        return

    # --- 2. INITIALIZE OPEN3D ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("ZED 0 Live Preview (<1.5m)", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    geom_added = False
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    pc0, pc1 = sl.Mat(), sl.Mat()
    runtime = sl.RuntimeParameters()
    
    all_data = []
    snapshot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    
    captures_done = 0
    last_capture_time = time.time()
    last_print_time = 0

    print("\n--- FULL DENSITY LIVE DATA COLLECTION ---")

    try:
        while captures_done < CAPTURE_COUNT:
            if cam0.grab(runtime) == sl.ERROR_CODE.SUCCESS and cam1.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam0.retrieve_measure(pc0, sl.MEASURE.XYZRGBA)
                cam1.retrieve_measure(pc1, sl.MEASURE.XYZRGBA)
                
                # --- PROCESS ZED 0 CLOUD ---
                pc0_np = pc0.get_data()
                xyz = pc0_np[:, :, :3].reshape(-1, 3)
                
                # Filter valid points
                valid_mask = np.isfinite(xyz).all(axis=1)
                xyz_valid = xyz[valid_mask]
                
                # Filter by Distance (< 150 cm)
                distances = np.linalg.norm(xyz_valid, axis=1)
                close_mask = distances < MAX_DIST_CM
                xyz_filtered = xyz_valid[close_mask]

                # Extract RGB Colors for valid, close points
                rgba_float = pc0_np[:, :, 3].reshape(-1)[valid_mask][close_mask]
                rgba_int = rgba_float.view(np.uint32)
                r = ((rgba_int) & 0xFF).astype(np.uint8)
                g = ((rgba_int >> 8) & 0xFF).astype(np.uint8)
                b = ((rgba_int >> 16) & 0xFF).astype(np.uint8)
                colors = np.stack([r, g, b], axis=1) / 255.0

                # --- TIMER LOGIC (Non-Blocking) ---
                current_time = time.time()
                elapsed = current_time - last_capture_time
                remaining = int(COUNTDOWN_SEC - elapsed)

                if int(current_time) != last_print_time:
                    print(f"SNAPSHOT {captures_done+1}: Capture in {max(0, remaining)}s... (Preview Active)", end="\r")
                    last_print_time = int(current_time)

                # TRIGGER CAPTURE
                if elapsed >= COUNTDOWN_SEC:
                    R_cam, P_cam = get_zed0_pose_from_zed1(pc1.get_data())
                    
                    if R_cam is None:
                        print(f"\n[!] ERROR: Tracker lost sight of LEDs. Retrying Snapshot {captures_done+1}...")
                        last_capture_time = time.time() 
                        continue

                    # Transform the filtered cloud
                    transformed_pts = (xyz_filtered @ R_cam.T) + P_cam

                    # Store points with the assigned snapshot color
                    r_col, g_col, b_col = snapshot_colors[captures_done]
                    new_points = [[p[0], p[1], p[2], r_col, g_col, b_col] for p in transformed_pts]
                    all_data.extend(new_points)

                    captures_done += 1
                    print(f"\n[+] SUCCESS: Snapshot {captures_done} Captured!")
                    print(f"    - Points Saved: {len(transformed_pts)}")
                    print(f"    - Pose (P): {np.round(P_cam, 2)}")
                    last_capture_time = time.time() 

                # --- OPEN3D LIVE PREVIEW UPDATE ---
                # Pushing the full filtered density straight to the viewer
                pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

    except KeyboardInterrupt:
        print("\n[!] Sequence interrupted by user.")

    # --- SAVE TO CSV ---
    if all_data:
        filename = "stitched.csv"
        print(f"\nFinalizing... Saving {len(all_data)} points to {filename}.")
        print("DO NOT CLOSE. This will take a moment.")
        
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Z", "R", "G", "B"])
            writer.writerows(all_data)
        print(f"\nSUCCESS. File saved.")
    else:
        print("\nNo data was captured.")

    cam0.close()
    cam1.close()
    vis.destroy_window()

if __name__ == "__main__":
    main()