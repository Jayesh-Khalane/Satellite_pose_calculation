import pyzed.sl as sl
import open3d as o3d
import numpy as np
import time
import csv
from scipy.spatial.transform import Rotation as R_tool
import sys
import os

# --- CONFIGURATION ---
CAPTURE_COUNT = 3
STAGE1_SEC = 15.0      # Time to move camera into position
STAGE2_SEC = 10.0      # Time to average the LED tracking position
MAX_DIST_CM = 100.0

# --- CAMERA SERIAL NUMBERS ---
ZED0_SN = 33140394       # Local camera (Cam2), looking at satellite & IMU source
ZED1_SN = 36763817       # Global camera (Cam1), looking at LEDs & Position source

def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def track_leds(full_data):
    """Tracks LEDs to get global position."""
    rgba_bits = full_data[:, :, 3].view(np.uint32)
    r = (rgba_bits >> 0) & 0xFF
    g = (rgba_bits >> 8) & 0xFF
    b = (rgba_bits >> 16) & 0xFF
    rgb_sum = r + g + b
    
    mask = (rgb_sum > 750) & (~np.isnan(full_data[:, :, 0]))
    num_points = np.count_nonzero(mask)

    if num_points < 10: return None, None, num_points, None

    xyz_all = full_data[mask][:, :3] * 100.0
    found_leds = []
    temp_xyz, temp_sums = xyz_all.copy(), rgb_sum[mask].copy()

    for _ in range(3):
        if len(temp_xyz) == 0: break
        seed_idx = np.argmax(temp_sums)
        seed_pt = temp_xyz[seed_idx]
        dists = np.linalg.norm(temp_xyz - seed_pt, axis=1)
        cluster_mask = dists <= 10.0 
        found_leds.append(np.mean(temp_xyz[cluster_mask], axis=0))
        temp_xyz, temp_sums = temp_xyz[~cluster_mask], temp_sums[~cluster_mask]

    if len(found_leds) != 3: return None, None, num_points, None

    A, B, C = found_leds
    d_list = [(get_dist(A, B), (A, B)), (get_dist(B, C), (B, C)), (get_dist(C, A), (C, A))]
    d_list.sort(key=lambda x: x[0], reverse=True)
    
    pair1, pair2 = d_list[0][1], d_list[1][1]
    p1 = next(pt1 for pt1 in pair1 if any(np.array_equal(pt1, pt2) for pt2 in pair2))
    p3 = pair1[1] if np.array_equal(pair1[0], p1) else pair1[0]
    p2 = pair2[1] if np.array_equal(pair2[0], p1) else pair2[0]

    centroid = (p1 + p2 + p3) / 3.0
    v_x = (p2 - p1) / np.linalg.norm(p2 - p1)
    v_short = (p3 - p1) / np.linalg.norm(p3 - p1)
    v_z = np.cross(v_x, v_short)
    v_z /= np.linalg.norm(v_z)
    v_y = np.cross(v_z, v_x)
    
    rot_matrix = np.stack([v_x, v_y, v_z], axis=1)
    eulers = R_tool.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
    stats = {'p1': p1, 'p2': p2, 'p3': p3, 'd12': get_dist(p1, p2), 'd23': get_dist(p2, p3), 'd13': get_dist(p1, p3)}

    return rot_matrix, centroid, num_points, (eulers, stats)

def render_dashboard(c1_fps, c1_miss, c1_tot, c1_status, pts_count, pose_ui, 
                     c0_fps, c0_miss, c0_tot, c0_status, live_imu,
                     phase, timer, history, total_pts):
    """Dual-Camera Static Dashboard with LIVE IMU"""
    sys.stdout.write('\033[H') 
    
    # --- CAM 1 (GLOBAL POSITION) ---
    out = "========================================================================\n"
    out += "                 [ CAM1 : GLOBAL SENSOR / LED TRACKING ]\n"
    out += f" FPS: {c1_fps:<10}  Frame missed: {c1_miss:<10}  Total Frames: {c1_tot}\n"
    out += f" STATUS: [ {c1_status:^15} ]        POINTS IN CLUSTER: {pts_count}\n"
    out += "------------------------------------------------------------------------\n"
    
    if pose_ui:
        eulers, stats, c = pose_ui[0], pose_ui[1], pose_ui[2]
        out += f" CENTROID    X:{c[0]:>8.3f}cm   Y:{c[1]:>8.3f}cm   Z:{c[2]:>8.3f}cm\n"
    else:
        out += " [!] WAITING FOR VALID LED CLUSTERS IN CAM1 FIELD OF VIEW...\n"
        
    # --- CAM 2 (LOCAL IMU & PCD) ---
    out += "========================================================================\n"
    out += "                 [ CAM2 : LOCAL SENSOR / IMU & POINT CLOUD ]\n"
    out += f" FPS: {c0_fps:<10}  Frame missed: {c0_miss:<10}  Total Frames: {c0_tot}\n"
    out += f" STATUS: [ {c0_status:^15} ]\n"
    if live_imu:
        out += f" LIVE IMU -> Roll: {live_imu[0]:>7.2f}° | Pitch: {live_imu[1]:>7.2f}° | Yaw: {live_imu[2]:>7.2f}°\n"
    else:
        out += " LIVE IMU -> WAITING FOR DATA...\n"
    
    # --- PIPELINE ---
    out += "========================================================================\n"
    out += "                [ HYBRID PIPELINE : DATA CAPTURE ]\n"
    out += f" CURRENT PHASE : [ {phase} ]   TIMER: {timer}\n"
    out += "------------------------------------------------------------------------\n"
    
    for i in range(CAPTURE_COUNT):
        if history[i]['pts'] > 0:
            p, e = history[i]['pos'], history[i]['euler']
            out += f" [ CAPTURE {i+1} ] Pts: {history[i]['pts']:<7} | Pos: [{p[0]:>6.1f}, {p[1]:>6.1f}, {p[2]:>6.1f}] | IMU: [{e[0]:>6.1f}°, {e[1]:>6.1f}°, {e[2]:>6.1f}°]\n"
        else:
            out += f" [ CAPTURE {i+1} ] Pts: ------- | Pos: [------, ------, ------] | IMU: [------°, ------°, ------°]\n"
            
    out += "------------------------------------------------------------------------\n"
    out += f" TOTAL POINTS SAVED: {total_pts}\n"
    out += "========================================================================\n"
    out += "\033[J" # Clear trailing artifacts
    sys.stdout.write(out)
    sys.stdout.flush()

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    sys.stdout.write('\033[2J') 

    # --- INITIALIZE CAMERAS ---
    cam0 = sl.Camera()
    cam1 = sl.Camera()

    init0 = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.NEURAL_PLUS, 
                              coordinate_units=sl.UNIT.METER, coordinate_system=sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP)
    init0.set_from_serial_number(ZED0_SN)

    init1 = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.NEURAL_PLUS, 
                              coordinate_units=sl.UNIT.METER, coordinate_system=sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP)
    init1.set_from_serial_number(ZED1_SN)

    if cam0.open(init0) != sl.ERROR_CODE.SUCCESS or cam1.open(init1) != sl.ERROR_CODE.SUCCESS:
        print("Camera Initialization failed.")
        return

    # CRITICAL FIX: Enable Positional Tracking on Cam0 so IMU remembers state
    tracking_params = sl.PositionalTrackingParameters()
    cam0.enable_positional_tracking(tracking_params)

    # --- OPEN3D SETUP ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Cam2 Live Preview (<1m) + Live IMU", width=1280, height=720)
    vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
    
    pcd = o3d.geometry.PointCloud()
    geom_added = False
    
    # Add a live 3D Marker to represent IMU orientation
    imu_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    vis.add_geometry(imu_marker)
    prev_rot_matrix = np.eye(3) # Keep track to undo relative rotation

    pc0, pc1 = sl.Mat(), sl.Mat()
    runtime = sl.RuntimeParameters()
    sensors_data = sl.SensorsData()

    all_local_list = [] 
    snapshot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    captures_done, total_pts_saved = 0, 0
    c0_tot, c0_miss, c1_tot, c1_miss = 0, 0, 0, 0
    history = [{'pts': 0, 'pos': None, 'euler': None} for _ in range(CAPTURE_COUNT)]

    # Helper function to process IMU visually
    def update_live_imu():
        nonlocal prev_rot_matrix
        live_euler = None
        if cam0.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
            current_rot = sensors_data.get_imu_data().get_pose().get_rotation_matrix().r
            live_euler = R_tool.from_matrix(current_rot).as_euler('xyz', degrees=True).tolist()
            
            # Spin the 3D marker in Open3D
            imu_marker.rotate(prev_rot_matrix.T, center=[0, 0, 0]) # Undo last
            imu_marker.rotate(current_rot, center=[0, 0, 0])       # Apply new
            prev_rot_matrix = current_rot
            vis.update_geometry(imu_marker)
        return live_euler

    try:
        while captures_done < CAPTURE_COUNT:
            # ================= STAGE 1: POSITIONING =================
            stage1_start = time.time()
            while time.time() - stage1_start < STAGE1_SEC:
                err0, err1 = cam0.grab(runtime), cam1.grab(runtime)
                if err0 == sl.ERROR_CODE.SUCCESS: c0_tot += 1
                else: c0_miss += 1
                if err1 == sl.ERROR_CODE.SUCCESS: c1_tot += 1
                else: c1_miss += 1
                
                if err0 == sl.ERROR_CODE.SUCCESS and err1 == sl.ERROR_CODE.SUCCESS:
                    cam1.retrieve_measure(pc1, sl.MEASURE.XYZRGBA)
                    R_cam, centroid, pts_count, ui_stats = track_leds(pc1.get_data())
                    
                    # Update Live IMU & Visual Marker
                    live_imu_angles = update_live_imu()
                    
                    pose_ui = (ui_stats[0], ui_stats[1], centroid) if ui_stats else None
                    render_dashboard(int(cam1.get_current_fps()), c1_miss, c1_tot, "TRACKING" if R_cam is not None else "SEARCHING", pts_count, pose_ui,
                                     int(cam0.get_current_fps()), c0_miss, c0_tot, "LIVE STREAMING", live_imu_angles,
                                     "STAGE 1: POSITIONING", f"{int(STAGE1_SEC - (time.time() - stage1_start))}s", history, total_pts_saved)

                    # Update Lidar Point Cloud
                    cam0.retrieve_measure(pc0, sl.MEASURE.XYZRGBA)
                    pc0_np = pc0.get_data()
                    xyz = pc0_np[:, :, :3].reshape(-1, 3) * 100.0
                    valid_mask = np.isfinite(xyz).all(axis=1)
                    xyz_valid = xyz[valid_mask]
                    xyz_filtered = xyz_valid[np.linalg.norm(xyz_valid, axis=1) < MAX_DIST_CM]

                    if len(xyz_filtered) > 0:
                        rgba_float = pc0_np[:, :, 3].reshape(-1)[valid_mask][np.linalg.norm(xyz_valid, axis=1) < MAX_DIST_CM]
                        rgba_int = rgba_float.view(np.uint32)
                        colors = np.stack([(rgba_int & 0xFF).astype(np.uint8), ((rgba_int >> 8) & 0xFF).astype(np.uint8), ((rgba_int >> 16) & 0xFF).astype(np.uint8)], axis=1) / 255.0

                        pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        if not geom_added: vis.add_geometry(pcd); geom_added = True
                        vis.update_geometry(pcd)
                    
                    vis.poll_events()
                    vis.update_renderer()

            # ================= STAGE 2: AVERAGING POSITION =================
            translations = []
            stage2_start = time.time()
            
            while time.time() - stage2_start < STAGE2_SEC:
                err0, err1 = cam0.grab(runtime), cam1.grab(runtime)
                if err0 == sl.ERROR_CODE.SUCCESS: c0_tot += 1
                else: c0_miss += 1
                if err1 == sl.ERROR_CODE.SUCCESS: c1_tot += 1
                else: c1_miss += 1
                
                if err0 == sl.ERROR_CODE.SUCCESS and err1 == sl.ERROR_CODE.SUCCESS:
                    cam1.retrieve_measure(pc1, sl.MEASURE.XYZRGBA)
                    R_cam, centroid, pts_count, ui_stats = track_leds(pc1.get_data())

                    if R_cam is not None: translations.append(centroid)
                        
                    live_imu_angles = update_live_imu()
                    pose_ui = (ui_stats[0], ui_stats[1], centroid) if ui_stats else None
                    
                    render_dashboard(int(cam1.get_current_fps()), c1_miss, c1_tot, "CALCULATING AVG" if R_cam is not None else "SEARCHING", pts_count, pose_ui,
                                     int(cam0.get_current_fps()), c0_miss, c0_tot, "HOLD STILL", live_imu_angles,
                                     "STAGE 2: AVERAGING POSE", f"{int(STAGE2_SEC - (time.time() - stage2_start))}s", history, total_pts_saved)
                    
                    vis.poll_events()
                    vis.update_renderer()

            if len(translations) == 0: continue 

            P_avg = np.mean(translations, axis=0)
            
            # Lock in the exact IMU angle at the moment capture ends
            final_euler = [0.0, 0.0, 0.0]
            if cam0.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                final_rot = sensors_data.get_imu_data().get_pose().get_rotation_matrix().r
                final_euler = R_tool.from_matrix(final_rot).as_euler('xyz', degrees=True).tolist()

            # ================= STAGE 3: RAW CAPTURE & SAVING =================
            cam0.retrieve_measure(pc0, sl.MEASURE.XYZRGBA)
            pc0_np = pc0.get_data()
            xyz = pc0_np[:, :, :3].reshape(-1, 3) * 100.0
            valid_mask = np.isfinite(xyz).all(axis=1)
            xyz_valid = xyz[valid_mask]
            xyz_filtered = xyz_valid[np.linalg.norm(xyz_valid, axis=1) < MAX_DIST_CM]

            r_col, g_col, b_col = snapshot_colors[captures_done]
            color_array = np.full((xyz_filtered.shape[0], 3), [r_col, g_col, b_col])
            all_local_list.append(np.hstack((xyz_filtered, color_array)))

            history[captures_done]['pts'] = len(xyz_filtered)
            history[captures_done]['pos'] = P_avg
            history[captures_done]['euler'] = final_euler
            total_pts_saved += len(xyz_filtered)
            captures_done += 1
            
            time.sleep(1.5)

    except KeyboardInterrupt:
        pass

    # --- FINAL EXPORT ---
    if all_local_list:
        np.savetxt("log/sat_point_cloud_capture.csv", np.vstack(all_local_list), delimiter=",", header="X,Y,Z,R,G,B", comments="", fmt="%.4f")
        with open("log/capture_poses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Capture_ID", "X_cm", "Y_cm", "Z_cm", "Roll", "Pitch", "Yaw"])
            for i in range(CAPTURE_COUNT):
                pos, rot = history[i]['pos'], history[i]['euler']
                if pos is not None:
                    writer.writerow([i+1, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]])
    
    cam0.close()
    cam1.close()
    vis.destroy_window()

if __name__ == "__main__":
    main()