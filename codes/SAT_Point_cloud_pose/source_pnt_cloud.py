import pyzed.sl as sl
import numpy as np
import sys
import itertools
from scipy.spatial.transform import Rotation as R_tool

# -----------------------------
# Logic Functions
# -----------------------------

def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def assign_labels(world_coords):
    if len(world_coords) != 3:
        return None
    
    idx_pairs = list(itertools.combinations(range(3), 2))
    dists = []
    for i, j in idx_pairs:
        d = np.linalg.norm(world_coords[i] - world_coords[j])
        dists.append({'pair': (i, j), 'dist': d})
    
    # Sort to find the longest and second longest segments
    dists.sort(key=lambda x: x['dist'], reverse=True)
    L1_pair, L2_pair = dists[0]['pair'], dists[1]['pair']
    
    common = set(L1_pair).intersection(set(L2_pair))
    if len(common) != 1: return None
        
    p1_idx = list(common)[0] # The Joint
    p2_idx = L2_pair[0] if L2_pair[0] != p1_idx else L2_pair[1] # X-Arm
    p3_idx = L1_pair[0] if L1_pair[0] != p1_idx else L1_pair[1] # Y-Arm (Longer)
    
    labels = [""] * 3
    labels[p1_idx], labels[p2_idx], labels[p3_idx] = "p1", "p2", "p3"
    return labels

def compute_pose_matrix(world_coords, labels):
    lookup = {label: coord for coord, label in zip(world_coords, labels)}
    p1, p2, p3 = lookup["p1"], lookup["p2"], lookup["p3"]
    centroid = np.mean(world_coords, axis=0)

    # --- YOUR SPECIFIED REFERENCE MATH ---
    x_axis = (p2 - p1) / np.linalg.norm(p2 - p1)
    z_axis = np.cross(p2 - p1, p3 - p1)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    return centroid, np.column_stack((x_axis, y_axis, z_axis)), [p1, p2, p3]

def print_dashboard(status, fps, missed, total, points_count=0, centroid=None, euler=None, quat=None, p_pts=None, sn="Unknown"):
    """ Restored original Terminal Dashboard """
    sys.stdout.write("\033[H") 
    sys.stdout.write("========================================================================\n")
    sys.stdout.write(f"           LED MARKER POSE TRACKING - CAMERA [1] (SN: {sn})           \n")
    sys.stdout.write(f" FPS: {fps:<4}         Frame missed: {missed:<5}       Total Frames: {total:<5} \n")
    sys.stdout.write(f" STATUS: {status:<20}         POINTS IN CLUSTER: {points_count}\n")
    sys.stdout.write("========================================================================\n\n")

    if centroid is not None:
        sys.stdout.write(f" CENTROID    X:{centroid[0]:>7.3f}cm   Y:{centroid[1]:>7.3f}cm   Z:{centroid[2]:>7.3f}cm\n")
        sys.stdout.write(f" ORIENTATION Rx:{euler[0]:>7.3f}°   Ry:{euler[1]:>7.3f}°   Rz:{euler[2]:>7.3f}°\n")
        sys.stdout.write(f" QUATERNION  X:{quat[0]:>6.3f}   Y:{quat[1]:>6.3f}   Z:{quat[2]:>6.3f}   W:{quat[3]:>6.3f}\n\n")
        
        labels = ["JOINT (P1)", "X-ARM (P2)", "Y-ARM (P3)"]
        for i, p in enumerate(p_pts):
            sys.stdout.write(f" [{labels[i]}]   X:{p[0]:>7.3f}cm   Y:{p[1]:>7.3f}cm   Z:{p[2]:>7.3f}cm \n")
    else:
        sys.stdout.write("\n\n\n\n\n\n\n\n")
        
    sys.stdout.write("========================================================================\n")
    sys.stdout.flush()

# -----------------------------
# Main Loop
# -----------------------------
def main():
    # Detect cameras and target Index [1]
    devices = sl.Camera.get_device_list()
    if len(devices) < 2:
        print(f"Error: Need at least 2 cameras for Index [1]. Found {len(devices)}.")
        return
    target_sn = devices[1].serial_number

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(target_sn)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.CENTIMETER # Matches your reference code
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    frame_drop_count = 0
    total_tracking_frames = 0
    tracking_locked = False 

    print("\033[2J", end="") # Clear screen once

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                camera_fps = int(zed.get_current_fps())
                if tracking_locked: total_tracking_frames += 1

                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                full_data = point_cloud.get_data()
                
                # Thresholding logic from original code
                rgba_bits = full_data[:, :, 3].view(np.uint32)
                r, g, b = (rgba_bits >> 0) & 0xFF, (rgba_bits >> 8) & 0xFF, (rgba_bits >> 16) & 0xFF
                rgb_sum = r + g + b
                
                mask = (rgb_sum > 750) & (~np.isnan(full_data[:, :, 0]))
                num_points = np.count_nonzero(mask)

                if num_points < 10:
                    if tracking_locked: frame_drop_count += 1
                    print_dashboard("[ SEARCHING... ]", camera_fps, frame_drop_count, total_tracking_frames, num_points, sn=target_sn)
                    continue

                # Clustering
                xyz_all = full_data[mask][:, :3] 
                found_leds = []
                temp_xyz, temp_sums = xyz_all.copy(), rgb_sum[mask].copy()

                for _ in range(3):
                    if len(temp_xyz) == 0: break
                    seed_idx = np.argmax(temp_sums)
                    seed_pt = temp_xyz[seed_idx]
                    dist_array = np.linalg.norm(temp_xyz - seed_pt, axis=1)
                    cluster_mask = dist_array <= 10.0 
                    found_leds.append(np.mean(temp_xyz[cluster_mask], axis=0))
                    temp_xyz, temp_sums = temp_xyz[~cluster_mask], temp_sums[~cluster_mask]

                if len(found_leds) != 3:
                    if tracking_locked: frame_drop_count += 1
                    print_dashboard("[ INCOMPLETE CLUSTER ]", camera_fps, frame_drop_count, total_tracking_frames, num_points, sn=target_sn)
                    continue

                # Compute Pose with your logic
                labels = assign_labels(found_leds)
                if labels:
                    tracking_locked = True
                    centroid, R_mat, sorted_pts = compute_pose_matrix(found_leds, labels)
                    
                    rot_obj = R_tool.from_matrix(R_mat)
                    quat = rot_obj.as_quat() # [x, y, z, w]
                    euler = rot_obj.as_euler('xyz', degrees=True)

                    print_dashboard(
                        "[ TRACKING ACTIVE ]", 
                        camera_fps, frame_drop_count, total_tracking_frames, 
                        num_points, centroid, euler, quat, sorted_pts, sn=target_sn
                    )

    except KeyboardInterrupt:
        print("\nTracking Stopped.")
    finally:
        zed.close()

if __name__ == "__main__":
    main()