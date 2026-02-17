import pyzed.sl as sl
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def print_dashboard(status, fps, missed, total, points_count=0, centroid=None, euler=None, quat=None, p_pts=None, dists=None):
    """ Helper to render the dashboard consistently """
    # \033[H moves cursor to top left, \033[J clears from cursor down
    sys.stdout.write("\033[H") 
    sys.stdout.write("========================================================================\n")
    sys.stdout.write("                LED MARKER POSE TRACKING                                \n")
    sys.stdout.write(f" FPS: {fps:<4}         Frame missed: {missed:<5}      Total Frames: {total:<5} \n")
    sys.stdout.write(f" STATUS: {status:<20}        POINTS IN CLUSTER: {points_count}\n")
    sys.stdout.write("========================================================================\n\n")

    if centroid is not None:
        sys.stdout.write(f" CENTROID    X:{centroid[0]:>7.3f}cm   Y:{centroid[1]:>7.3f}cm   Z:{centroid[2]:>7.3f}cm\n")
        sys.stdout.write(f" ORIENTATION Rx:{euler[0]:>7.3f}°   Ry:{euler[1]:>7.3f}°   Rz:{euler[2]:>7.3f}°\n")
        sys.stdout.write(f" QUATERNION  X:{quat[0]:>6.3f}   Y:{quat[1]:>6.3f}   Z:{quat[2]:>6.3f}   W:{quat[3]:>6.3f}\n\n")
        
        for i, p in enumerate(p_pts):
            sys.stdout.write(f" [ P{i+1}]   X:{p[0]:>7.3f}cm   Y:{p[1]:>7.3f}cm   Z:{p[2]:>7.3f}cm \n")
        
        sys.stdout.write(f"\n [DISTANCES] P1-P2: {dists[0]:.2f}cm  P2-P3: {dists[1]:.2f}cm  P1-P3: {dists[2]:.2f}cm\n")
    else:
        # Clear the rest of the dashboard area if no data
        sys.stdout.write("\n\n\n\n\n\n\n\n")
        
    sys.stdout.write("========================================================================\n")
    sys.stdout.flush()

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    frame_drop_count = 0
    total_tracking_frames = 0
    tracking_locked = False 

    print("\033[2J", end="") # Clear screen once at start

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                camera_fps = int(zed.get_current_fps())
                if tracking_locked: total_tracking_frames += 1

                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                full_data = point_cloud.get_data()
                
                # Extract RGB and Create Mask
                rgba_bits = full_data[:, :, 3].view(np.uint32)
                r = (rgba_bits >> 0) & 0xFF
                g = (rgba_bits >> 8) & 0xFF
                b = (rgba_bits >> 16) & 0xFF
                rgb_sum = r + g + b
                
                mask = (rgb_sum > 750) & (~np.isnan(full_data[:, :, 0]))
                num_points = np.count_nonzero(mask)

                if num_points < 10:
                    if tracking_locked: frame_drop_count += 1
                    print_dashboard("[ SEARCHING... ]", camera_fps, frame_drop_count, total_tracking_frames, num_points)
                    continue

                # Clustering
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

                if len(found_leds) != 3:
                    if tracking_locked: frame_drop_count += 1
                    print_dashboard("[ INCOMPLETE CLUSTER ]", camera_fps, frame_drop_count, total_tracking_frames, num_points)
                    continue

                # Lock tracking on first successful 3-point detection
                if not tracking_locked:
                    tracking_locked = True
                    total_tracking_frames = 1

                # Sort Points to maintain consistent orientation
                A, B, C = found_leds
                d_list = [(get_dist(A, B), (A, B)), (get_dist(B, C), (B, C)), (get_dist(C, A), (C, A))]
                d_list.sort(key=lambda x: x[0], reverse=True)
                
                pair1, pair2 = d_list[0][1], d_list[1][1]
                p1 = next(pt1 for pt1 in pair1 if any(np.array_equal(pt1, pt2) for pt2 in pair2))
                p3 = pair1[1] if np.array_equal(pair1[0], p1) else pair1[0]
                p2 = pair2[1] if np.array_equal(pair2[0], p1) else pair2[0]

                # Math for Pose
                centroid = (p1 + p2 + p3) / 3.0
                v_x = (p2 - p1) / np.linalg.norm(p2 - p1)
                v_short = (p3 - p1) / np.linalg.norm(p3 - p1)
                v_z = np.cross(v_x, v_short)
                v_z /= np.linalg.norm(v_z)
                v_y = np.cross(v_z, v_x)
                
                rot_matrix = np.stack([v_x, v_y, v_z], axis=1)
                r_obj = R.from_matrix(rot_matrix)
                
                # Output
                print_dashboard(
                    "[ TRACKING ACTIVE ]", 
                    camera_fps, frame_drop_count, total_tracking_frames, 
                    num_points, centroid, 
                    r_obj.as_euler('xyz', degrees=True), 
                    r_obj.as_quat(),
                    [p1, p2, p3],
                    [get_dist(p1,p2), get_dist(p2,p3), get_dist(p1,p3)]
                )

    except KeyboardInterrupt:
        print("\nTracking Stopped.")
    finally:
        zed.close()

if __name__ == "__main__":
    main()