import cv2 as cv
import numpy as np
import pyzed.sl as sl
import itertools
from scipy.spatial.transform import Rotation as R_tool

# -----------------------------
# Logic Functions
# -----------------------------

def get_3d_position_at_pixel(pc, px, py):
    if 0 <= px < pc.shape[1] and 0 <= py < pc.shape[0]:
        point = pc[py, px, :3]
        if np.isfinite(point).all():
            return np.array(point)
    return None

def find_blob_centroids(binary_mask):
    # Filter blobs: MIN_BLOB_SIZE = 20\



    
    _, _, stats, centroids = cv.connectedComponentsWithStats(binary_mask)
    return [(int(c[0]), int(c[1])) for i, c in enumerate(centroids[1:]) 
            if stats[i+1, cv.CC_STAT_AREA] >= 20]

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
        
    p1_idx = list(common)[0]
    p3_idx = L1_pair[0] if L1_pair[0] != p1_idx else L1_pair[1]
    p2_idx = L2_pair[0] if L2_pair[0] != p1_idx else L2_pair[1]
    
    labels = [""] * 3
    labels[p1_idx], labels[p2_idx], labels[p3_idx] = "p1", "p2", "p3"
    return labels

def compute_pose_matrix(world_coords, labels):
    lookup = {label: coord for coord, label in zip(world_coords, labels)}
    p1, p2, p3 = lookup["p1"], lookup["p2"], lookup["p3"]
    centroid = np.mean(world_coords, axis=0)

    x_axis = (p2 - p1) / np.linalg.norm(p2 - p1)
    z_axis = np.cross(p2 - p1, p3 - p1)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    return centroid, np.column_stack((x_axis, y_axis, z_axis))

def project_point(point_3d, camera_params):
    fx, fy, cx, cy = camera_params.fx, camera_params.fy, camera_params.cx, camera_params.cy
    x, y, z = point_3d
    if z <= 0: return None 
    return (int((x * fx / z) + cx), int(cy - (y * fy / z)))

def draw_orientation_axes(image, centroid, R, cam_params):
    origin_2d = project_point(centroid, cam_params)
    if not origin_2d: return
    # Colors: BGR (Red for X, Green for Y, Blue for Z)
    colors, lbls = [(0, 0, 255), (0, 255, 0), (255, 0, 0)], ["X", "Y", "Z"]
    for i in range(3):
        # Axis length = 10.0 units
        p2d = project_point(centroid + R[:, i] * 10.0, cam_params)
        if p2d:
            cv.line(image, origin_2d, p2d, colors[i], 2, cv.LINE_AA)
            cv.putText(image, lbls[i], p2d, cv.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1, cv.LINE_AA)

# -----------------------------
# Main
# -----------------------------
def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: return

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    left, point_cloud = sl.Mat(), sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    cv.namedWindow("LED Detection", cv.WINDOW_NORMAL)
    frame_count = 0

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS: continue

        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        
        frame = left.get_data()
        pc_data = point_cloud.get_data()
        
        # Threshold: RGB Sum > 750
        binary_mask = (np.sum(frame[:, :, :3], axis=2) > 755).astype(np.uint8) * 255
        display = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)

        centroids = find_blob_centroids(binary_mask)
        pixel_coords, world_coords = [], []
        
        for cx, cy in centroids:
            w_point = get_3d_position_at_pixel(pc_data, cx, cy)
            if w_point is not None:
                pixel_coords.append((cx, cy))
                world_coords.append(w_point)

        labels = assign_labels(world_coords)

        for i, ((px, py), (x, y, z)) in enumerate(zip(pixel_coords, world_coords)):
            # Draw LED dot (radius 3)
            cv.circle(display, (px, py), 2, (0, 0, 255), -1, cv.LINE_AA)
            
            prefix = f"{labels[i]} " if labels else ""
            txt = f"{prefix}({x:.2f},{y:.2f},{z:.2f})"
            # Text Scale: 0.4, Thickness: 1, Color: Cyan (0, 255, 255)
            cv.putText(display, txt, (px + 5, py - 5), cv.FONT_HERSHEY_SIMPLEX, 
                        0.4, (0, 255, 255), 1, cv.LINE_AA)

            # Terminal Log: Every 24 frames
            if labels and frame_count % 24 == 0:
                print(f"{labels[i].upper()}: x={x:>7.2f}, y={y:>7.2f}, z={z:>7.2f}")
        
        if labels and frame_count % 24 == 0:
            print("-" * 30)

        if len(world_coords) == 3 and labels:
            centroid, R_mat = compute_pose_matrix(world_coords, labels)
            if centroid is not None:
                rot_obj = R_tool.from_matrix(R_mat)
                quat = rot_obj.as_quat()
                euler = rot_obj.as_euler('xyz', degrees=True)

                # Connect points with Cyan line (thickness 1)
                cv.polylines(display, [np.array(pixel_coords)], True, (0, 255, 255), 1, cv.LINE_AA)
                draw_orientation_axes(display, centroid, R_mat, calib)
                
                # UI Overlays
                cv.putText(display, f"POSE: {centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}", 
                           (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(display, f"ROT: X:{euler[0]:.1f} Y:{euler[1]:.1f} Z:{euler[2]:.1f}", 
                           (15, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(display, f"QUAT: {np.array2string(quat, precision=2, separator=',')}", 
                           (15, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)

        frame_count += 1
        cv.imshow("LED Detection", display)
        if cv.waitKey(1) & 0xFF == 27: break # ESC to quit

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()