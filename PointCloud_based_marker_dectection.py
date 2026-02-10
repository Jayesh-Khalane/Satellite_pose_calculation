import pyzed.sl as sl
import numpy as np
import sys

def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

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

    # Clear screen initially
    print("\033[2J", end="") 

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                full_data = point_cloud.get_data()
                
                # Unpack RGB and find bright pixels
                rgba_bits = full_data[:, :, 3].view(np.uint32)
                r = (rgba_bits >> 0) & 0xFF
                g = (rgba_bits >> 8) & 0xFF
                b = (rgba_bits >> 16) & 0xFF
                rgb_sum = r + g + b
                
                # Filter valid points
                mask = (rgb_sum > 750) & (~np.isnan(full_data[:, :, 0]))
                
                # If we don't have enough points for 3 LEDs, show status and loop
                if np.count_nonzero(mask) < 10: # Small threshold to avoid noise
                    sys.stdout.write("\033[H")
                    sys.stdout.write("=========================================\n")
                    sys.stdout.write("       SATELLITE MARKER TRACKER          \n")
                    sys.stdout.write("=========================================\n")
                    sys.stdout.write(" STATUS: SCANNING (LEDs NOT FOUND)       \n")
                    sys.stdout.write("                                         \n"*10) # Clear previous data
                    sys.stdout.flush()
                    continue

                xyz_all = full_data[mask][:, :3]
                sums_all = rgb_sum[mask]
                
                found_leds = []
                temp_xyz = xyz_all.copy()
                temp_sums = sums_all.copy()

                # Find up to 3 clusters
                for _ in range(3):
                    if len(temp_xyz) == 0: break
                    seed_idx = np.argmax(temp_sums)
                    seed_pt = temp_xyz[seed_idx]
                    
                    # Group points within 10cm (as per your update cluster_mask = dists <= 0.1)
                    dists = np.linalg.norm(temp_xyz - seed_pt, axis=1)
                    cluster_mask = dists <= 0.1
                    
                    found_leds.append(np.mean(temp_xyz[cluster_mask], axis=0))
                    temp_xyz = temp_xyz[~cluster_mask]
                    temp_sums = temp_sums[~cluster_mask]

                if len(found_leds) == 3:
                    A, B, C = found_leds
                    
                    # Sort pairs by distance
                    dAB, dBC, dCA = get_dist(A, B), get_dist(B, C), get_dist(C, A)
                    dist_map = [(dAB, (A, B)), (dBC, (B, C)), (dCA, (C, A))]
                    dist_map.sort(key=lambda x: x[0], reverse=True)
                    
                    pair1, pair2 = dist_map[0][1], dist_map[1][1]
                    
                    # Identify P1 (Common vertex)
                    p1 = None
                    for pt1 in pair1:
                        for pt2 in pair2:
                            if np.array_equal(pt1, pt2):
                                p1 = pt1
                                break
                    
                    if p1 is not None:
                        p3 = pair1[1] if np.array_equal(pair1[0], p1) else pair1[0]
                        p2 = pair2[1] if np.array_equal(pair2[0], p1) else pair2[0]

                        # --- REFRESH DASHBOARD ---
                        sys.stdout.write("\033[H") # Move cursor to top-left
                        sys.stdout.write("=========================================\n")
                        sys.stdout.write("       SATELLITE MARKER TRACKER          \n")
                        sys.stdout.write("=========================================\n")
                        sys.stdout.write(f" STATUS: TRACKING (3/3 LEDS FOUND)      \n\n")
                        
                        sys.stdout.write(f" P1 (CORNER): X:{p1[0]:>7.3f} Y:{p1[1]:>7.3f} Z:{p1[2]:>7.3f}\n")
                        sys.stdout.write(f" P2 (SHORT):  X:{p2[0]:>7.3f} Y:{p2[1]:>7.3f} Z:{p2[2]:>7.3f}\n")
                        sys.stdout.write(f" P3 (LONG):   X:{p3[0]:>7.3f} Y:{p3[1]:>7.3f} Z:{p3[2]:>7.3f}\n\n")
                        
                        sys.stdout.write(" DISTANCES (METERS):\n")
                        sys.stdout.write(f" P1-P2: {get_dist(p1, p2):.4f}\n")
                        sys.stdout.write(f" P2-P3: {get_dist(p2, p3):.4f}\n")
                        sys.stdout.write(f" P3-P1: {get_dist(p1, p3):.4f}\n")
                        sys.stdout.write("=========================================\n")
                        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping...")
    zed.close()

if __name__ == "__main__":
    main()