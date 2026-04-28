import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_cloud(inner_r, outer_r, num_points, noise, env_points):
    """Generates the noisy annulus and environment clutter."""
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(np.random.uniform(inner_r**2, outer_r**2, num_points))
    
    x = r * np.cos(theta) + np.random.normal(0, noise, num_points)
    y = r * np.sin(theta) + np.random.normal(0, noise, num_points)
    z = np.zeros(num_points) + np.random.normal(0, noise/2, num_points)
    
    annulus_pts = np.vstack((x, y, z)).T
    
    spread = outer_r * 3.0
    if env_points > 0:
        env_x = np.random.uniform(-spread, spread, env_points)
        env_y = np.random.uniform(-spread, spread, env_points)
        env_z = np.random.uniform(-spread/3, spread/3, env_points)
        env_pts = np.vstack((env_x, env_y, env_z)).T
        final_pts = np.vstack((annulus_pts, env_pts))
    else:
        final_pts = annulus_pts
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.paint_uniform_color([1.0, 1.0, 1.0]) # Pure white
    return pcd

def fit_circle_least_squares(xy_points):
    x = xy_points[:, 0]
    y = xy_points[:, 1]
    
    A = np.c_[x, y, np.ones(len(x))]
    B = x**2 + y**2
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    xc = C[0] / 2
    yc = C[1] / 2
    radius = np.sqrt(C[2] + xc**2 + yc**2)
    return xc, yc, radius

def create_drawn_circle(xc, yc, zc, radius, color=[0.0, 0.8, 1.0], num_edges=100):
    theta = np.linspace(0, 2 * np.pi, num_edges)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    z = np.full_like(x, zc)
    
    points = np.c_[x, y, z]
    lines = [[i, (i + 1) % num_edges] for i in range(num_edges)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def attempt_fit(pcd, eps, min_points):
    """Attempts to isolate the shape and run Least Squares. Returns None on failure."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if len(labels) == 0 or labels.max() == -1:
        return None, None, None, None, None, None
        
    largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
    object_indices = np.where(labels == largest_cluster_idx)[0]
    obj_points = np.asarray(pcd.points)[object_indices]
    
    xy_points = obj_points[:, :2]
    center_guess = np.mean(xy_points, axis=0)
    shifted = xy_points - center_guess
    r = np.linalg.norm(shifted, axis=1)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    
    num_bins = 72
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    indices = np.digitize(theta, bins)
    
    inner_boundary = []
    outer_boundary = []
    
    for i in range(1, num_bins + 1):
        bin_mask = (indices == i)
        if not np.any(bin_mask): continue
        
        relative_indices = np.where(bin_mask)[0]
        bin_r = r[bin_mask]
        
        inner_boundary.append(xy_points[relative_indices[np.argmin(bin_r)]])
        outer_boundary.append(xy_points[relative_indices[np.argmax(bin_r)]])
        
    if len(inner_boundary) < 3 or len(outer_boundary) < 3:
        return None, None, None, None, None, None

    in_xc, in_yc, in_r = fit_circle_least_squares(np.array(inner_boundary))
    out_xc, out_yc, out_r = fit_circle_least_squares(np.array(outer_boundary))
    
    return in_r, out_r, in_xc, in_yc, out_xc, out_yc

def run_experiment_and_visualize():
    INNER_R = 3.0
    OUTER_R = 6.0
    NUM_ANNULUS_POINTS = 4000
    
    noise_history = []
    in_r_history = []
    out_r_history = []
    eps_history = []
    
    final_pcd = None
    final_fit = None

    print("Starting Dynamic Noise Experiment...\n")

    # Slowly increase noise across 15 steps
    for step in range(15):
        shape_noise = step * 0.1
        env_pts = step * 1000  # Increase clutter by 1000 each step
        
        pcd = generate_dummy_cloud(INNER_R, OUTER_R, NUM_ANNULUS_POINTS, shape_noise, env_pts)
        
        # Default starting parameters
        eps = 0.5
        min_pts = 20
        
        in_r, out_r, in_xc, in_yc, out_xc, out_yc = attempt_fit(pcd, eps, min_pts)
        
        # Check if the algorithm failed (deviated more than 20% from truth)
        def is_failed(ir, orad):
            if ir is None or orad is None: return True
            if abs(ir - INNER_R)/INNER_R > 0.20: return True
            if abs(orad - OUTER_R)/OUTER_R > 0.20: return True
            return False

        if is_failed(in_r, out_r):
            print(f"[Noise {shape_noise:.1f} | Env {env_pts}] -> FAIL. Triggering Dynamic Recovery...")
            recovered = False
            
            # Grid search for new stable parameters
            test_eps_values = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
            test_min_values = [10, 20, 30, 50, 100]
            
            for test_eps in test_eps_values:
                for test_min in test_min_values:
                    tir, tor, tix, tiy, tox, toy = attempt_fit(pcd, test_eps, test_min)
                    if not is_failed(tir, tor):
                        print(f"    [+] RECOVERED! New params -> eps={test_eps}, min_pts={test_min}")
                        in_r, out_r, in_xc, in_yc, out_xc, out_yc = tir, tor, tix, tiy, tox, toy
                        eps = test_eps
                        min_pts = test_min
                        recovered = True
                        break
                if recovered: break
                
            if not recovered:
                print(f"    [-] RECOVERY FAILED. Lost track of the shape entirely.")
                in_r, out_r = 0.0, 0.0 # Zero out for the plot
        else:
            print(f"[Noise {shape_noise:.1f} | Env {env_pts}] -> SUCCESS. Radii: In={in_r:.2f}, Out={out_r:.2f}")

        # Store data for plotting
        noise_history.append(shape_noise)
        in_r_history.append(in_r)
        out_r_history.append(out_r)
        eps_history.append(eps)
        
        # Save the worst/last iteration for Open3D
        if step == 14:
            final_pcd = pcd
            final_fit = (in_r, out_r, in_xc, in_yc, out_xc, out_yc)

    # --- Plotting the Results (Matplotlib) ---
    plt.figure(figsize=(10, 5))
    
    # Left Axis: Radii Tracking
    ax1 = plt.gca()
    ax1.plot(noise_history, in_r_history, 'b-o', label='Detected Inner Radius (Target 3.0)')
    ax1.plot(noise_history, out_r_history, 'r-o', label='Detected Outer Radius (Target 6.0)')
    ax1.axhline(y=3.0, color='b', linestyle='--', alpha=0.3)
    ax1.axhline(y=6.0, color='r', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Shape Noise Level')
    ax1.set_ylabel('Radius Magnitude')
    ax1.set_ylim(0, 10)
    ax1.legend(loc='upper left')
    
    # Right Axis: Dynamic EPS Tracking
    ax2 = ax1.twinx()
    ax2.plot(noise_history, eps_history, 'g--x', label='Dynamic EPS used')
    ax2.set_ylabel('EPS (Bridging Distance)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')
    
    plt.title('Algorithm Stability & Dynamic Parameter Recovery')
    plt.grid(True, alpha=0.3)
    
    # Show plot without blocking the rest of the script
    plt.show(block=False) 
    
    # --- Show the Worst-Case Cloud (Open3D) ---
    print("\nLaunching Open3D Viewer for the Final (Noisiest) Step...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Max Noise Final State", width=1280, height=720)
    vis.add_geometry(final_pcd)
    
    # If the last step successfully recovered, draw the circles
    if final_fit[0] > 0: 
        in_r, out_r, in_xc, in_yc, out_xc, out_yc = final_fit
        zc = np.mean(np.asarray(final_pcd.points)[:, 2])
        inner_viz = create_drawn_circle(in_xc, in_yc, zc, in_r)
        outer_viz = create_drawn_circle(out_xc, out_yc, zc, out_r)
        vis.add_geometry(inner_viz)
        vis.add_geometry(outer_viz)
        
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 2.0
    opt.line_width = 5.0
    
    vis.run()
    vis.destroy_window()
    
    # Keep the matplotlib window open after Open3D closes
    plt.show()

if __name__ == "__main__":
    run_experiment_and_visualize()