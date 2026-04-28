import open3d as o3d
import numpy as np

def generate_dummy_cloud(inner_r, outer_r, num_points, noise, env_points):
    """Generates the noisy annulus and environment clutter."""
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.sqrt(np.random.uniform(inner_r**2, outer_r**2, num_points))
    
    x = r * np.cos(theta) + np.random.normal(0, noise, num_points)
    y = r * np.sin(theta) + np.random.normal(0, noise, num_points)
    z = np.zeros(num_points) + np.random.normal(0, noise/2, num_points)
    
    annulus_pts = np.vstack((x, y, z)).T
    
    spread = outer_r * 3.0
    env_x = np.random.uniform(-spread, spread, env_points)
    env_y = np.random.uniform(-spread, spread, env_points)
    env_z = np.random.uniform(-spread/3, spread/3, env_points)
    env_pts = np.vstack((env_x, env_y, env_z)).T
    
    final_pts = np.vstack((annulus_pts, env_pts))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.paint_uniform_color([1.0, 1.0, 1.0]) # Changed: Pure white for base cloud
    return pcd

def fit_circle_least_squares(xy_points):
    """
    Applies algebraic Least Squares to fit a circle to 2D points.
    Solves the equation: x^2 + y^2 = c1*x + c2*y + c3
    """
    x = xy_points[:, 0]
    y = xy_points[:, 1]
    
    # Set up the matrices A and B
    A = np.c_[x, y, np.ones(len(x))]
    B = x**2 + y**2
    
    # Solve the linear system
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # Extract center and radius
    xc = C[0] / 2
    yc = C[1] / 2
    radius = np.sqrt(C[2] + xc**2 + yc**2)
    
    return xc, yc, radius

def create_drawn_circle(xc, yc, zc, radius, color=[0.0, 0.8, 1.0], num_edges=100):
    """Creates a thick circle using Open3D LineSet for visualization."""
    theta = np.linspace(0, 2 * np.pi, num_edges)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    z = np.full_like(x, zc)
    
    points = np.c_[x, y, z]
    lines = [[i, (i + 1) % num_edges] for i in range(num_edges)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    return line_set

def detect_annulus_least_squares(pcd):
    print("Isolating object from environment...")
    # Use DBSCAN to find the main shape, ignoring scattered environment noise
    labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=20))
    if len(labels) == 0 or labels.max() == -1:
        print("Failed to isolate object.")
        return
        
    # Find the largest cluster (assuming it's our annulus)
    largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
    object_indices = np.where(labels == largest_cluster_idx)[0]
    obj_points = np.asarray(pcd.points)[object_indices]
    
    # Flatten to 2D (ignoring Z for circle fitting)
    xy_points = obj_points[:, :2]
    zc = np.mean(obj_points[:, 2]) # Keep average Z height for drawing later
    
    # 1. Angular Slicing to find boundaries
    print("Extracting inner and outer boundary points...")
    center_guess = np.mean(xy_points, axis=0)
    shifted = xy_points - center_guess
    r = np.linalg.norm(shifted, axis=1)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    
    num_bins = 72 # 5-degree slices
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    indices = np.digitize(theta, bins)
    
    inner_boundary = []
    outer_boundary = []
    
    for i in range(1, num_bins + 1):
        bin_mask = (indices == i)
        if not np.any(bin_mask): continue
        
        # Get the relative indices within this cluster
        relative_indices = np.where(bin_mask)[0]
        bin_r = r[bin_mask]
        
        # Find the min and max radius indices within this bin
        min_idx = relative_indices[np.argmin(bin_r)]
        max_idx = relative_indices[np.argmax(bin_r)]
        
        # Store the points for math
        inner_boundary.append(xy_points[min_idx])
        outer_boundary.append(xy_points[max_idx])
        
    inner_boundary = np.array(inner_boundary)
    outer_boundary = np.array(outer_boundary)

    # 2. Least Squares Fitting
    print("Running Least Squares Optimization...")
    in_xc, in_yc, in_r = fit_circle_least_squares(inner_boundary)
    out_xc, out_yc, out_r = fit_circle_least_squares(outer_boundary)
    
    print(f"Fitted Inner Circle: Center=({in_xc:.2f}, {in_yc:.2f}), Radius={in_r:.2f}")
    print(f"Fitted Outer Circle: Center=({out_xc:.2f}, {out_yc:.2f}), Radius={out_r:.2f}")

    # 3. Visualization
    # Draw the bright blue circles
    inner_circle_viz = create_drawn_circle(in_xc, in_yc, zc, in_r, color=[0.0, 0.8, 1.0])
    outer_circle_viz = create_drawn_circle(out_xc, out_yc, zc, out_r, color=[0.0, 0.8, 1.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Least Squares Annulus Fit", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(inner_circle_viz)
    vis.add_geometry(outer_circle_viz)
    
    # Changed: Set background to black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 3.0
    opt.line_width = 5.0 
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # --- YOUR EXPERIMENT PARAMETERS ---
    INNER_R = 3.0
    OUTER_R = 6.0
    NUM_ANNULUS_POINTS = 4000
    SHAPE_NOISE = 0.5    # Warps the annulus
    NUM_ENV_POINTS = 4000     # Scatters random points around the scene
    
    print(f"Generating Scene: Inner={INNER_R}, Outer={OUTER_R}")
    print(f"Points: Annulus={NUM_ANNULUS_POINTS}, Env={NUM_ENV_POINTS}, Noise={SHAPE_NOISE}")
    
    cloud = generate_dummy_cloud(
        inner_r=INNER_R, 
        outer_r=OUTER_R, 
        num_points=NUM_ANNULUS_POINTS, 
        noise=SHAPE_NOISE, 
        env_points=NUM_ENV_POINTS
    )
    
    # Run the detector
    detect_annulus_least_squares(cloud)