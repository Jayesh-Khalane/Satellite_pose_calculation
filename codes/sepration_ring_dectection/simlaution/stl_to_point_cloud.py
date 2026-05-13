import open3d as o3d
import numpy as np
import time

# 1. Load the CAD Model (MUST be .stl or .obj)
file_path = 'D:/ring.stl'
print(f"Loading {file_path}...")
mesh = o3d.io.read_triangle_mesh(file_path)
mesh.compute_vertex_normals()

# Auto-center and calculate radius
mesh.translate(-mesh.get_center())
max_bound = mesh.get_max_bound()
min_bound = mesh.get_min_bound()
radius = np.max(max_bound - min_bound) * 1.5

# 2. Setup Open3D Raycasting Engine
# Convert legacy mesh to the new Tensor format for fast raycasting
t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(t_mesh)

# 3. Setup the Live Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Live LiDAR Scan (Press ESC to Stop and Save)", width=1024, height=768)

# Add the original mesh (painted gray, slightly transparent)
mesh.paint_uniform_color([0.5, 0.5, 0.5])
vis.add_geometry(mesh)

# Create an empty point cloud to hold our live scan data
live_pcd = o3d.geometry.PointCloud()
vis.add_geometry(live_pcd)

all_points = []
angle = 0.0
speed = 0.1 # Radians to move per frame

print("\n Scanning started! Look at the pop-up window.")
print(" Press 'ESC' in the window or 'CTRL+C' in the terminal to stop and save.")

try:
    # Loop until the window is closed
    keep_running = True
    while keep_running:
        # Move the "Camera" in a circle
        origin = np.array([radius * np.cos(angle), radius * np.sin(angle), radius * np.sin(angle*0.5)])
        
        # --- Fast Ray Generation ---
        # Create a vector pointing from camera to origin
        forward = -origin
        forward = forward / np.linalg.norm(forward)
        
        # Create local coordinate frame
        up = np.array([0, 0, 1])
        if np.abs(np.dot(forward, up)) > 0.99: up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        
        # Generate a grid of rays (150x150 = 22,500 rays per frame)
        fov = np.pi / 4
        theta = np.linspace(-fov, fov, 150)
        phi = np.linspace(-fov, fov, 150)
        T, P = np.meshgrid(theta, phi)
        
        # Calculate ray directions
        dx, dy, dz = np.cos(P) * np.sin(T), np.sin(P), np.cos(P) * np.cos(T)
        directions = np.outer(dx.flatten(), right) + np.outer(dy.flatten(), true_up) + np.outer(dz.flatten(), forward)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
        # Format for Open3D: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
        origins = np.tile(origin, (len(directions), 1))
        rays = np.hstack([origins, directions])
        ray_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        # --- Perform the Raycast ---
        # This takes milliseconds in Open3D!
        ans = scene.cast_rays(ray_tensor)
        
        # Extract points where the ray actually hit the object
        hit = ans['t_hit'].isfinite()
        hit_np = hit.numpy()
        
        if hit_np.any():
            # Calculate 3D coordinates: Point = Origin + Direction * Distance
            distances = ans['t_hit'].numpy()[hit_np].reshape(-1, 1)
            hit_points = origins[hit_np] + directions[hit_np] * distances
            
            all_points.append(hit_points)
            
            # Update the live visualizer
            current_cloud = np.vstack(all_points)
            live_pcd.points = o3d.utility.Vector3dVector(current_cloud)
            # Color the points neon green
            live_pcd.paint_uniform_color([0.1, 0.9, 0.1]) 
            
            vis.update_geometry(live_pcd)
        
        # Update the window and check if user pressed ESC
        keep_running = vis.poll_events()
        vis.update_renderer()
        
        angle += speed
        time.sleep(0.01) # Small delay to prevent locking up your CPU

except KeyboardInterrupt:
    # Catches CTRL+C in the terminal
    print("\n Interrupted by user (CTRL+C).")

finally:
    # This block ALWAYS runs, ensuring your data is saved
    vis.destroy_window()
    
    if all_points:
        final_cloud = np.vstack(all_points)
        # Remove duplicate/overlapping points
        final_cloud = np.unique(np.round(final_cloud, decimals=4), axis=0)
        
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(final_cloud)
        
        # Save to PLY
        save_path = "satellite_complete_scan.ply"
        o3d.io.write_point_cloud(save_path, save_pcd)
        print(f"Successfully saved {len(final_cloud)} points to '{save_path}'")
    else:
        print(" No points were captured.")