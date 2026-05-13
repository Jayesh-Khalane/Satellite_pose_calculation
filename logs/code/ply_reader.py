import open3d as o3d
import os
import numpy as np

def visualize_ply(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Loading PLY file: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    if pcd.is_empty():
        print("Error: Point cloud is empty or format is incompatible.")
        return
    
    print(f"Successfully loaded {len(pcd.points)} points.")

    # --- ADVANCED VISUALIZATION SETUP ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Satellite PLY Viewer", width=1280, height=720)
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # --- ORIGIN & ORIENTATION MARKERS ---
    # 1. The Coordinate Frame (Orientation Marker)
    # Increased size from 1.0 to 5.0 so it protrudes past the satellite points.
    # Red = X-axis, Green = Y-axis, Blue = Z-axis
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 2. A distinct sphere to mark the absolute origin point
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    origin_sphere.translate([0.0, 0.0, 0.0])
    origin_sphere.paint_uniform_color([1.0, 1.0, 1.0]) # Solid white
    vis.add_geometry(origin_sphere)

    # --- RENDER OPTIONS ---
    opt = vis.get_render_option()
    
    # 1. Set Background to Solid Black
    opt.background_color = np.asarray([0, 0, 0])
    # 2. Set Point Size (Lower = Smaller)
    opt.point_size = 1.0 
    
    # 3. Make points appear round and smooth
    opt.point_show_normal = False
    try:
        opt.point_smooth = True 
    except:
        pass

    print("Opening 3D Viewer...")
    print("Controls: Mouse to rotate, Shift+Mouse to pan, Scroll to zoom.")
    print("Keyboard: Use '-' or '+' to adjust point size live.")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    path_to_ply = r"logs/data/to_align_half_sat.ply"  # Update this path to your PLY file
    visualize_ply(path_to_ply)