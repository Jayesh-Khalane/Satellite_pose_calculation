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
    
    # Create coordinate frame
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(origin)

    # --- RENDER OPTIONS ---
    opt = vis.get_render_option()
    
    # 1. Set Background to Solid Black
    opt.background_color = np.asarray([0, 0, 0])
    
    # 2. Set Point Size (Lower = Smaller)
    # 1.0 is standard, try 0.5 or 0.1 for very high density clouds
    opt.point_size = 1.0 
    
    # 3. Make points appear round and smooth
    # This prevents the "square pixel" look
    opt.point_show_normal = False
    # Use this to enable smooth circular point rendering
    # Note: Compatibility depends on your GPU drivers
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
    path_to_ply = r"logs\data\sat.ply"
    visualize_ply(path_to_ply)