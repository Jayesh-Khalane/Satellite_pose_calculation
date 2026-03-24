import pyzed.sl as sl
import open3d as o3d
import numpy as np
import os

def run_live_scanner():
    # 1. Initialize ZED 2i
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED.")
        return

    # 2. Enable Tracking and Spatial Mapping (This handles the automatic stitching/registration)
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    
    map_params = sl.SpatialMappingParameters()
    map_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    map_params.range_meter = 2.0  
    map_params.resolution_meter = 0.01 
    zed.enable_spatial_mapping(map_params)

    runtime_params = sl.RuntimeParameters()

    print("\n--- SCANNING LIVE ---")
    print("Move the camera slowly around the object.")
    print("Press Ctrl+C in this terminal to stop scanning and generate the 3D model.")

    # 3. Scanning Loop (No Visualization, just data capture)
    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # We don't need to extract the map every frame. 
                # ZED is building it in the background automatically.
                pass
    except KeyboardInterrupt:
        print("\n\nScanning stopped by user. Processing 3D Model...")

    # 4. Extract the fully stitched map
    fused_cloud = sl.FusedPointCloud()
    zed.extract_whole_spatial_map(fused_cloud)
    
    raw_vertices = fused_cloud.vertices
    if raw_vertices is not None and len(raw_vertices) > 0 and raw_vertices.shape[1] == 4:
        print(f"Total points captured: {len(raw_vertices)}")
        
        # Split Coordinates
        xyz = raw_vertices[:, :3]
        
        # Unpack True RGB Colors (BGRA format)
        rgba = raw_vertices[:, 3].copy().view(np.uint32)
        b = (rgba & 0x000000FF)
        g = ((rgba & 0x0000FF00) >> 8)
        r = ((rgba & 0x00FF0000) >> 16)
        colors = np.vstack((r, g, b)).T / 255.0 
        
        # 5. Filter Points strictly within 100cm of the starting position
        print("Filtering points outside the 100cm radius...")
        distances = np.linalg.norm(xyz, axis=1)
        mask = distances < 100.0  
        
        filtered_xyz = xyz[mask]
        filtered_colors = colors[mask]
        
        print(f"Points remaining after 100cm filter: {len(filtered_xyz)}")

        # 6. Save using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        save_path = r"logs\data\live_scan.ply"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        
        print(f"SUCCESS: Filtered RGB Scan saved to {save_path}")
    else:
        print("Error: No spatial map could be extracted.")

    # Cleanup
    zed.disable_spatial_mapping()
    zed.close()

if __name__ == "__main__":
    run_live_scanner()