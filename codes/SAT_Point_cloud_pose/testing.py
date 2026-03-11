import pyzed.sl as sl
import sys

def main():
    # 1. Create a ZED Camera object
    zed = sl.Camera()

    # 2. Set Initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    # THE CATCH: Clamp maximum depth distance to 1 meter.
    # This prevents the camera from generating point cloud data beyond 1m,
    # meaning the stitched map will only contain points < 1m.
    init_params.depth_maximum_distance = 100 

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {err}")
        sys.exit(1)

    # 3. Enable Positional Tracking (IMU + Visual Odometry)
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Tracking initialization failed: {err}")
        zed.close()
        sys.exit(1)

    # 4. Enable Spatial Mapping to stitch the point cloud in real-time
    mapping_params = sl.SpatialMappingParameters()
    # Explicitly tell the mapping module we want a Point Cloud, not a Mesh
    mapping_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    
    err = zed.enable_spatial_mapping(mapping_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Spatial mapping initialization failed: {err}")
        zed.close()
        sys.exit(1)

    print("Started capturing. Move the camera around to scan.")
    print("Press Ctrl+C to stop recording and save the .ply file.")

    # 5. Capture Loop
    try:
        while True:
            # Grab a new frame. The Spatial Mapping module automatically 
            # takes the depth map and tracks it in the background.
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # You can monitor mapping state here if needed
                mapping_state = zed.get_spatial_mapping_state()
                print(f"\rMapping State: {mapping_state}", end="")
                
    except KeyboardInterrupt:
        print("\n\nCapture stopped by user. Processing final stitched point cloud...")

    # 6. Extract and save the Fused Point Cloud
    fused_pc = sl.FusedPointCloud()
    
    # Extract the whole spatial map from the background thread
    zed.extract_whole_spatial_map(fused_pc)

    # Save to a .ply file
    save_path = "stitched_point_cloud_under_1m.ply"
    fused_pc.save(save_path)
    print(f"Success! Point cloud saved to: {save_path}")

    # 7. Clean up and close
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    main()