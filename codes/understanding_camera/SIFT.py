import pyzed.sl as sl
import sys

def main():
    # 1. Initialize the Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER # Crucial for real-world scale
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS # High quality depth for mapping

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {err}")
        sys.exit(1)

    # 2. Enable Positional Tracking (Visual-Inertial Odometry + Loop Closure)
    tracking_parameters = sl.PositionalTrackingParameters()
    tracking_parameters.enable_area_memory = True # Loop Closure / Pose Graph Optimization
    
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Tracking error: {err}")
        zed.close()
        sys.exit(1)

    # 3. Enable Spatial Mapping (Building the 3D Map)
    mapping_parameters = sl.SpatialMappingParameters()
    mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    
    # [FIXED SDK 4.x LINES] -> Using 4.x Enums instead of array indexing
    mapping_parameters.resolution_meter = mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
    mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.MEDIUM)
    # Note: You can change MEDIUM to LOW or HIGH depending on how large your room is
    
    err = zed.enable_spatial_mapping(mapping_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Mapping error: {err}")
        zed.disable_positional_tracking()
        zed.close()
        sys.exit(1)

    # Variables to store data
    pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters()
    
    print("VI-SLAM Started! Move the camera around the room to build the map.")
    print("Loop closure is ACTIVE. Walk in a circle and return to your start point to optimize the graph.")
    print("Press Ctrl+C in the terminal to stop and save the map.")

    try:
        frames_captured = 0
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get the camera's pose (Position + Orientation) fused with IMU
                tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    frames_captured += 1
                    
                    # Print position every 30 frames (approx 1 second)
                    if frames_captured % 30 == 0:
                        translation = pose.get_translation().get()
                        print(f"Camera Position -> X: {translation[0]:.2f}m, Y: {translation[1]:.2f}m, Z: {translation[2]:.2f}m")
                        
                        # Check the status of the spatial mapping
                        map_status = zed.get_spatial_mapping_state()
                        print(f"Mapping State: {map_status}")

    except KeyboardInterrupt:
        print("\nMapping interrupted by user. Finalizing Pose Graph and saving map...")

    # 4. Extract and Save the 3D Map
    print("Extracting Point Cloud... This might take a moment depending on the map size.")
    point_cloud_map = sl.FusedPointCloud()
    zed.extract_whole_spatial_map(point_cloud_map)
    
    output_filename = "my_vi_slam_map.ply"
    point_cloud_map.save(output_filename)
    print(f"Success! 3D Map saved to: {output_filename}")
    print("You can open this file using MeshLab, CloudCompare, or Open3D.")

    # 5. Clean up
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    main()