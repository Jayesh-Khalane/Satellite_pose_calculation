import cv2
import numpy as np
import sys

# --- 1. ZED 2i 1080p Camera Parameters ---
fx = 1050.0       # Focal length in pixels for 1080p
baseline = 0.12   # Distance between lenses in meters

# --- Global Variables ---
current_left_gray = None
current_right_gray = None
# Stores dictionary of matched pairs: {'left_pt': (x,y), 'right_pt': (x,y), 'depth': z}
measured_points = [] 

def find_feature_match(source_img, search_img, x, y, direction):
    """
    Extracts a 31x31 pixel patch around the clicked point and 
    searches for it in the other image using Normalized Cross-Correlation.
    """
    patch_size = 31
    half = patch_size // 2
    
    # Boundary check to ensure we don't extract outside the image
    if x - half < 0 or y - half < 0 or x + half >= source_img.shape[1] or y + half >= source_img.shape[0]:
        return None

    # Extract the feature template
    patch = source_img[y-half : y+half+1, x-half : x+half+1]
    
    # We allow a small vertical tolerance (+/- 15px) in case the lenses 
    # are slightly misaligned vertically (since we lack SDK rectification)
    y_search_radius = 15 
    y_min = max(0, y - half - y_search_radius)
    y_max = min(search_img.shape[0], y + half + y_search_radius + 1)
    
    # Horizontal search constraints (Maximum search range = 300 pixels shift)
    max_disp = 300
    if direction == "left_to_right":
        # The object in the right eye is always shifted left relative to the left eye
        x_min = max(0, x - max_disp - half)
        x_max = min(search_img.shape[1], x + half)
    else: 
        # The object in the left eye is always shifted right relative to the right eye
        x_min = max(0, x - half)
        x_max = min(search_img.shape[1], x + max_disp + half)

    search_roi = search_img[y_min:y_max, x_min:x_max]
    
    if search_roi.shape[0] < patch_size or search_roi.shape[1] < patch_size:
        return None

    # Perform Template Matching
    res = cv2.matchTemplate(search_roi, patch, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, max_loc = cv2.minMaxLoc(res)
    
    # Reject weak matches (e.g., clicking on a blank white wall)
    if confidence < 0.60: 
        return None
        
    # Translate ROI coordinates back to full image coordinates
    match_x = x_min + max_loc[0] + half
    match_y = y_min + max_loc[1] + half
    
    # Calculate Disparity and Depth
    disparity = (x - match_x) if direction == "left_to_right" else (match_x - x)
    if disparity <= 0: 
        return None
        
    depth = (fx * baseline) / disparity
    return match_x, match_y, depth

def mouse_callback(event, x, y, flags, param):
    global measured_points
    
    # Right Click -> Clear all points
    if event == cv2.EVENT_RBUTTONDOWN:
        measured_points.clear()
        print("\n--- All points cleared ---")
        
    # Left Click -> Find Match and Add Point
    elif event == cv2.EVENT_LBUTTONDOWN:
        if current_left_gray is None or current_right_gray is None:
            return
            
        # Determine which image was clicked based on the X coordinate
        if x < 1920:  # Clicked Left Image
            match = find_feature_match(current_left_gray, current_right_gray, x, y, "left_to_right")
            if match:
                mx, my, depth = match
                measured_points.append({'left_pt': (x, y), 'right_pt': (mx, my), 'depth': depth})
                print(f"Matched Feature (L->R) | Depth: {depth:.2f} meters")
            else:
                print("Failed to find a strong feature match. Try clicking an edge or textured surface.")
        
        else:  # Clicked Right Image
            rx = x - 1920  # Adjust X coordinate relative to the right frame
            match = find_feature_match(current_right_gray, current_left_gray, rx, y, "right_to_left")
            if match:
                mx, my, depth = match
                measured_points.append({'left_pt': (mx, my), 'right_pt': (rx, y), 'depth': depth})
                print(f"Matched Feature (R->L) | Depth: {depth:.2f} meters")
            else:
                print("Failed to find a strong feature match. Try clicking an edge or textured surface.")


def main():
    global current_left_gray, current_right_gray
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    window_name = "ZED 2i Interactive Stereo (L: Click | R: Clear)"
    
    # Allow the window to be resized so it fits on standard monitors
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 540) # Scale GUI down to 50% for viewing
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=========================================================")
    print("  Interactive Feature Matching Pipeline Running        ")
    print("=========================================================")
    print("-> LEFT CLICK anywhere to find feature & calculate depth.")
    print("-> RIGHT CLICK to wipe all points off the screen.")
    print("-> PRESS 'q' to exit.\n")

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    color_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        left_img = frame[:, :1920]
        right_img = frame[:, 1920:]
        
        current_left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        current_right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Draw all saved points and connecting graphics
        display_frame = frame.copy()
        
        for i, pt_data in enumerate(measured_points):
            lx, ly = pt_data['left_pt']
            rx, ry = pt_data['right_pt']
            depth = pt_data['depth']
            
            # Right eye coordinates are offset by 1920 on the combined canvas
            rx_canvas = rx + 1920 
            color = colors[i % len(colors)]

            # Draw circles on both eyes
            cv2.circle(display_frame, (lx, ly), 6, color, -1)
            cv2.circle(display_frame, (rx_canvas, ry), 6, color, -1)
            
            # Draw a line linking the two matched features
            cv2.line(display_frame, (lx, ly), (rx_canvas, ry), color, 1)

            # Draw text labels
            text = f"P{i}: {depth:.2f}m"
            cv2.putText(display_frame, text, (lx - 30, ly - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, text, (rx_canvas - 30, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()