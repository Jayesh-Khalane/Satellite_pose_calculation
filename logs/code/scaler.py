import pandas as pd
import os

def scale_and_save_new_csv(input_path, output_path, real_cm, colmap_units):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Calculate Scale Factor
    # If 0.96374 units = 10cm, then 1 unit = 10.376... cm
    scale_factor = real_cm / colmap_units
    
    print(f"--- Scaling Configuration ---")
    print(f"Target Real World: {real_cm} cm")
    print(f"COLMAP Units:      {colmap_units}")
    print(f"Multiplier:        {scale_factor:.6f}")
    print("-----------------------------")

    # 2. Load the Cleaned Data
    df = pd.read_csv(input_path)

    # 3. Create a Copy and Apply Scaling
    # We only scale the spatial coordinates (X, Y, Z)
    scaled_df = df.copy()
    scaled_df['X'] = df['X'] * scale_factor
    scaled_df['Y'] = df['Y'] * scale_factor
    scaled_df['Z'] = df['Z'] * scale_factor

    # 4. Save to the NEW CSV file
    scaled_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"SUCCESS: Scaled {len(scaled_df)} points.")
    print(f"New file saved at: {output_path}")
    print("The units in this file are now in CENTIMETERS.")

if __name__ == "__main__":
    # Define your paths
    input_file = r"logs\data\unscaled_satellite.csv"
    output_file = r"logs\data\scaled_satellite.csv"
    
    # Your specific scale: 10cm = 0.833503 units
    target_cm = 10.0
    measured_units = 0.833503
    
    scale_and_save_new_csv(input_file, output_file, target_cm, measured_units)