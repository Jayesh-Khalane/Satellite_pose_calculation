import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_pose(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please check your file path.")
        return

    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("Error: CSV file is empty. No data to plot.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle("Pose Error Analysis ", fontsize=18, fontweight='bold')

    columns = ["x_cm", "y_cm", "z_cm", "roll_deg", "pitch_deg", "yaw_deg"]
    titles = ["X Translation", "Y Translation", "Z Translation", "Roll", "Pitch", "Yaw"]
    units = ["cm", "cm", "cm", "deg", "deg", "deg"]
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#bcbd22', '#17becf']

    for i, col in enumerate(columns):
        ax = axes[i % 3, i // 3]
        
        try:
            data = df[col].values
            t = df["timestamp"].values
        except KeyError:
            print(f"Error: Column '{col}' not found in CSV. Check your headers.")
            continue

        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        p2p_val = max_val - min_val
        std_dev = np.std(data)

        # Plot the data
        ax.plot(t, data, color=colors[i], linewidth=1.0, label=f"Raw {titles[i]}")
        ax.axhline(mean_val, color='black', linestyle='--', alpha=0.8, 
                   label=f"Mean: {mean_val:.3f}\nStdDev: {std_dev:.4f}")
        
        # ==========================================
        # FULLY DYNAMIC PROPORTIONAL SCALING
        # ==========================================
        if p2p_val == 0:
            # Fallback ONLY if the data is literally identical (0.0000 difference)
            y_margin = 0.5 
        else:
            # Always add exactly 25% visual padding based on the actual movement.
            # This forces the data to take up exactly ~66% of the visual space.
            y_margin = p2p_val * 0.25 
            
        ax.set_ylim([min_val - y_margin, max_val + y_margin])
        # ==========================================

        # Formatting
        ax.set_title(f"{titles[i]} | Peak-to-Peak Δ: {p2p_val:.4f} {units[i]}", fontsize=10, pad=10)
        ax.set_ylabel(f"Value ({units[i]})")
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize='7')

    # X-axis labels for the bottom plots
    for ax in axes[2, :]:
        ax.set_xlabel("Time (seconds)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig("pose_analysis.png", dpi=300)
    print("Plot generated and saved as 'pose_analysis.png'.")
    plt.show()

if __name__ == "__main__":
    # Ensure raw string (r"...") is used for Windows file paths
    analyze_pose(r"D:\Satellite_pose_calculation\pose_data.csv")