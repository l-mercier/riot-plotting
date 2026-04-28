#!/usr/bin/env python3
"""
Script to compute and visualize intensity from dog motion data
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from intensity import (
    process_intensity_pipeline, 
    IntensityMode, 
    NormMode,
    compute_derivatives,
    apply_moving_average
)

# Set seaborn style
sns.set_theme(style="darkgrid")

# Load the JSON file
json_path = "/Users/mercier/Documents/Projects for after/AnimalMotion/recordings/dog.json"
print(f"Loading {json_path}...")

with open(json_path, 'r') as f:
    data = json.load(f)

# Extract the sig1 track (motion data)
sig1_track = None
for track in data['tracks']:
    if track['name'] == 'sig1':
        sig1_track = track
        break

if sig1_track is None:
    print("sig1 track not found!")
    exit(1)

# Extract motion data
mxData = sig1_track['buffers'][0]['mxData']
sample_rate = sig1_track['buffers'][0]['sampleRate']
num_cols = sig1_track['mxCols']

print(f"Sample rate: {sample_rate} Hz")
print(f"Channels: {num_cols}")

# Reshape into DataFrame
num_samples = len(mxData) // num_cols
motion_data = np.array([mxData[i:i+num_cols] for i in range(0, len(mxData), num_cols)])
df = pd.DataFrame(motion_data, columns=['X', 'Y', 'Z'])
df['Time'] = df.index / sample_rate

print(f"Data shape: {df.shape}")
print(f"Duration: {df['Time'].iloc[-1]:.2f} seconds\n")

# ==================== COMPUTE INTENSITY ====================
print("Computing intensity metrics...")

# Configuration
MOVING_AVG_SIZE = 3
DELTA_SIZE = 5
CUT_FREQUENCY = 10.0
GAIN = 1.0
POWER_EXP = 1.0

# Process with different normalization modes
intensity_data = {}

# Mode 1: L2 Pre-normalization (L2 norm before processing)
print("  - L2 Pre-normalization...")
intensity_l2pre, deriv_l2pre, smooth_l2pre = process_intensity_pipeline(
    df[['X', 'Y', 'Z']].values,
    sample_rate=sample_rate,
    moving_average_size=MOVING_AVG_SIZE,
    delta_size=DELTA_SIZE,
    norm_mode=NormMode.L2_PRE,
    cut_frequency=CUT_FREQUENCY,
    gain=GAIN,
    power_exp=POWER_EXP,
    mode=IntensityMode.ABS
)
intensity_data['L2_Pre'] = intensity_l2pre

# Mode 2: Absolute value mode (simpler intensity)
print("  - Absolute value mode...")
intensity_abs, deriv_abs, smooth_abs = process_intensity_pipeline(
    df[['X', 'Y', 'Z']].values,
    sample_rate=sample_rate,
    moving_average_size=MOVING_AVG_SIZE,
    delta_size=DELTA_SIZE,
    norm_mode=NormMode.L2_PRE,
    cut_frequency=CUT_FREQUENCY,
    gain=1.0,
    power_exp=2.0,  # quadratic response
    mode=IntensityMode.ABS
)
intensity_data['Absolute_Squared'] = intensity_abs

# Add to dataframe
df['Intensity_L2Pre'] = intensity_l2pre
df['Intensity_Squared'] = intensity_abs
df['DerivX'] = deriv_l2pre[:, 0]
df['DerivY'] = deriv_l2pre[:, 1]
df['DerivZ'] = deriv_l2pre[:, 2]

print(f"\nIntensity statistics:")
print(f"  L2 Pre - Min: {intensity_l2pre.min():.4f}, Max: {intensity_l2pre.max():.4f}, Mean: {intensity_l2pre.mean():.4f}")
print(f"  Squared - Min: {intensity_abs.min():.4f}, Max: {intensity_abs.max():.4f}, Mean: {intensity_abs.mean():.4f}")

# ==================== VISUALIZATION ====================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot 1: Raw position data
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Time'], df['X'], label='X', alpha=0.7, linewidth=0.8)
ax1.plot(df['Time'], df['Y'], label='Y', alpha=0.7, linewidth=0.8)
ax1.plot(df['Time'], df['Z'], label='Z', alpha=0.7, linewidth=0.8)
ax1.set_ylabel('Position')
ax1.set_title('Raw Position Data Over Time', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Derivatives (acceleration)
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(df['Time'], df['DerivX'], label='dX/dt', alpha=0.7, linewidth=0.8)
ax2.plot(df['Time'], df['DerivY'], label='dY/dt', alpha=0.7, linewidth=0.8)
ax2.plot(df['Time'], df['DerivZ'], label='dZ/dt', alpha=0.7, linewidth=0.8)
ax2.set_ylabel('Velocity/Acceleration')
ax2.set_title('Derivatives (Velocity/Acceleration) Over Time', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Intensity - L2 Pre
ax3 = fig.add_subplot(gs[2, 0])
ax3.fill_between(df['Time'], intensity_l2pre, alpha=0.6, color='steelblue')
ax3.plot(df['Time'], intensity_l2pre, color='steelblue', linewidth=1)
ax3.set_ylabel('Intensity')
ax3.set_title('Intensity (L2 Pre-norm)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Intensity - Squared
ax4 = fig.add_subplot(gs[2, 1])
ax4.fill_between(df['Time'], intensity_abs, alpha=0.6, color='coral')
ax4.plot(df['Time'], intensity_abs, color='coral', linewidth=1)
ax4.set_ylabel('Intensity')
ax4.set_title('Intensity (Squared)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Combined intensities
ax5 = fig.add_subplot(gs[3, 0])
ax5.plot(df['Time'], intensity_l2pre, label='L2 Pre', alpha=0.8, linewidth=1)
ax5.plot(df['Time'], intensity_abs, label='Squared', alpha=0.8, linewidth=1)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Intensity')
ax5.set_title('Intensity Comparison', fontsize=11, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Heatmap of derivatives over time (spectrogram-like)
ax6 = fig.add_subplot(gs[3, 1])
# Downsample for visualization
downsample = 100
time_downsampled = df['Time'].values[::downsample]
deriv_matrix = np.array([df['DerivX'].values[::downsample],
                         df['DerivY'].values[::downsample],
                         df['DerivZ'].values[::downsample]])
im = ax6.imshow(deriv_matrix, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                 extent=[time_downsampled[0], time_downsampled[-1], -0.5, 2.5])
ax6.set_yticks([0, 1, 2])
ax6.set_yticklabels(['X', 'Y', 'Z'])
ax6.set_xlabel('Time (s)')
ax6.set_title('Derivatives Heatmap (Downsampled)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax6, label='Velocity')

fig.suptitle('Dog Motion Analysis - Intensity Computation', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('dog_intensity_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'dog_intensity_analysis.png'")
plt.show()

# ==================== EXPORT RESULTS ====================
# Save intensity data to CSV
output_df = df[['Time', 'X', 'Y', 'Z', 'DerivX', 'DerivY', 'DerivZ', 'Intensity_L2Pre', 'Intensity_Squared']].copy()
output_df.to_csv('dog_intensity.csv', index=False)
print("✓ Data exported to 'dog_intensity.csv'")

# Summary statistics
print("\n" + "="*60)
print("INTENSITY ANALYSIS SUMMARY")
print("="*60)
print(f"Total duration: {df['Time'].iloc[-1]:.2f} seconds")
print(f"Sample rate: {sample_rate} Hz")
print(f"Total samples: {len(df)}")

print(f"\nIntensity Statistics (L2 Pre-normalization):")
print(f"  Mean: {intensity_l2pre.mean():.6f}")
print(f"  Median: {np.median(intensity_l2pre):.6f}")
print(f"  Std Dev: {intensity_l2pre.std():.6f}")
print(f"  Min: {intensity_l2pre.min():.6f}")
print(f"  Max: {intensity_l2pre.max():.6f}")

print(f"\nIntensity Statistics (Squared):")
print(f"  Mean: {intensity_abs.mean():.6f}")
print(f"  Median: {np.median(intensity_abs):.6f}")
print(f"  Std Dev: {intensity_abs.std():.6f}")
print(f"  Min: {intensity_abs.min():.6f}")
print(f"  Max: {intensity_abs.max():.6f}")

# Find peaks (high intensity moments)
threshold_l2 = intensity_l2pre.mean() + 2 * intensity_l2pre.std()
peaks_l2 = np.where(intensity_l2pre > threshold_l2)[0]
print(f"\nHigh intensity moments (L2 Pre > mean + 2σ):")
print(f"  Count: {len(peaks_l2)} samples ({100*len(peaks_l2)/len(intensity_l2pre):.1f}%)")
if len(peaks_l2) > 0:
    print(f"  Times: {df['Time'].iloc[peaks_l2].values[:10]} ... (first 10)")
