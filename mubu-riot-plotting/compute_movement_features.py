#!/usr/bin/env python3
"""
Script to compute movement features (Intensity, Jerkiness, Kinetic Energy, Fluidity Index)
from dog motion data using the integrated MovementFeatures module.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from movement_features import MovementFeatures, summarize_features
from intensity import compute_derivatives

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

# ==================== COMPUTE ACCELERATION ====================
print("Computing acceleration from position data...")
# Use the compute_derivatives function to get acceleration
acceleration = compute_derivatives(df[['X', 'Y', 'Z']].values, delta_size=5, sample_rate=sample_rate)
print(f"Acceleration shape: {acceleration.shape}")

# ==================== COMPUTE MOVEMENT FEATURES ====================
print("\nComputing movement features...")
processor = MovementFeatures(buffer_size=20, sample_rate=sample_rate)

# For this single-limb data, use unilateral processing
features = processor.process_unilateral_features(acceleration, sample_rate=sample_rate)

# Extract features
intensity = features['intensity']
jerkiness = features['jerkiness']
kinetic_energy = features['kinetic_energy']
fluidity = features['fluidity']

# Add to dataframe
df['Intensity'] = intensity
df['Jerkiness'] = jerkiness
df['Kinetic_Energy'] = kinetic_energy
df['Fluidity'] = fluidity
df['Acceleration_Magnitude'] = np.linalg.norm(acceleration, axis=1)

# Compute basic statistics
print("\nFeature Statistics:")
print("\nIntensity:")
print(f"  Mean: {np.mean(intensity):.6f}")
print(f"  Std:  {np.std(intensity):.6f}")
print(f"  Min:  {np.min(intensity):.6f}")
print(f"  Max:  {np.max(intensity):.6f}")

print("\nJerkiness:")
print(f"  Mean: {np.mean(jerkiness):.6f}")
print(f"  Std:  {np.std(jerkiness):.6f}")
print(f"  Min:  {np.min(jerkiness):.6f}")
print(f"  Max:  {np.max(jerkiness):.6f}")

print("\nKinetic Energy:")
print(f"  Mean: {np.mean(kinetic_energy):.6f}")
print(f"  Std:  {np.std(kinetic_energy):.6f}")
print(f"  Min:  {np.min(kinetic_energy):.6f}")
print(f"  Max:  {np.max(kinetic_energy):.6f}")

print("\nFluidity Index:")
print(f"  Mean: {np.mean(fluidity):.6f}")
print(f"  Std:  {np.std(fluidity):.6f}")
print(f"  Min:  {np.min(fluidity):.6f}")
print(f"  Max:  {np.max(fluidity):.6f}")

# ==================== ANALYZE MOVEMENT QUALITY ====================
print("\n" + "="*60)
print("MOVEMENT QUALITY ANALYSIS")
print("="*60)

# Categorize intensity levels
high_intensity = np.sum(intensity > np.percentile(intensity, 75))
medium_intensity = np.sum((intensity > np.percentile(intensity, 25)) & (intensity <= np.percentile(intensity, 75)))
low_intensity = np.sum(intensity <= np.percentile(intensity, 25))

total_samples = len(intensity)
print(f"\nIntensity Distribution:")
print(f"  High (>75th percentile):       {high_intensity:6d} samples ({100*high_intensity/total_samples:5.1f}%)")
print(f"  Medium (25-75th):              {medium_intensity:6d} samples ({100*medium_intensity/total_samples:5.1f}%)")
print(f"  Low (<25th percentile):        {low_intensity:6d} samples ({100*low_intensity/total_samples:5.1f}%)")

# Categorize fluidity levels
very_high_fluidity = np.sum(fluidity > np.percentile(fluidity, 90))
high_fluidity = np.sum((fluidity > np.percentile(fluidity, 75)) & (fluidity <= np.percentile(fluidity, 90)))
medium_fluidity = np.sum((fluidity > np.percentile(fluidity, 25)) & (fluidity <= np.percentile(fluidity, 75)))
low_fluidity = np.sum(fluidity <= np.percentile(fluidity, 25))

print(f"\nFluidity Distribution:")
print(f"  Very High (>75th percentile):  {very_high_fluidity:6d} samples ({100*very_high_fluidity/total_samples:5.1f}%)")
print(f"  High (50-75th):                {high_fluidity:6d} samples ({100*high_fluidity/total_samples:5.1f}%)")
print(f"  Medium (25-50th):              {medium_fluidity:6d} samples ({100*medium_fluidity/total_samples:5.1f}%)")
print(f"  Low (<25th percentile):        {low_fluidity:6d} samples ({100*low_fluidity/total_samples:5.1f}%)")

# Find movement peaks
high_energy_threshold = np.percentile(kinetic_energy, 90)
high_energy_idx = np.where(kinetic_energy > high_energy_threshold)[0]
print(f"\nHigh Energy Moments (>90th percentile): {len(high_energy_idx)} samples")

jerky_moments = np.where(jerkiness > np.percentile(jerkiness, 90))[0]
print(f"Jerky Moments (high jerkiness >90th): {len(jerky_moments)} samples")

smooth_moments = np.where(fluidity > np.percentile(fluidity, 90))[0]
print(f"Smooth Moments (high fluidity >90th): {len(smooth_moments)} samples")

intense_moments = np.where(intensity > np.percentile(intensity, 90))[0]
print(f"Intense Moments (high intensity >90th): {len(intense_moments)} samples")

# ==================== VISUALIZATION ====================
fig = plt.figure(figsize=(16, 16))
gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3)

# Plot 1: Raw position
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Time'], df['X'], label='X', alpha=0.7, linewidth=0.8)
ax1.plot(df['Time'], df['Y'], label='Y', alpha=0.7, linewidth=0.8)
ax1.plot(df['Time'], df['Z'], label='Z', alpha=0.7, linewidth=0.8)
ax1.set_ylabel('Position')
ax1.set_title('Raw Position Data', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Acceleration magnitude
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(df['Time'], df['Acceleration_Magnitude'], alpha=0.6, color='steelblue')
ax2.plot(df['Time'], df['Acceleration_Magnitude'], color='steelblue', linewidth=0.8)
ax2.axhline(np.percentile(df['Acceleration_Magnitude'], 90), color='red', linestyle='--', 
            alpha=0.7, label='90th percentile')
ax2.set_ylabel('Acceleration Magnitude')
ax2.set_title('Acceleration Over Time', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Intensity
ax3 = fig.add_subplot(gs[2, 0])
ax3.fill_between(df['Time'], intensity, alpha=0.6, color='purple')
ax3.plot(df['Time'], intensity, color='purple', linewidth=0.8)
ax3.axhline(np.percentile(intensity, 75), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_ylabel('Intensity (normalized)')
ax3.set_title('Motion Intensity', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Jerkiness
ax4 = fig.add_subplot(gs[2, 1])
ax4.fill_between(df['Time'], jerkiness, alpha=0.6, color='coral')
ax4.plot(df['Time'], jerkiness, color='coral', linewidth=0.8)
ax4.axhline(np.percentile(jerkiness, 90), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax4.set_ylabel('Jerkiness')
ax4.set_title('Jerkiness (Squared Jerk)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Kinetic Energy
ax5 = fig.add_subplot(gs[3, 0])
ax5.fill_between(df['Time'], kinetic_energy, alpha=0.6, color='mediumseagreen')
ax5.plot(df['Time'], kinetic_energy, color='mediumseagreen', linewidth=0.8)
ax5.axhline(np.percentile(kinetic_energy, 90), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax5.set_ylabel('Kinetic Energy (normalized)')
ax5.set_title('Kinetic Energy', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Fluidity Index
ax6 = fig.add_subplot(gs[3, 1])
sc = ax6.scatter(df['Time'], fluidity, c=fluidity, cmap='RdYlGn', s=5, alpha=0.7)
ax6.axhline(np.percentile(fluidity, 50), color='blue', linestyle='--', alpha=0.5, label='Median')
ax6.axhline(np.percentile(fluidity, 75), color='green', linestyle='--', alpha=0.5, label='75th')
ax6.axhline(np.percentile(fluidity, 25), color='red', linestyle='--', alpha=0.5, label='25th')
ax6.set_ylabel('Fluidity Index')
ax6.set_title('Fluidity Index (Lower Jerk / Higher Energy)', fontsize=11, fontweight='bold')
ax6.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax6, label='Fluidity')

# Plot 7: All features combined
ax7 = fig.add_subplot(gs[4, :])
ax7_twin1 = ax7.twinx()
ax7_twin2 = ax7.twinx()
ax7_twin2.spines['right'].set_position(('outward', 60))

p1, = ax7.plot(df['Time'], intensity, color='purple', linewidth=1, alpha=0.8, label='Intensity')
p2, = ax7_twin1.plot(df['Time'], jerkiness, color='coral', linewidth=1, alpha=0.8, label='Jerkiness')
p3, = ax7_twin2.plot(df['Time'], kinetic_energy, color='green', linewidth=1, alpha=0.8, label='Kinetic Energy')

ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Intensity', color='purple')
ax7_twin1.set_ylabel('Jerkiness', color='coral')
ax7_twin2.set_ylabel('Kinetic Energy', color='green')
ax7.tick_params(axis='y', labelcolor='purple')
ax7_twin1.tick_params(axis='y', labelcolor='coral')
ax7_twin2.tick_params(axis='y', labelcolor='green')
ax7.set_title('All Movement Features Combined', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

lines = [p1, p2, p3]
labels = [l.get_label() for l in lines]
ax7.legend(lines, labels, loc='upper right')

# Plot 8: Feature heatmap
ax8 = fig.add_subplot(gs[5, 0])
downsample = 50
feature_matrix = np.array([
    intensity[::downsample],
    jerkiness[::downsample],
    kinetic_energy[::downsample],
    fluidity[::downsample]
])
im = ax8.imshow(feature_matrix, aspect='auto', cmap='viridis', interpolation='bilinear')
ax8.set_yticks([0, 1, 2, 3])
ax8.set_yticklabels(['Intensity', 'Jerkiness', 'Kinetic Energy', 'Fluidity'])
ax8.set_xlabel('Time (samples, downsampled)')
ax8.set_title('Feature Timeline (Downsampled)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax8, label='Normalized Value')

# Plot 9: Intensity vs Jerkiness scatter
ax9 = fig.add_subplot(gs[5, 1])
scatter = ax9.scatter(intensity, jerkiness, c=fluidity, cmap='RdYlGn', s=5, alpha=0.5)
ax9.set_xlabel('Intensity')
ax9.set_ylabel('Jerkiness')
ax9.set_title('Intensity vs Jerkiness (colored by Fluidity)', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax9, label='Fluidity')

fig.suptitle('Dog Motion Analysis - Integrated Movement Features (Intensity, Jerkiness, KE, Fluidity)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('dog_movement_features.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'dog_movement_features.png'")
plt.show()

# ==================== EXPORT RESULTS ====================
output_df = df[['Time', 'X', 'Y', 'Z', 'Acceleration_Magnitude', 
                 'Intensity', 'Jerkiness', 'Kinetic_Energy', 'Fluidity']].copy()
output_df.to_csv('dog_movement_features.csv', index=False)
print("✓ Data exported to 'dog_movement_features.csv'")

# Export feature summary
summary = summarize_features({
    'Intensity': intensity,
    'Jerkiness': jerkiness,
    'Kinetic_Energy': kinetic_energy,
    'Fluidity': fluidity
})

print("\n" + "="*60)
print("FEATURE SUMMARY STATISTICS")
print("="*60)
for key, value in summary.items():
    print(f"{key:30s}: {value:10.6f}")
