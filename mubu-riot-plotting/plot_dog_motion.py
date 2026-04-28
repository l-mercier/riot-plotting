#!/usr/bin/env python3
"""
Script to read and plot dog motion data from dog.json using seaborn
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_theme(style="darkgrid")

# Load the JSON file
json_path = "/Users/mercier/Documents/Projects for after/AnimalMotion/recordings/dog.json"
print(f"Loading {json_path}...")

with open(json_path, 'r') as f:
    data = json.load(f)

# Extract the sig1 track (motion data with 3 columns)
sig1_track = None
for track in data['tracks']:
    if track['name'] == 'sig1':
        sig1_track = track
        break

if sig1_track is None:
    print("sig1 track not found!")
    exit(1)

# Extract the motion data
mxData = sig1_track['buffers'][0]['mxData']
sample_rate = sig1_track['buffers'][0]['sampleRate']
num_cols = sig1_track['mxCols']

print(f"Sample rate: {sample_rate} Hz")
print(f"Number of columns: {num_cols}")
print(f"Total data points: {len(mxData)} values")
print(f"Samples: {len(mxData) // num_cols}")

# Reshape data into a 2D array
num_samples = len(mxData) // num_cols
reshaped_data = [mxData[i:i+num_cols] for i in range(0, len(mxData), num_cols)]

# Create a DataFrame
df = pd.DataFrame(reshaped_data, columns=['X', 'Y', 'Z'])
df['Time'] = df.index / sample_rate  # Convert to time in seconds

print(f"\nDataFrame shape: {df.shape}")
print(f"Duration: {df['Time'].iloc[-1]:.2f} seconds")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dog Motion Data Analysis', fontsize=16)

# Plot 1: Time series of X, Y, Z
ax = axes[0, 0]
ax.plot(df['Time'], df['X'], label='X', alpha=0.7)
ax.plot(df['Time'], df['Y'], label='Y', alpha=0.7)
ax.plot(df['Time'], df['Z'], label='Z', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position')
ax.set_title('Motion Trajectories Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: 3D scatter plot (projected as 2D)
ax = axes[0, 1]
scatter = ax.scatter(df['X'], df['Y'], c=df['Time'], cmap='viridis', alpha=0.6, s=2)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('XY Trajectory')
plt.colorbar(scatter, ax=ax, label='Time (s)')

# Plot 3: XZ projection
ax = axes[1, 0]
scatter = ax.scatter(df['X'], df['Z'], c=df['Time'], cmap='plasma', alpha=0.6, s=2)
ax.set_xlabel('X Position')
ax.set_ylabel('Z Position')
ax.set_title('XZ Trajectory')
plt.colorbar(scatter, ax=ax, label='Time (s)')

# Plot 4: Distribution comparison using seaborn
ax = axes[1, 1]
# Melt the dataframe for seaborn
df_melted = df[['X', 'Y', 'Z']].melt(var_name='Axis', value_name='Position')
sns.violinplot(data=df_melted, x='Axis', y='Position', ax=ax)
ax.set_title('Distribution of Positions by Axis')
ax.set_ylabel('Position Value')

plt.tight_layout()
plt.savefig('dog_motion_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'dog_motion_analysis.png'")
plt.show()

# Additional statistics
print("\n=== Data Statistics ===")
print(df[['X', 'Y', 'Z']].describe())
