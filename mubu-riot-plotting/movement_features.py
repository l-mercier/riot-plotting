#!/usr/bin/env python3
"""
Python translation of movement features:
- Intensity: acceleration-based motion intensity with low-pass filtering
- Jerkiness: squared jerk (third derivative) normalized over a buffer
- Kinetic Energy: integrated kinetic energy from acceleration
- Fluidity Index: ratio-based fluidity measurement

Based on motion capture analysis framework describing hand movement quality.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


# Intensity computation constants
SAMPLING_RATE_REF = 100.0
DEFAULT_CUT_FREQUENCY = 10.0
DEFAULT_FEEDBACK = 0.9
DEFAULT_GAIN = 1.0
GAIN_ADJUSTMENT = 0.01
DEFAULT_DELTA_SIZE = 3
DEFAULT_MOVING_AVERAGE_SIZE = 1


class MovementFeatures:
    """
    Computes movement quality features from linear acceleration data:
    - Intensity: acceleration-based motion intensity
    - Jerkiness: squared jerk (third derivative)
    - Kinetic Energy: integrated from acceleration
    - Fluidity Index: movement smoothness ratio
    """
    
    # Default buffer size for normalization
    DEFAULT_BUFFER_SIZE = 20
    
    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE,
                 sample_rate: float = SAMPLING_RATE_REF,
                 cut_frequency: float = DEFAULT_CUT_FREQUENCY,
                 gain: float = DEFAULT_GAIN):
        """
        Args:
            buffer_size: Window size for normalized jerkiness computation (default: 20)
            sample_rate: Sampling rate in Hz
            cut_frequency: Cut frequency for low-pass filter (Hz)
            gain: Overall gain for intensity computation
        """
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.cut_frequency = cut_frequency
        self.gain = gain
        
        self.max_jerkiness = 1.0
        self.max_kinetic_energy = 1.0
        self.max_intensity = 1.0
        
        # Low-pass filter state
        self.memory_vector = None
        self.feedback = DEFAULT_FEEDBACK
        self._update_feedback()
    
    def _update_feedback(self):
        """Update feedback coefficient based on cut frequency and sample rate"""
        normed_cut_frequency = self.cut_frequency / self.sample_rate
        self.feedback = 1.0 - (normed_cut_frequency / (normed_cut_frequency + 1.0))
    
    def _get_value_by_mode(self, val, mode='abs'):
        """Apply value transformation based on mode
        
        Args:
            val: input value
            mode: 'abs' (absolute), 'square', 'pos' (positive part), 'neg' (negative part)
        """
        if mode == 'square':
            return val * val
        elif mode == 'pos':
            return max(val, 0.0)
        elif mode == 'neg':
            return -min(val, 0.0)
        else:  # 'abs'
            return abs(val)
    
    def compute_intensity(self, acceleration: np.ndarray,
                         power_exp: float = 1.0,
                         offset: bool = False,
                         offset_value: float = 0.0,
                         clip_max: bool = False,
                         clip_max_value: float = 1.0,
                         mode: str = 'abs') -> np.ndarray:
        """
        Compute intensity from acceleration with low-pass filtering.
        
        Args:
            acceleration: array of shape (num_samples, 3) or (num_samples,)
            power_exp: power exponent for values
            offset: whether to remove offset
            offset_value: offset value to subtract
            clip_max: whether to clip at maximum
            clip_max_value: maximum clip value
            mode: value transformation mode ('abs', 'square', 'pos', 'neg')
            
        Returns:
            intensity: array of intensity values
        """
        if len(acceleration.shape) == 1:
            acceleration = acceleration.reshape(-1, 1)
        
        num_samples = acceleration.shape[0]
        num_channels = acceleration.shape[1]
        
        # Initialize memory for low-pass filter
        if self.memory_vector is None or len(self.memory_vector) != num_channels:
            self.memory_vector = np.zeros(num_channels)
        
        output = np.zeros(num_samples)
        
        for j in range(num_samples):
            for i in range(num_channels):
                # Apply gain adjustment
                delta_value = acceleration[j, i] * GAIN_ADJUSTMENT
                
                # Apply value mode transformation
                processed = self._get_value_by_mode(delta_value, mode)
                
                # Low-pass filter (1st order IIR)
                filtered = (processed * (1.0 - self.feedback) + 
                           self.feedback * self.memory_vector[i])
                
                # Store for next iteration
                self.memory_vector[i] = filtered
                
                # Apply overall gain
                filtered *= self.gain
                
                # Apply power exponent
                filtered = np.power(filtered, power_exp)
                
                # Apply offset
                if offset:
                    filtered -= offset_value
                    filtered = max(filtered, 0.0)
                
                # Clip maximum
                if clip_max:
                    filtered = min(filtered, clip_max_value)
                
                output[j] += filtered
        
        # Normalize
        self.max_intensity = np.max(output) if np.max(output) > 0 else 1.0
        output = output / self.max_intensity
        
        return output
    
    @staticmethod
    def compute_jerkiness(acceleration: np.ndarray) -> np.ndarray:
        """
        Compute jerkiness (squared jerk) from acceleration components.
        
        Jerkiness is the squared third derivative (jerk):
        J^N = (a_x_dot)^2 + (a_y_dot)^2 + (a_z_dot)^2
        
        Args:
            acceleration: array of shape (num_samples, 3) containing [a_x, a_y, a_z]
                         (linear acceleration components)
        
        Returns:
            jerkiness: array of shape (num_samples,) with squared jerk values
        """
        if len(acceleration.shape) == 1:
            acceleration = acceleration.reshape(-1, 1)
        
        num_samples = acceleration.shape[0]
        jerkiness = np.zeros(num_samples)
        
        # Compute jerk (derivative of acceleration) using centered differences
        for i in range(num_samples):
            jerk_components = np.zeros(3)
            
            for j in range(min(3, acceleration.shape[1])):
                # Centered difference where possible
                if i > 0 and i < num_samples - 1:
                    jerk_components[j] = acceleration[i + 1, j] - acceleration[i - 1, j]
                elif i < num_samples - 1:
                    jerk_components[j] = acceleration[i + 1, j] - acceleration[i, j]
                elif i > 0:
                    jerk_components[j] = acceleration[i, j] - acceleration[i - 1, j]
            
            # Compute squared jerk
            jerkiness[i] = np.sum(jerk_components ** 2)
        
        return jerkiness
    
    def normalize_jerkiness(self, jerkiness: np.ndarray) -> np.ndarray:
        """
        Normalize jerkiness over a buffer of values.
        
        J_tot^N = sum(J_i for i in buffer) / Max(J^N)
        
        Args:
            jerkiness: array of squared jerk values
            
        Returns:
            normalized_jerkiness: array of normalized values
        """
        num_samples = len(jerkiness)
        normalized = np.zeros(num_samples)
        
        # Store global max for normalization
        self.max_jerkiness = np.max(jerkiness) if np.max(jerkiness) > 0 else 1.0
        
        # Compute rolling sum over buffer size
        for i in range(num_samples):
            start_idx = max(0, i - self.buffer_size + 1)
            buffer = jerkiness[start_idx:i + 1]
            normalized[i] = np.sum(buffer) / self.max_jerkiness
        
        return normalized
    
    @staticmethod
    def compute_velocity_from_acceleration(acceleration: np.ndarray, 
                                          sample_rate: float = 100.0,
                                          initial_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Integrate acceleration to obtain velocity.
        
        v(t) = integral of a(t) dt
        
        Args:
            acceleration: array of shape (num_samples, 3)
            sample_rate: sampling rate in Hz
            initial_velocity: initial velocity vector [v_x, v_y, v_z] (default: zeros)
        
        Returns:
            velocity: integrated velocity array
        """
        if len(acceleration.shape) == 1:
            acceleration = acceleration.reshape(-1, 1)
        
        dt = 1.0 / sample_rate
        num_samples = acceleration.shape[0]
        num_channels = acceleration.shape[1]
        
        if initial_velocity is None:
            initial_velocity = np.zeros(num_channels)
        
        # Cumulative trapezoidal integration
        velocity = np.zeros_like(acceleration)
        velocity[0] = initial_velocity
        
        for i in range(1, num_samples):
            # Trapezoidal rule
            velocity[i] = velocity[i - 1] + (acceleration[i] + acceleration[i - 1]) / 2.0 * dt
        
        return velocity
    
    def compute_kinetic_energy(self, acceleration: np.ndarray, 
                              sample_rate: float = 100.0,
                              mass: float = 1.0) -> np.ndarray:
        """
        Compute kinetic energy from acceleration components.
        
        Velocity components are obtained by integrating acceleration:
        E^N = 0.5 * m * [(integral(a_x))^2 + (integral(a_y))^2 + (integral(a_z))^2]
        
        Then normalize: E_tot^N = E^N / Max(E^N)
        
        Args:
            acceleration: array of shape (num_samples, 3) with acceleration components
            sample_rate: sampling rate in Hz
            mass: mass approximation (default: 1.0 for hands)
            
        Returns:
            kinetic_energy: normalized kinetic energy values
        """
        if len(acceleration.shape) == 1:
            acceleration = acceleration.reshape(-1, 1)
        
        # Integrate acceleration to get velocity
        velocity = self.compute_velocity_from_acceleration(acceleration, sample_rate)
        
        # Compute kinetic energy: KE = 0.5 * m * v^2
        speed_squared = np.sum(velocity ** 2, axis=1)
        kinetic_energy = 0.5 * mass * speed_squared
        
        # Normalize by max value
        self.max_kinetic_energy = np.max(kinetic_energy) if np.max(kinetic_energy) > 0 else 1.0
        normalized_ke = kinetic_energy / self.max_kinetic_energy
        
        return normalized_ke
    
    def compute_fluidity_index(self, normalized_jerkiness: np.ndarray,
                              normalized_kinetic_energy: np.ndarray) -> np.ndarray:
        """
        Compute fluidity index from jerkiness and kinetic energy.
        
        F_tot^N = 1 / (J_tot^N / E_tot^N)
        
        This measures smooth vs jerky movement:
        - High fluidity: high kinetic energy with low jerkiness (smooth)
        - Low fluidity: low kinetic energy or high jerkiness (jerky)
        
        Args:
            normalized_jerkiness: array of normalized jerkiness values
            normalized_kinetic_energy: array of normalized kinetic energy values
            
        Returns:
            fluidity: fluidity index for each time point
        """
        num_samples = len(normalized_jerkiness)
        fluidity = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Avoid division by zero
            jerk_energy_ratio = normalized_jerkiness[i] / (normalized_kinetic_energy[i] + 1e-10)
            fluidity[i] = 1.0 / (jerk_energy_ratio + 1e-10)
        
        return fluidity
    
    def process_bilateral_features(self, acceleration_right: np.ndarray,
                                  acceleration_left: np.ndarray,
                                  sample_rate: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Process movement features for bilateral data (e.g., both hands).
        
        Computes intensity, jerkiness, kinetic energy, and fluidity for both sides,
        then returns individual and combined (mean) values.
        
        Args:
            acceleration_right: right hand acceleration data (num_samples, 3)
            acceleration_left: left hand acceleration data (num_samples, 3)
            sample_rate: sampling rate in Hz
            
        Returns:
            Dictionary containing:
                - 'intensity_right': normalized intensity for right hand
                - 'intensity_left': normalized intensity for left hand
                - 'jerkiness_right': normalized jerkiness for right hand
                - 'jerkiness_left': normalized jerkiness for left hand
                - 'kinetic_energy_right': normalized KE for right hand
                - 'kinetic_energy_left': normalized KE for left hand
                - 'fluidity_right': fluidity for right hand
                - 'fluidity_left': fluidity for left hand
                - 'intensity_combined': mean intensity
                - 'fluidity_index': mean fluidity (FI)
        """
        # Process right hand
        intensity_right = self.compute_intensity(acceleration_right)
        jerk_right = self.compute_jerkiness(acceleration_right)
        jerk_norm_right = self.normalize_jerkiness(jerk_right)
        ke_right = self.compute_kinetic_energy(acceleration_right, sample_rate)
        fluidity_right = self.compute_fluidity_index(jerk_norm_right, ke_right)
        
        # Process left hand
        intensity_left = self.compute_intensity(acceleration_left)
        jerk_left = self.compute_jerkiness(acceleration_left)
        jerk_norm_left = self.normalize_jerkiness(jerk_left)
        ke_left = self.compute_kinetic_energy(acceleration_left, sample_rate)
        fluidity_left = self.compute_fluidity_index(jerk_norm_left, ke_left)
        
        # Compute combined metrics
        intensity_combined = (intensity_right + intensity_left) / 2.0
        fluidity_index = (fluidity_right + fluidity_left) / 2.0
        
        return {
            'intensity_right': intensity_right,
            'intensity_left': intensity_left,
            'intensity_combined': intensity_combined,
            'jerkiness_right': jerk_norm_right,
            'jerkiness_left': jerk_norm_left,
            'kinetic_energy_right': ke_right,
            'kinetic_energy_left': ke_left,
            'fluidity_right': fluidity_right,
            'fluidity_left': fluidity_left,
            'fluidity_index': fluidity_index
        }
    
    def process_unilateral_features(self, acceleration: np.ndarray,
                                   sample_rate: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Process movement features for single hand/limb.
        
        Args:
            acceleration: acceleration data (num_samples, 3)
            sample_rate: sampling rate in Hz
            
        Returns:
            Dictionary containing:
                - 'intensity': normalized intensity
                - 'jerkiness': normalized jerkiness
                - 'kinetic_energy': normalized kinetic energy
                - 'fluidity': fluidity index
        """
        # Compute intensity
        intensity = self.compute_intensity(acceleration)
        
        # Compute jerkiness
        jerk = self.compute_jerkiness(acceleration)
        jerk_norm = self.normalize_jerkiness(jerk)
        
        # Compute kinetic energy
        ke = self.compute_kinetic_energy(acceleration, sample_rate)
        
        # Compute fluidity
        fluidity = self.compute_fluidity_index(jerk_norm, ke)
        
        return {
            'intensity': intensity,
            'jerkiness': jerk_norm,
            'kinetic_energy': ke,
            'fluidity': fluidity
        }


def process_movement_features_from_velocity(velocity: np.ndarray,
                                           sample_rate: float = 100.0,
                                           buffer_size: int = 20) -> Dict[str, np.ndarray]:
    """
    Compute movement features directly from velocity data (if acceleration not available).
    
    Estimates jerkiness from numerical differentiation of velocity.
    
    Args:
        velocity: velocity data (num_samples, 3)
        sample_rate: sampling rate in Hz
        buffer_size: buffer size for jerkiness normalization
        
    Returns:
        Dictionary with computed features
    """
    # Approximate acceleration by differentiating velocity
    dt = 1.0 / sample_rate
    acceleration = np.zeros_like(velocity)
    
    for i in range(1, len(velocity)):
        acceleration[i] = (velocity[i] - velocity[i - 1]) / dt
    
    processor = MovementFeatures(buffer_size=buffer_size)
    return processor.process_unilateral_features(acceleration, sample_rate)


def summarize_features(features_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute summary statistics for movement features.
    
    Args:
        features_dict: dictionary of feature arrays
        
    Returns:
        Dictionary with mean, std, min, max for each feature
    """
    summary = {}
    
    for feature_name, feature_values in features_dict.items():
        summary[f'{feature_name}_mean'] = float(np.mean(feature_values))
        summary[f'{feature_name}_std'] = float(np.std(feature_values))
        summary[f'{feature_name}_min'] = float(np.min(feature_values))
        summary[f'{feature_name}_max'] = float(np.max(feature_values))
    
    return summary
