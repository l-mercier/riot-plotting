#!/usr/bin/env python3
"""
Python translation of PiPoIntensity.h
Computes intensity over time from motion data using low-pass filtering and various normalization modes.
"""

import numpy as np
import pandas as pd
from enum import Enum


class IntensityMode(Enum):
    """Value transformation modes"""
    ABS = 0           # Absolute value
    POS = 1           # Positive part only
    NEG = 2           # Negative part only (inverted)
    SQUARE = 3        # Square of value


class NormMode(Enum):
    """Normalization modes"""
    L2_PRE = 0        # Pre normalization: L2 norm computed before processing
    L2_POST = 1       # Post normalization: L2 norm computed after processing
    MEAN_PRE = 2      # Pre normalization: mean computed before processing
    MEAN_POST = 3     # Post normalization: mean computed after processing


# Default parameters
SAMPLING_RATE_REF = 100.0
DEFAULT_CUT_FREQUENCY = 10.0
DEFAULT_FEEDBACK = 0.9
DEFAULT_GAIN = 1.0
GAIN_ADJUSTMENT = 0.01
DEFAULT_DELTA_SIZE = 3
DEFAULT_MOVING_AVERAGE_SIZE = 1


class Intensity:
    """
    Computes intensity metrics from motion data.
    Applies: moving average → delta (derivative) → intensity processing
    """
    
    def __init__(self,
                 sample_rate=SAMPLING_RATE_REF,
                 gain=DEFAULT_GAIN,
                 cut_frequency=DEFAULT_CUT_FREQUENCY,
                 mode=IntensityMode.ABS,
                 norm_mode=NormMode.L2_PRE,
                 offset=False,
                 offset_value=0.0,
                 clip_max=False,
                 clip_max_value=1.0,
                 power_exp=1.0,
                 delta_size=DEFAULT_DELTA_SIZE,
                 moving_average_size=DEFAULT_MOVING_AVERAGE_SIZE):
        
        self.sample_rate = sample_rate
        self.gain = gain
        self.cut_frequency = cut_frequency
        self.mode = mode
        self.norm_mode = norm_mode
        self.offset = offset
        self.offset_value = offset_value
        self.clip_max = clip_max
        self.clip_max_value = clip_max_value
        self.power_exp = power_exp
        self.delta_size = delta_size
        self.moving_average_size = moving_average_size
        
        self.memory_vector = None  # Low-pass filter memory
        self.feedback = DEFAULT_FEEDBACK
    
    def set_stream_attributes(self, sample_rate, num_channels):
        """Initialize processing for given sample rate and number of channels"""
        self.sample_rate = sample_rate
        
        # Calculate feedback coefficient for low-pass filter
        normed_cut_frequency = self.cut_frequency / sample_rate
        self.feedback = 1.0 - (normed_cut_frequency / (normed_cut_frequency + 1.0))
        
        # Initialize memory vector for low-pass filtering
        self.memory_vector = np.zeros(num_channels)
    
    def get_value_by_mode(self, val):
        """Apply value transformation based on mode"""
        if self.mode == IntensityMode.SQUARE:
            return val * val
        elif self.mode == IntensityMode.ABS:
            return abs(val)
        elif self.mode == IntensityMode.POS:
            return max(val, 0.0)
        elif self.mode == IntensityMode.NEG:
            return -min(val, 0.0)
        return abs(val)  # default to abs
    
    def process_frame(self, values):
        """
        Process a single frame of values.
        
        Args:
            values: array of shape (num_channels,)
            
        Returns:
            Processed intensity value(s)
        """
        num_channels = len(values)
        output = np.zeros(num_channels)
        
        # Ensure memory vector is initialized
        if self.memory_vector is None:
            self.set_stream_attributes(self.sample_rate, num_channels)
        
        norm = 0.0
        
        # Process each channel
        for i in range(num_channels):
            # Apply gain adjustment and value mode transformation
            delta_value = values[i] * GAIN_ADJUSTMENT
            processed_value = self.get_value_by_mode(delta_value)
            
            # Low-pass filter (1st order IIR)
            filtered_value = (processed_value * (1.0 - self.feedback) + 
                             self.feedback * self.memory_vector[i])
            
            # Store for next iteration
            self.memory_vector[i] = filtered_value
            
            # Apply overall gain
            filtered_value *= self.gain
            
            # Compute normalization if needed
            if self.norm_mode in (NormMode.L2_POST, NormMode.MEAN_POST):
                norm += filtered_value * filtered_value if self.norm_mode == NormMode.L2_POST else filtered_value
            
            # Apply power exponent
            filtered_value = np.power(filtered_value, self.power_exp)
            
            # Apply offset
            if self.offset:
                filtered_value -= self.offset_value
                filtered_value = max(filtered_value, 0.0)
            
            # Clip maximum
            if self.clip_max:
                filtered_value = min(filtered_value, self.clip_max_value)
            
            output[i] = filtered_value
        
        # Apply post-normalization if needed
        if self.norm_mode == NormMode.L2_POST:
            norm = np.sqrt(norm)
            norm = np.power(norm, self.power_exp)
            if self.offset:
                norm -= self.offset_value
                norm = max(norm, 0.0)
            if self.clip_max:
                norm = min(norm, self.clip_max_value)
            output[0] = norm
        elif self.norm_mode == NormMode.MEAN_POST:
            norm /= num_channels
            norm = np.power(norm, self.power_exp)
            if self.offset:
                norm -= self.offset_value
                norm = max(norm, 0.0)
            if self.clip_max:
                norm = min(norm, self.clip_max_value)
            output[0] = norm
        
        return output
    
    def process(self, data):
        """
        Process array of motion data.
        
        Args:
            data: DataFrame or numpy array with shape (num_samples, num_channels)
            
        Returns:
            numpy array of intensity values
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        num_samples = data.shape[0]
        num_channels = data.shape[1] if len(data.shape) > 1 else 1
        
        self.set_stream_attributes(self.sample_rate, num_channels)
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        output = np.zeros((num_samples, num_channels))
        
        for i in range(num_samples):
            output[i] = self.process_frame(data[i])
        
        return output


def compute_derivatives(data, delta_size=DEFAULT_DELTA_SIZE, sample_rate=SAMPLING_RATE_REF):
    """
    Compute numerical derivatives (deltas) of the data.
    Uses centered difference approximation.
    
    Args:
        data: input array of shape (num_samples, num_channels)
        delta_size: window size for derivative (must be odd)
        sample_rate: sampling rate in Hz
        
    Returns:
        derivatives: array of same shape as input
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # Ensure delta_size is odd
    if delta_size % 2 == 0:
        delta_size += 1
    
    half_size = delta_size // 2
    derivatives = np.zeros_like(data, dtype=float)
    dt = 1.0 / sample_rate
    
    # Use centered difference for smoother derivative
    for i in range(data.shape[1]):
        for j in range(len(data)):
            start = max(0, j - half_size)
            end = min(len(data), j + half_size + 1)
            window = data[start:end, i]
            
            # Use simple centered difference or forward/backward at edges
            if j - half_size >= 0 and j + half_size < len(data):
                # Centered difference
                derivatives[j, i] = (data[j + half_size, i] - data[j - half_size, i]) / (2 * half_size * dt)
            elif j + 1 < len(data):
                # Forward difference
                derivatives[j, i] = (data[j + 1, i] - data[j, i]) / dt
            elif j > 0:
                # Backward difference
                derivatives[j, i] = (data[j, i] - data[j - 1, i]) / dt
    
    return derivatives


def apply_moving_average(data, window_size=DEFAULT_MOVING_AVERAGE_SIZE):
    """
    Apply moving average filter.
    
    Args:
        data: input array
        window_size: size of moving average window
        
    Returns:
        smoothed array
    """
    if window_size <= 1:
        return data
    
    if len(data.shape) == 1:
        smoothed = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean().values
        return np.nan_to_num(smoothed, nan=0.0)
    else:
        result = data.copy()
        for i in range(data.shape[1]):
            smoothed = pd.Series(data[:, i]).rolling(window=window_size, center=True, min_periods=1).mean().values
            result[:, i] = np.nan_to_num(smoothed, nan=0.0)
        return result


def process_intensity_pipeline(data, sample_rate=SAMPLING_RATE_REF,
                               moving_average_size=DEFAULT_MOVING_AVERAGE_SIZE,
                               delta_size=DEFAULT_DELTA_SIZE,
                               **intensity_kwargs):
    """
    Full intensity processing pipeline: moving average → derivatives → intensity
    
    Args:
        data: input motion data (num_samples, 3) for X, Y, Z
        sample_rate: sampling rate in Hz
        moving_average_size: size of moving average filter
        delta_size: delta window size for derivatives
        **intensity_kwargs: additional arguments for Intensity class
        
    Returns:
        intensity: computed intensity values
    """
    # Step 1: Apply moving average filter
    smoothed = apply_moving_average(data, moving_average_size)
    
    # Step 2: Compute derivatives
    derivatives = compute_derivatives(smoothed, delta_size, sample_rate)
    
    # Step 3: Compute intensity
    intensity_processor = Intensity(sample_rate=sample_rate, **intensity_kwargs)
    
    # Compute pre-normalization if needed
    norm_mode = intensity_kwargs.get('norm_mode', NormMode.L2_PRE)
    
    if norm_mode == NormMode.L2_PRE:
        # L2 norm of derivatives before processing
        intensity_vals = np.linalg.norm(derivatives, axis=1)
    elif norm_mode == NormMode.MEAN_PRE:
        # Mean of derivatives before processing
        intensity_vals = np.mean(derivatives, axis=1)
    else:
        # Process each channel and combine in post-processing
        intensity_vals = intensity_processor.process(derivatives)
        if intensity_vals.shape[1] > 1:
            intensity_vals = intensity_vals[:, 0]  # Take first column which contains normalized value
        else:
            intensity_vals = intensity_vals[:, 0]
    
    return intensity_vals, derivatives, smoothed
