import numpy as np

def cosine(duration, frequency, sampling_frequency, amplitude=1.0, phase=0.0):
    """
    Generate a cosine signal.

    Parameters:
        duration (float): Signal duration in seconds
        frequency (float): Frequency of the cosine in Hz
        sampling_frequency (float): Sampling frequency in Hz
        amplitude (float): Amplitude of the cosine (default=1.0)
        phase (float): Phase shift in radians (default=0.0)

    Returns:
        t (numpy.ndarray): Time vector
        x (numpy.ndarray): Cosine signal
    """
    t = np.arange(0, duration, 1 / sampling_frequency)
    x = amplitude * np.cos(2 * np.pi * frequency * t + phase)
    return t, x

def cosine_with_arctan_freq(duration, 
                            sampling_frequency,
                            f0,
                            atan_amp,
                            atan_rate
                           ):
    
    """
    Returns time, frequ_function, signal
    """

    n = duration*sampling_frequency
    t = np.arange(n)*(1/sampling_frequency)

    freq_func = f0 + atan_amp * np.arctan(atan_rate*(t - t.max()//2)/2)
    s = np.cos(2*np.pi*np.cumsum(freq_func)*(1/sampling_frequency))
    return t, freq_func, s

def sum_of_cosines(duration,
                   frequency1,
                   frequency2,
                   sampling_frequency,
                   amplitude1=1.0,
                   amplitude2=1.0,
                   phase1=0.0,
                   phase2=0.0):
    """
    Sum of two cosines.

    Parameters:
        duration (float): Signal duration in seconds
        frequency1 (float): First frequency in Hz
        frequency2 (float): Second frequency in Hz
        sampling_frequency (float): Sampling frequency in Hz
        amplitude1 (float): Amplitude of first cosine (default=1.0)
        amplitude2 (float): Amplitude of second cosine (default=1.0)
        phase1 (float): Phase of first cosine in radians (default=0.0)
        phase2 (float): Phase of second cosine in radians (default=0.0)

    Returns:
        t (numpy.ndarray): Time vector
        x (numpy.ndarray): sigmal vector
    """
    t = np.arange(0, duration, 1 / sampling_frequency)

    x1 = amplitude1 * np.cos(2 * np.pi * frequency1 * t + phase1)
    x2 = amplitude2 * np.cos(2 * np.pi * frequency2 * t + phase2)
    
    return t, x1 + x2