import numpy as np

def sum_power(res, num_bins_to_sum):
    """ Total power contained in indicated bins of a periodogram.
    
    Parameters
    ----------
    
        res : dict
            output of `symmetric_peak_periodogram` command.
            Contains PSD values, frequency values, peak index, etc.
            
        num_bins_to_sum : tupple of two integers
            The first element is the number of the bins on the left (towards
            lower frequencies) of the peak. The second element is the 
            number of bins on the right side of the peak. PSD values in all
            these bins, including the peak bin, are summed after subtraction
            of the baseline, also contained in `res`. 
            
    Returns
    -------
    
        power : float
            Summed PSD values multiplied by the frequency step.
    
    """
    idx_0 = res['peak_index'] - num_bins_to_sum[0]
    idx_1 = res['peak_index'] + num_bins_to_sum[1] + 1
    freq_step = res['frequencies'][1]
    return (res['power'][idx_0 : idx_1].sum() - res['noise_level'])*freq_step

def integrate_psd(Pxx, f, f_center, half_width):
    """
    Integrate PSD within a frequency window centered at a given index.

    Parameters
    ----------
    Pxx : array-like
        1D power spectral density
    f : array-like
        Frequency vector (same length as Pxx)
    f_center : float
        Frequency in the center of the integration window
    half_width : float
        Half-width of the integration window (in frequency units)

    Returns
    -------
    - integrated power
    - number of samples inside the freuqency band
    """

    Pxx = np.asarray(Pxx)
    f = np.asarray(f)

    if len(Pxx) != len(f):
        raise ValueError("Pxx and f must have the same length")

    if not (min(f) <= f_center < max(f)):
        raise ValueError("f_center out of bounds")

    # Define frequency window
    f_min = f_center - half_width
    f_max = f_center + half_width

    # Mask for window
    mask = (f >= f_min) & (f <= f_max)

    if not np.any(mask):
        print('No data inside integration window.')
        return 0.0

    # Integrate using trapezoidal rule
    integrated_power = np.trapz(Pxx[mask], f[mask])

    return integrated_power, len(Pxx[mask])