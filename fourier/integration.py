import numpy as np


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