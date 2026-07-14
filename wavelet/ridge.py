import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def interpolate(y, x, x_new):
    """ Linearly interpolates between
    data points. Extrapolates outside
    x boundaries with constant values:
    y[0] for x_new<x, y[-1] for x_new>x.
    """
    f = interp1d(x, y, fill_value=tuple(y[[0, -1]]),
                 bounds_error=False) # doesn't complain that new x is wider than original x
    return f(x_new)

def split_scalogram(p, num_segments):
    window_size = p.t.size//num_segments
    return p.coarsen(t=window_size, boundary='trim').mean('t')

def peak_freq(x, f, **find_peaks_kwargs):
    peaks, _ = find_peaks(x, **find_peaks_kwargs)

    if len(peaks) == 0:
        return np.nan

    idx = peaks[np.argmax(x[peaks])]
    return f[idx]

def integrate_peak(profile, band):
    return profile.sel(f=slice(*band)).integrate('f')

def interpolate_peak(x, f, f0, n_side=2):
    
    """
    Refine peak location estimate by quadratically interpolating
    data around the peak with maximum at f0.
    
    Parameters
    ----------
    x : array_like
        Spectrum.
    f : array_like
        Frequency vector (need not be uniformly spaced).
    f0 : float
    n_side : int, default 2
        Number of samples on both sides of f0 to use for interpolation.
        Default is 2, so interpolation is over 5 samples.

    Returns
    -------
    float
    """
    
    if not np.isfinite(f0):
        return np.nan

    idx = np.argmin(np.abs(f - f0))
    
    # Not enough points for fitting
    if idx < n_side or idx >= len(f) - n_side:
        print(f'Not enough data aroudn peak for interpolation. N_side = {n_side}')
        return f0

    # Five-point quadratic fit
    ff = f[idx - n_side : idx + n_side + 1]
    xx = x[idx - n_side : idx + n_side + 1]
    
    if np.any(np.isnan(xx)):
        ff = ff[~np.isnan(xx)] # Only for profiles close to COI. If f0 is low, ff may catch NaNs if COI is masked out of scalogram
        xx = xx[~np.isnan(xx)]

    a, b, c = np.polyfit(ff, xx, deg=2)

    # Degenerate fit or parabola opening upward
    if np.isclose(a, 0) or a >= 0:
        print('Degenrate fit or parabola opening upward. Return non-interpolated frequency.')
        return f0
    
    f_peak = -b / (2*a)
    
    # Reject unreasonable extrapolation
    if not (ff[0] <= f_peak <= ff[-1]):
        print("Interpolated peak is outside interpolation interval. Return f0")
        return f0

    return f_peak

def track_band(
    profiles,
    f0,
    integrate_func,
    noise_floor,
    frac_search=0.5,
    frac=0.3,
    threshold_k=3,

):
    """
    Track an oscillation band through a sequence of spectra.

    Parameters
    ----------
    profiles : xr.DataArray
        Dimensions (t, f). Iterates over the first dimension.
    f0 : float
        Initial center frequency.
    frac_search : float, default 0.5
        Relative half-bandwidth (e.g. 0.1 -> +/-10%). Used to search peaks.
    frac : float, default 0.3
        Relative half-bandwidth (e.g. 0.1 -> ±10%). Used to calculate power
        and estimate peak significance.
    integrate_func : callable
        integrate_func(profile, band) -> (power, noise)
    noise_floor : float
        Global mean of the noise power. From it, the integrated noise
        power is computed by `noise_floor * band_width`.
    threshold_k : float, default 3
        Minimum power/noise ratio required to update the tracked frequency.
        Integrated peak power is divided with the integrated noise power.

    Returns
    -------
    xr.Dataset
    """

    t = profiles.t.values

    freq = []
    fmin = []
    fmax = []
    power = []
    noise = []
    snr = []

    fc = f0 # current frequency guess
    
    def _process_profile(profile, band):
        pwr = integrate_func(profile, band)
        bandwidth = band[1] - band[0]
        noise_power = noise_floor * bandwidth
        pwr_norm = pwr / noise_power
        return pwr, noise_power, pwr_norm

    for profile in profiles:

        search_band = (fc * (1 - frac_search),
                       fc * (1 + frac_search))
        x = profile.sel(f=slice(*search_band))

        fp = peak_freq(x.values, x.f.values)
        fp = interpolate_peak(profile.values, profile.f.values, fp)

        if np.isfinite(fp): # A peak is found in the spectrum
            candidate_band = (fp * (1 - frac),
                              fp * (1 + frac))
            pwr, noise_power, pwr_norm = _process_profile(profile, candidate_band)

            # If the peak power is significant, update fc, and band. Else, keep fc, and band unchanged
            if pwr_norm > threshold_k:
                fc = fp 
                band = candidate_band
            else:
                band = (fc * (1 - frac),
                        fc * (1 + frac))
                pwr, noise_power, pwr_norm = _process_profile(profile, band)

        else: # No peak in spectrum: use frequency from the previous spectrum
            band = (fc * (1 - frac),
                    fc * (1 + frac))
            pwr, noise_power, pwr_norm = _process_profile(profile, band)

        freq.append(fc)
        fmin.append(band[0])
        fmax.append(band[1])
        power.append(pwr)
        noise.append(noise_power) # Mean noise is fixed for each profile. But the integrated noise power depends on the band width
        snr.append(pwr_norm)

    return xr.Dataset(
                data_vars=dict(
                    frequency=("t", freq),
                    fmin=("t", fmin),
                    fmax=("t", fmax),
                    power=("t", power),
                    noise=("t", noise),
                    snr=("t", snr),
                ),
                coords=dict(t=t),
            )