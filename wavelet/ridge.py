import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import wavelet_funcs as wf

def is_jumping(freq, thresh=1e-5, axis=-1):
    """
    Check if the ridge frequency is 'jumping'.
    
    Parameters
    ----------
    
        freq : nd array
            Ridge frequency values.
        thresh : float
            If abs(np.diff(freq)) > thresh, there is a jump.
        axis : defaulf -1
            Axis along which to compute the difference.
            
    Returns
    -------
    
        df : nd array
            np.diff(freq)
            
        isjumping : bool
            True if any value in abs(df) > thresh
    
    Notes
    -----
    
    Possible reasons for 'jumps' in the ridge freuqency:
        - 'rings' in the scalogram (beating oscillatios, phase shifts)
        - decrease in oscillation amplitude such that it is indistinguishable
            from noise.
        
    """
    df = np.diff(freq, n=1, axis=axis)
    return df, np.any(abs(df) > thresh, axis=axis)

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
    """Block-averages an xarray along ``t`` dimension
    into ``num_segments`` blocks. Computes the block size
    as ``p.t.size//num_segments``. Trims the end.
    """
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

def spectral_peak_score(fc, peak_freq, peak_power, noise_power, lam=0.0):
    # Relative power
    rel_power = peak_power / noise_power

    # Relative frequency change
    rel_df = np.abs(peak_freq - fc) / fc

    # Higher score is better
    score = rel_power - lam * rel_df**2

    return score

def remove_short_segments(signif, t, min_duration):
    """
    Parameters
    ----------
    signif : array-like of bool
    t : array-like
        Time coordinate (numeric or datetime64).
    min_duration : float or np.timedelta64
        Minimum allowed segment duration.
    """
    signif = np.asarray(signif).copy()

    # Find starts and ends of True runs
    x = np.concatenate([[False], signif, [False]])
    edges = np.diff(x.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for s, e in zip(starts, ends):
        duration = t[e - 1] - t[s]
        if duration < min_duration:
            signif[s:e] = False

    return signif


def coi_intersections(t, freq, coi):
    """
    Determine the times at which a time-varying frequency intersects the COI.

    Parameters
    ----------
    t : array-like
        Time coordinate (numeric or datetime64).
    freq : array-like
        Time-varying frequency.
    coi : array-like
        Cone-of-influence frequency at each time.

    Returns
    -------
    crossings : ndarray
        Times at which freq crosses the COI.
    """
    t = np.asarray(t)
    freq = np.asarray(freq)
    coi = np.asarray(coi)

    d = freq - coi

    # Indices where the sign changes
    idx = np.where(d[:-1] * d[1:] < 0)[0]

    crossings = []

    for i in idx:
        # Fraction between t[i] and t[i+1]
        alpha = -d[i] / (d[i + 1] - d[i])

        # if np.issubdtype(t.dtype, np.datetime64):
        #     dt = t[i + 1] - t[i]
        #     tcross = t[i] + alpha * dt
        # else:
        tcross = t[i] + alpha * (t[i + 1] - t[i])

        crossings.append(tcross)

    return np.asarray(crossings)

def process_band(profile, noise_floor, fp, frac):
    """
    For each peak location in `fp`, compute the relative band,
    the integrated peak power, the noise power, and the normalized
    peak power. 
    
    Parameters
    ----------
        profile : 1darray
        fp : scalar or ndarray
            Candidate peak frequencies.
        frac : scalar
            Between 0 and 1. Used to compute band boundaries
            around fp.

    Returns
    -------
        band : tuple(ndarray, ndarray)
            Lower and upper band edges.
        pwr : ndarray
        noise_power : ndarray
        pwr_norm : ndarray
    """
    scalar = np.ndim(fp) == 0
    fp = np.atleast_1d(fp)

    band_lo = fp * (1 - frac)
    band_hi = fp * (1 + frac)

    pwr = np.array([integrate_peak(profile, (lo, hi)) 
                    for lo, hi in zip(band_lo, band_hi)])

    noise_power = noise_floor * (band_hi - band_lo)
    pwr_norm = pwr / noise_power

    if scalar:
        return (band_lo[0], band_hi[0]), pwr[0], noise_power[0], pwr_norm[0]
    else:
        return (band_lo, band_hi), pwr, noise_power, pwr_norm

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
    power_norm = []

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
        power_norm.append(pwr_norm)

    return xr.Dataset(
                data_vars=dict(
                    frequency=("t", freq),
                    fmin=("t", fmin),
                    fmax=("t", fmax),
                    power=("t", power),
                    noise=("t", noise),
                    power_norm=("t", power_norm),
                ),
                coords=dict(t=t),
            )

def track_band2(
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
        
    f0 : array-like or floar
        Reference frequency at each time step. Used only as the center of the
        search window. The tracked frequency remains the previously accepted
        peak if no valid peak is found. If float, the reference frequency is 
        this value at each time.

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
    
    if np.ndim(f0) == 0:
        f0 = np.ones(t.size)*f0

    # Ensure f0 is an array with one value per profile
    if isinstance(f0, xr.DataArray):
        f0 = f0.values
    else:
        f0 = np.asarray(f0)

    if len(f0) != len(profiles):
        raise ValueError("f0 must have one value per time step.")

    freq = []
    fmin = []
    fmax = []
    power = []
    noise = []
    power_norm = []

    # Initialize tracked frequency from first reference value
    fc = f0[0]

    def _process_profile(profile, band):
        pwr = integrate_func(profile, band)
        bandwidth = band[1] - band[0]
        noise_power = noise_floor * bandwidth
        pwr_norm = pwr / noise_power
        return pwr, noise_power, pwr_norm

    for profile, fref in zip(profiles, f0):

        # Search around the reference ridge, not the previous tracked peak
        search_band = (
            fref * (1 - frac_search),
            fref * (1 + frac_search),
        )

        x = profile.sel(f=slice(*search_band))

        fp = peak_freq(x.values, x.f.values)
        fp = interpolate_peak(profile.values, profile.f.values, fp)

        if np.isfinite(fp):
            candidate_band = (fp * (1 - frac), fp * (1 + frac))

            pwr, noise_power, pwr_norm = _process_profile(profile, candidate_band)

            # Accept new peak only if sufficiently significant
            if pwr_norm > threshold_k:
                fc = fp
                band = candidate_band
            else:
                # Keep previous tracked frequency
                band = (fc * (1 - frac), fc * (1 + frac))
                pwr, noise_power, pwr_norm = _process_profile(profile, band)

        else:
            # No detected peak: keep previous tracked frequency
            band = (fc * (1 - frac), fc * (1 + frac))
            pwr, noise_power, pwr_norm = _process_profile(profile, band)
        
        freq.append(fc)
        fmin.append(band[0])
        fmax.append(band[1])
        power.append(pwr)
        noise.append(noise_power)
        power_norm.append(pwr_norm)

    return xr.Dataset(
        data_vars=dict(
            frequency=("t", freq),
            fmin=("t", fmin),
            fmax=("t", fmax),
            power=("t", power),
            noise=("t", noise),
            power_norm=("t", power_norm),
        ),
        coords=dict(t=t),
    )

import numpy as np
import xarray as xr


def track_band_seed_bidirectional(
    profiles,
    f0,
    integrate_func,
    noise_floor,
    frac_search=0.5,
    frac=0.3,
    threshold_k=3,
    track_threshold_k=None,
):
    """
    Track an oscillation band using an initial ridge to find seeds, then
    replace the initial ridge with ridges propagated bidirectionally
    from each seed.

    Parameters
    ----------
    profiles : xr.DataArray
        Dimensions (t, f).

    f0 : array-like or float
        Reference frequency at each time step. Used only to construct the
        initial ridge from which seeds are obtained.

    integrate_func : callable
        integrate_func(profile, band) -> power

    noise_floor : float
        Noise power density.

    frac_search : float, default 0.5
        Relative half-bandwidth used when searching for a peak.

    frac : float, default 0.3
        Relative half-bandwidth used for power integration.

    threshold_k : float, default 3
        Threshold for identifying seeds.

    track_threshold_k : float, optional
        Threshold used when propagating from seeds. If None, uses
        threshold_k / 2.

    Returns
    -------
    xr.Dataset
        Final merged ridge.

    Notes
    -----
    The algorithm has three stages:

    1. Construct an initial ridge using the original f0-based method.
    2. Find all seed points where normalized power exceeds threshold_k.
    3. Starting from every seed, propagate forward and backward using
       track_threshold_k.

    The propagated ridges replace the initial ridge wherever they
    produce a valid track. Where propagation temporarily fails, the
    initial ridge is used as a fallback and propagation continues from
    that frequency.
    """

    t = profiles.t.values
    n = len(profiles)

    if track_threshold_k is None:
        track_threshold_k = threshold_k / 2

    # ------------------------------------------------------------------
    # Prepare f0
    # ------------------------------------------------------------------
    if np.ndim(f0) == 0:
        f0 = np.full(n, float(f0))
    elif isinstance(f0, xr.DataArray):
        f0 = f0.values
    else:
        f0 = np.asarray(f0)

    if len(f0) != n:
        raise ValueError("f0 must have one value per time step.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _band(center, fraction):
        return (
            center * (1 - fraction),
            center * (1 + fraction),
        )

    def _evaluate(profile, center):
        """
        Find the strongest peak near center and calculate its
        normalized integrated power.
        """
        if not np.isfinite(center) or center <= 0:
            return np.nan, None, np.nan, np.nan, np.nan

        search_band = _band(center, frac_search)

        x = profile.sel(f=slice(*search_band))

        if x.size == 0:
            return np.nan, None, np.nan, np.nan, np.nan

        fp = peak_freq(x.values, x.f.values)

        if not np.isfinite(fp):
            return np.nan, None, np.nan, np.nan, np.nan

        # Refine peak frequency using the full spectrum
        fp = interpolate_peak(
            profile.values,
            profile.f.values,
            fp,
        )

        if not np.isfinite(fp):
            return np.nan, None, np.nan, np.nan, np.nan

        candidate_band = _band(fp, frac)

        pwr = integrate_func(profile, candidate_band)

        bandwidth = candidate_band[1] - candidate_band[0]
        noise_power = noise_floor * bandwidth

        if noise_power > 0:
            pwr_norm = pwr / noise_power
        else:
            pwr_norm = np.nan

        return (
            fp,
            candidate_band,
            pwr,
            noise_power,
            pwr_norm,
        )

    # ==================================================================
    # 1. CONSTRUCT INITIAL RIDGE
    # ==================================================================
    #
    # This is essentially the old algorithm. It is NOT the final ridge.
    # It is only used to find candidate seed points and as a fallback
    # during propagation.
    #
    initial_freq = np.full(n, np.nan)
    initial_fmin = np.full(n, np.nan)
    initial_fmax = np.full(n, np.nan)
    initial_power = np.full(n, np.nan)
    initial_noise = np.full(n, np.nan)
    initial_power_norm = np.full(n, np.nan)

    fc = f0[0]

    for i, (profile, fref) in enumerate(zip(profiles, f0)):

        fp, band, pwr, noise_power, pwr_norm = _evaluate(
            profile,
            fref,
        )

        if np.isfinite(fp) and pwr_norm >= threshold_k:
            # Strong enough to update initial ridge
            fc = fp

            initial_freq[i] = fp
            initial_fmin[i] = band[0]
            initial_fmax[i] = band[1]
            initial_power[i] = pwr
            initial_noise[i] = noise_power
            initial_power_norm[i] = pwr_norm

        elif np.isfinite(fc):
            # Keep previous accepted frequency
            band = _band(fc, frac)

            pwr = integrate_func(profile, band)
            bandwidth = band[1] - band[0]
            noise_power = noise_floor * bandwidth

            if noise_power > 0:
                pwr_norm = pwr / noise_power
            else:
                pwr_norm = np.nan

            initial_freq[i] = fc
            initial_fmin[i] = band[0]
            initial_fmax[i] = band[1]
            initial_power[i] = pwr
            initial_noise[i] = noise_power
            initial_power_norm[i] = pwr_norm

    # ==================================================================
    # 2. FIND SEEDS
    # ==================================================================
    #
    # Seeds are obtained from the initial ridge. We use the conservative
    # threshold here.
    #
    seed_indices = np.where(
        initial_power_norm >= threshold_k
    )[0]

    if len(seed_indices) == 0:
        return xr.Dataset(
            data_vars=dict(
                frequency=("t", initial_freq),
                fmin=("t", initial_fmin),
                fmax=("t", initial_fmax),
                power=("t", initial_power),
                noise=("t", initial_noise),
                power_norm=("t", initial_power_norm),
                source=("t", np.zeros(n, dtype=int)),
            ),
            coords=dict(t=t),
        )

    # ==================================================================
    # 3. STORAGE FOR FINAL MERGED RIDGE
    # ==================================================================
    #
    # Start with the initial ridge. Propagated tracks will replace it.
    #
    final_freq = initial_freq.copy()
    final_fmin = initial_fmin.copy()
    final_fmax = initial_fmax.copy()
    final_power = initial_power.copy()
    final_noise = initial_noise.copy()
    final_power_norm = initial_power_norm.copy()

    # 0 = initial ridge
    # 1 = propagated ridge
    final_source = np.zeros(n, dtype=int)

    # Keep track of the confidence of propagated assignments.
    propagated_score = np.full(n, -np.inf)

    # ==================================================================
    # 4. PROPAGATE EACH SEED
    # ==================================================================
    def _propagate_from_seed(seed_i, direction):
        """
        Propagate one seed in one direction.

        direction = +1 -> forward
        direction = -1 -> backward

        When the propagated candidate falls below track_threshold_k,
        the initial ridge is used as a fallback. Propagation then
        continues from that fallback frequency.

        Returns
        -------
        list of dict
            Accepted points along this propagation.
        """

        results = []

        # Start from the seed itself
        fc = initial_freq[seed_i]

        if not np.isfinite(fc):
            return results

        i = seed_i

        while True:

            i += direction

            if i < 0 or i >= n:
                break

            profile = profiles.isel(t=i)

            # ----------------------------------------------------------
            # Try to continue the propagated ridge
            # ----------------------------------------------------------
            fp, band, pwr, noise_power, pwr_norm = _evaluate(
                profile,
                fc,
            )

            if (
                np.isfinite(fp)
                and np.isfinite(pwr_norm)
                and pwr_norm >= track_threshold_k
            ):
                # Strong enough: use newly detected peak
                accepted_freq = fp
                accepted_band = band
                accepted_power = pwr
                accepted_noise = noise_power
                accepted_norm = pwr_norm
                source = 1

            else:
                # ------------------------------------------------------
                # Weak propagation:
                # fall back to the original ridge at this time.
                # ------------------------------------------------------
                fallback_freq = initial_freq[i]

                if not np.isfinite(fallback_freq):
                    # No fallback available.
                    #
                    # We could terminate here, but instead simply skip
                    # this point and keep searching from the old fc.
                    continue

                accepted_freq = fallback_freq
                accepted_band = _band(
                    accepted_freq,
                    frac,
                )

                accepted_power = integrate_func(
                    profile,
                    accepted_band,
                )

                bandwidth = (
                    accepted_band[1] -
                    accepted_band[0]
                )

                accepted_noise = noise_floor * bandwidth

                if accepted_noise > 0:
                    accepted_norm = (
                        accepted_power /
                        accepted_noise
                    )
                else:
                    accepted_norm = np.nan

                source = 0

            # ----------------------------------------------------------
            # Store result
            # ----------------------------------------------------------
            results.append(
                {
                    "i": i,
                    "frequency": accepted_freq,
                    "band": accepted_band,
                    "power": accepted_power,
                    "noise": accepted_noise,
                    "power_norm": accepted_norm,
                    "source": source,
                }
            )

            # IMPORTANT:
            # Continue propagation from the frequency that was actually
            # accepted. Thus after a weak region, tracking can reconnect
            # to a strong peak later.
            fc = accepted_freq

        return results

    # ==================================================================
    # 5. RUN PROPAGATION FROM EVERY SEED
    # ==================================================================
    for seed_i in seed_indices:

        # --------------------------------------------------------------
        # The seed itself is always a propagated point.
        # --------------------------------------------------------------
        seed_result = {
            "i": seed_i,
            "frequency": initial_freq[seed_i],
            "band": (
                initial_fmin[seed_i],
                initial_fmax[seed_i],
            ),
            "power": initial_power[seed_i],
            "noise": initial_noise[seed_i],
            "power_norm": initial_power_norm[seed_i],
            "source": 1,
        }

        # --------------------------------------------------------------
        # Propagate both directions
        # --------------------------------------------------------------
        backward = _propagate_from_seed(
            seed_i,
            direction=-1,
        )

        forward = _propagate_from_seed(
            seed_i,
            direction=+1,
        )

        track = [seed_result] + backward + forward

        # --------------------------------------------------------------
        # Merge this propagated ridge into the final ridge.
        #
        # If multiple seeds generate overlapping ridges, keep the
        # propagated result with the highest normalized power.
        # --------------------------------------------------------------
        for result in track:

            i = result["i"]
            score = result["power_norm"]

            if not np.isfinite(score):
                continue

            if score > propagated_score[i]:

                propagated_score[i] = score

                final_freq[i] = result["frequency"]
                final_fmin[i] = result["band"][0]
                final_fmax[i] = result["band"][1]
                final_power[i] = result["power"]
                final_noise[i] = result["noise"]
                final_power_norm[i] = result["power_norm"]

                final_source[i] = 1

    # ==================================================================
    # 6. RETURN FINAL RIDGE
    # ==================================================================
    return xr.Dataset(
        data_vars=dict(
            frequency=("t", final_freq),
            fmin=("t", final_fmin),
            fmax=("t", final_fmax),
            power=("t", final_power),
            noise=("t", final_noise),
            power_norm=("t", final_power_norm),
            source=("t", final_source),
        ),
        coords=dict(t=t),
    )

def track_band_scored(
    profiles,
    f0,
    noise_floor,
    frac_search=0.5,
    frac=0.3,
    lam=0.0,
):
    """
    Note: might give unpredictable results....
    
    Track a wavelet ridge by maximizing

        score = relative_power - lam * relative_frequency_change

    over all peaks in the search window.
    
    Parameters
    ----------
        profiles : 2d array
            Set of profiles, in which to search for the ridge.
            
        f0 : 1d array or scalar
            Initial guess for the ridge frequency. If scalar, this 
            value used as initial guess for each profile.

        noise_floor : float
            Mean (over time and frequency) noise floor.
            
        frac_search : float
            Fractional half-width of the band around ridge frequency.
            Used to search for peaks around the initial frequency guess.
            Computed as `[f_init*(1-frac_search), f_init*(1+frac_search)]`.
        
        frac : float
            Fractional half-width of the band around ridge frequency.
            Used to integrate power in it. Calculated as 
            [f_ridge*(1-frac), f_ridge*(1+frac)].
        
        lam : scalar
            Penalty coefficient for the relative frequency change.

    Returns
    -------
        xarray.Dataset
    """

    t = profiles.t.values
    
    if np.ndim(f0) == 0:
        f0 = np.ones(t.size)*f0

    if isinstance(f0, xr.DataArray):
        f0 = f0.values
    else:
        f0 = np.asarray(f0)

    if len(f0) != len(profiles):
        raise ValueError("f0 must have one value per time step.")

    freq = []
    fmins = []
    fmaxs = []
    power = []
    noise = []
    power_norm = []

    # Initial ridge
    fc = f0[0]

    for profile, fref in zip(profiles, f0):

        search_band = (
            fref * (1 - frac_search),
            fref * (1 + frac_search),
        )

        x = profile.sel(f=slice(*search_band))

        # Find every local maximum
        peak_idx, _ = find_peaks(x.values)

        if len(peak_idx):

            # Interpolate all peak locations
            fp = np.array([
                interpolate_peak(profile.values,
                                 profile.f.values,
                                 x.f.values[i])
                for i in peak_idx
            ])

            # Evaluate every candidate
            band, pwr, noise_power, pwr_norm = process_band(profile, noise_floor, fp, frac)

            scores = spectral_peak_score(fc, fp, pwr, noise_power, lam)
            best = np.argmax(scores)
            fc = fp[best]

            fmin = band[0][best]
            fmax = band[1][best]
            pwr = pwr[best]
            noise_power = noise_power[best]
            pwr_norm = pwr_norm[best]
        else:
            (fmin, fmax), pwr, noise_power, pwr_norm = process_band(profile, noise_floor, fc, frac)
            
        
        freq.append(fc)
        fmins.append(fmin)
        fmaxs.append(fmax)
        power.append(pwr)
        noise.append(noise_power)
        power_norm.append(pwr_norm)

    
    return xr.Dataset(
        data_vars=dict(
            frequency=("t", freq),
            fmin=("t", fmins),
            fmax=("t", fmaxs),
            power=("t", power),
            noise=("t", noise),
            power_norm=("t", power_norm),
        ),
        coords=dict(t=t),
    )


def analyze_scalogram(scalogram, freq_band_coarse, noise, threshold_k=3, upsample=10):
    """
    Quantify oscillation power within a time-varying frequency band.

    The frequency band is obtained by interpolating a coarse estimate of the
    band center and limits onto the time axis of the input scalogram. At each
    time point, the wavelet power is integrated within the interpolated
    frequency band and normalized by the expected integrated noise power.

    Parameters
    ----------
    scalogram : xr.DataArray
        Wavelet power scalogram with dimensions including ``t`` (time) and
        ``f`` (frequency).
    freq_band_coarse : xr.Dataset
        Dataset describing the oscillation frequency band on a (possibly
        coarser) time grid. It must contain the variables:

        - ``frequency`` : center frequency,
        - ``fmin`` : lower band limit,
        - ``fmax`` : upper band limit.

        All variables must be indexed by the coordinate ``t``.
    noise : float
        Mean noise floor of the scalogram. Used to compute time-varying
        noise power (because the relative bandwidth is preserved, and
        fc varies in time).
    threshold_k : float, default 3
        Threshold applied to the normalized band power. Time points with
        ``power_norm > threshold_k`` are marked as significant.
        
    upsample : int, default 10
        Upsampling factor to linearly interpolate the scalogram along the
        frequency axis prior to integration.

    Returns
    -------
    xr.Dataset
        Dataset indexed by the scalogram time coordinate containing:

        - ``frequency`` : interpolated center frequency,
        - ``fmin`` : interpolated lower band limit,
        - ``fmax`` : interpolated upper band limit,
        - ``power`` : wavelet power integrated within the frequency band,
        - ``power_noise`` : expected integrated noise power within the band,
        - ``power_norm`` : normalized band power
          (``power / power_noise``),
        - ``significance`` : boolean indicating whether the normalized power
          exceeds ``threshold_k``.

        The dataset attribute ``threshold_for_power_norm`` stores the value of
        ``threshold_k`` used for the analysis.
    """
    
    results = xr.Dataset(
        data_vars = dict(
            frequency = ('t', interpolate(freq_band_coarse.frequency.data, freq_band_coarse.t.data, scalogram.t.data)),
            fmin = ('t', interpolate(freq_band_coarse.fmin.data, freq_band_coarse.t.data, scalogram.t.data)),
            fmax = ('t', interpolate(freq_band_coarse.fmax.data, freq_band_coarse.t.data, scalogram.t.data)),
        ),
        coords = dict(t=scalogram.t.data)
    )

    ### Linearly interpolate scalogram for integration
    freq_fine = np.geomspace(
        scalogram.f.max().item(),
        scalogram.f.min().item(),
        upsample * (scalogram.sizes["f"] - 1) + 1,
    )
    scalogram_fine = scalogram.interp(f=freq_fine)

    ### Integrate to estimate band power
    results['power'] = scalogram_fine[::-1].where((scalogram_fine.f>results.fmin)&(scalogram_fine.f<results.fmax)).fillna(0).integrate('f') # CWT power is selected between frequency bands. Everything outside the band is NaN. Replace those NaNs with 0 to compute trapz.
    # results['power'] = scalogram_fine[::-1].where((scalogram_fine.f>results.fmin)&(scalogram_fine.f<results.fmax)).integrate('f')
    results['power_noise'] = noise * (results.fmax - results.fmin)
    results['power_norm'] = results.power / results.power_noise
    results['significance'] = results.power_norm > threshold_k
    
    results.attrs['threshold_for_power_norm'] = threshold_k
    
    return results

def plot_band_tracking(
    pv,
    tracks,
    threshold_k,
    bv=None,
    coi=None,
    colors=None,
    labels=None,
    ylim=(None, 0.06),
    figsize=(8, 8),
    track_init=None,
):
    """
    Plot wavelet spectrum with tracked bands and normalized power.

    Parameters
    ----------
    pv : xr.DataArray
        Wavelet power scalogram.
    tracks : list of xr.Dataset
        Ridge tracking results.
    threshold_k : float
        Detection threshold.
    bv : xr.DataArray, optional
        Time series to plot above the scalogram. A Gaussian-smoothed version
        (sigma=6) is overlaid.
    coi : xr.DataArray, optional
        Cone of influence.
    colors : list, optional
        Colors for each track.
    labels : list, optional
        Labels for the legend.
    ylim : tuple, default=(None, 0.06)
        Frequency limits.
    figsize : tuple, default=(8, 6)
    track_init : xr.DataArray
        Initial guess used for tracking.

    Returns
    -------
    fig, axes
    """

    if colors is None:
        colors = ["w"] * len(tracks)

    if bv is None:
        fig, (ax, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axes = (ax, ax1)
    else:
        fig, (ax0, ax, ax1) = plt.subplots( 3, 1, figsize=figsize, sharex=True, height_ratios= [1, 1.5, 1])
        axes = (ax0, ax, ax1)

    plt.subplots_adjust(hspace=0.1)

    if bv is not None:
        bv.plot(ax=ax0, color='k', alpha=0.2)
        ax0.plot(bv.t, wf.gaussian_filter(bv, sigma=6), color="k")
        ax0.set(xlabel="", ylabel=bv.name or "")

    wf.plot_cwt(pv, pv.f, pv.t, ax=ax, add_colorbar=True)
    if coi is not None:
        coi.plot(ax=ax, ls='--', color='w', alpha=0.5)

    if track_init is not None:
        if np.issubdtype(type(track_init), np.number): # Scalar and not string
            ax.axhline(track_init, color='r', ls='--', alpha=0.7)
        else:
            track_init.plot(ax=ax, color='r', ls='--', alpha=0.7)
        
    for i, (track, color) in enumerate(zip(tracks, colors)):
        label = None if labels is None else labels[i]

        track.frequency.plot( ax=ax, color=color, ls="--", alpha=0.7, label=label)

        ax.fill_between( track.t, track.fmin, track.fmax, color=color, alpha=0.03)

        track.power_norm.plot( ax=ax1, label=label)

    
    
    if ylim[0] is None:
        ylim = (pv.f.min().item(), ylim[1])
    ax.set_ylim(*ylim)
    ax.set(xlabel="", ylabel="Frequency (Hz)", title='')

    ax1.axhline(threshold_k, color="k", ls="--")
    ax1.set( xlabel="Time (s)", ylabel="Band power / Noise power", title="")

    if labels is not None:
        ax.legend()
        ax1.legend()

    return fig, axes
