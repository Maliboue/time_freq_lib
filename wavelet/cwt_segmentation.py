import numpy as np
from scipy import signal
from scipy.stats import chi2
from statsmodels.tsa.ar_model import AutoReg
import pywt


def preprocess_signal(x, detrend=True, demean=True):
    """Return cleaned 1D signal."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if detrend:
        x = signal.detrend(x)

    if demean:
        x = x - np.mean(x)

    return x


def remove_frequency_band(x, fs, fmin=0.01, fmax=0.08, order=4):
    """
    Remove oscillatory band before AR fitting.
    Uses zero-phase Butterworth band-stop filter.
    """
    nyq = fs / 2
    sos = signal.butter(
        order,
        [fmin / nyq, fmax / nyq],
        btype="bandstop",
        output="sos"
    )
    return signal.sosfiltfilt(sos, x)


def fit_ar_model(x, order=1):
    """
    Fit AR(order) model to signal.
    For Torrence-Compo style red noise, use order=1.
    """
    model = AutoReg(x, lags=order, trend="c", old_names=False)
    result = model.fit()

    intercept = result.params[0]
    ar_coeffs = result.params[1:]
    noise_std = np.std(result.resid, ddof=1)

    return {
        "result": result,
        "intercept": intercept,
        "ar_coeffs": np.asarray(ar_coeffs),
        "noise_std": noise_std,
        "order": order
    }


def simulate_ar_surrogate(ar_params, n, burnin=500):
    """
    Simulate one AR surrogate with fitted AR parameters.
    """
    intercept = ar_params["intercept"]
    phi = ar_params["ar_coeffs"]
    noise_std = ar_params["noise_std"]
    order = len(ar_params["ar_coeffs"])

    y = np.zeros(n + burnin)

    for t in range(order, n + burnin):
        past = y[t-order:t][::-1]
        y[t] = intercept + np.dot(phi, past) + np.random.normal(0, noise_std)

    return y[burnin:]


def compute_cwt_power(
    x,
    fs,
    freqs,
    wavelet="cmor1.5-1.0",
    pad_fraction=0.2,
    pad_mode="symmetric"
):
    """
    Compute CWT power with optional symmetric padding using numpy.pad.

    Padding is removed before returning results.
    """

    x = np.asarray(x)
    dt = 1 / fs
    n = len(x)

    # --- compute padding length ---
    pad_len = int(np.round(n * pad_fraction))

    # --- apply padding ---
    if pad_mode == "constant":
        x_pad = np.pad(x, (pad_len, pad_len), mode=pad_mode, constant_values=0)
    else:
        x_pad = np.pad(x, (pad_len, pad_len), mode=pad_mode)

    # --- compute scales ---
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq / (freqs * dt)

    # --- CWT ---
    coeffs_pad, _ = pywt.cwt(x_pad, scales, wavelet, sampling_period=dt)
    power_pad = np.abs(coeffs_pad) ** 2

    # --- remove padding ---
    if pad_len > 0:
        coeffs = coeffs_pad[:, pad_len:-pad_len]
        power = power_pad[:, pad_len:-pad_len]
    else:
        coeffs = coeffs_pad
        power = power_pad

    return power, coeffs, scales


def monte_carlo_cwt_threshold(
    ar_params,
    n,
    fs,
    freqs,
    n_surrogates=1000,
    alpha=0.95,
    wavelet="cmor1.5-1.0",
    pad_fraction=0.2,
    pad_mode="symmetric"
):
    """
    Generate AR surrogates, compute CWT power, and estimate
    frequency-specific thresholds using identical padding.
    """

    all_power = []

    for _ in range(n_surrogates):
        surrogate = simulate_ar_surrogate(ar_params, n)

        pwr, _, _ = compute_cwt_power(
            surrogate,
            fs,
            freqs,
            wavelet=wavelet,
            pad_fraction=pad_fraction,
            pad_mode=pad_mode
        )

        all_power.append(pwr)

    all_power = np.stack(all_power, axis=0)

    # Quantile across surrogates and time
    threshold = np.quantile(all_power, alpha, axis=(0, 2))

    return threshold[:, None]


def segment_significant_power(power, threshold):
    """
    Segment CWT coefficients above significance threshold.
    """
    return power > threshold


def remove_small_clusters(mask, min_size=20):
    """
    Remove small connected significant regions in time-frequency space.
    """
    from scipy.ndimage import label

    labeled, n_labels = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)

    for lab in range(1, n_labels + 1):
        cluster = labeled == lab
        if np.sum(cluster) >= min_size:
            cleaned[cluster] = True

    return cleaned


def run_ar_cwt_significance_pipeline(
    x,
    fs,
    fmin=0.01,
    fmax=0.08,
    n_freqs=50,
    ar_order=1,
    exclude_band_before_ar=True,
    n_surrogates=1000,
    alpha=0.95,
    min_cluster_size=20
):
    """
    Full pipeline:
    1. preprocess signal
    2. optionally remove oscillatory band for AR fitting
    3. fit AR noise model
    4. compute observed CWT power
    5. compute Monte Carlo AR threshold
    6. segment significant CWT coefficients
    """

    x0 = preprocess_signal(x)

    if exclude_band_before_ar:
        x_for_ar = remove_frequency_band(x0, fs, fmin=fmin, fmax=fmax)
    else:
        x_for_ar = x0.copy()

    ar_params = fit_ar_model(x_for_ar, order=ar_order)

    freqs = np.linspace(fmin, fmax, n_freqs)

    power, coeffs, scales = compute_cwt_power(x0, fs, freqs)

    threshold = monte_carlo_cwt_threshold(
        ar_params,
        n=len(x0),
        fs=fs,
        freqs=freqs,
        n_surrogates=n_surrogates,
        alpha=alpha
    )

    significant = segment_significant_power(power, threshold)

    significant_clean = remove_small_clusters(
        significant,
        min_size=min_cluster_size
    )

    return {
        "x": x0,
        "freqs": freqs,
        "scales": scales,
        "coeffs": coeffs,
        "power": power,
        "threshold": threshold,
        "significant": significant,
        "significant_clean": significant_clean,
        "ar_params": ar_params
    }