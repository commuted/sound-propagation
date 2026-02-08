#!/usr/bin/env python3
"""Plot PSD at three positions: source, listening point, and target.

Shows the frequency-dependent power spectral density of pink noise as it
would be heard at the source, at the shared listening point, and at a
target evaluation distance — illustrating how ISO 9613 atmospheric
absorption preferentially removes high-frequency energy with distance.

Usage:
    python examples/plot_propagation_psd.py
    python examples/plot_propagation_psd.py --target-distance 100
    python examples/plot_propagation_psd.py --target-distance 500 --output plot.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, fftconvolve

from sound_propagation import AtmosphericPropagation

# Re-use the FIR pipeline for high-resolution filtering
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "propagation_filter"))
from propagation_fir_filter import design_propagation_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PSD at source, listener, and target positions."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input WAV file (default: generate 5 s pink noise)",
    )
    parser.add_argument(
        "--listener-distance",
        type=float,
        default=200.0,
        help="Source-to-listener distance in metres (default: 200)",
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=500.0,
        help="Source-to-target distance in metres (default: 500)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=20.0,
        help="Ambient temperature in degrees Celsius (default: 20)",
    )
    parser.add_argument(
        "--humidity",
        type=float,
        default=70.0,
        help="Relative humidity in percent (default: 70)",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=101.325,
        help="Atmospheric pressure in kPa (default: 101.325)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save plot to file instead of showing (e.g., plot.png)",
    )
    return parser.parse_args()


def generate_pink_noise(duration, sr, seed=42):
    """Generate pink noise. Uses custom_noise if available, else fallback."""
    try:
        from custom_noise import pink_noise
        return pink_noise(duration=duration, sample_rate=sr,
                          random_seed=seed, dtype="float64")
    except ImportError:
        # Fallback: shape white noise spectrum by 1/sqrt(f)
        rng = np.random.default_rng(seed)
        n = int(duration * sr)
        n_freqs = n // 2 + 1
        phases = rng.uniform(0, 2 * np.pi, n_freqs)
        spectrum = np.exp(1j * phases)
        freqs = np.arange(n_freqs)
        freqs[0] = 1
        spectrum *= 1.0 / np.sqrt(freqs)
        spectrum[0] = 0
        signal = np.fft.irfft(spectrum, n=n)
        peak = np.max(np.abs(signal))
        return signal / peak if peak > 0 else signal


def filter_at_distance(signal, sr, prop, distance, numtaps):
    """Apply normalized FIR propagation filter (atmospheric absorption only).

    Returns the filtered signal and the geometric spreading in dB.
    """
    fir, _, _, geometric_db = design_propagation_filter(
        sr, numtaps, prop, distance, normalize=True
    )
    return fftconvolve(signal, fir, mode="same"), geometric_db


def main():
    args = parse_args()

    sr = 44100
    nperseg = 8192

    # Load or generate source signal
    if args.input:
        from scipy.io import wavfile
        sr, data = wavfile.read(args.input)
        if data.dtype == np.int16:
            signal = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            signal = data.astype(np.float64) / 2147483648.0
        else:
            signal = data.astype(np.float64)
        if signal.ndim > 1:
            signal = signal[:, 0]  # use first channel
    else:
        signal = generate_pink_noise(duration=5.0, sr=sr)

    # Propagation model: source at origin, recording at 1 m reference
    prop = AtmosphericPropagation(
        temperature_c=args.temperature,
        relative_humidity_pct=args.humidity,
        pressure_kpa=args.pressure,
        source=(0.0, 0.0, 0.0),
        recording=(1.0, 0.0, 0.0),
    )

    numtaps = sr | 1  # odd, ~1 Hz resolution

    # Filter at listener and target distances (atmospheric absorption only)
    sig_listener, geo_listener_db = filter_at_distance(
        signal, sr, prop, args.listener_distance, numtaps
    )
    sig_target, geo_target_db = filter_at_distance(
        signal, sr, prop, args.target_distance, numtaps
    )
    geo_listener_lin = 10.0 ** (geo_listener_db / 20.0)
    geo_target_lin = 10.0 ** (geo_target_db / 20.0)

    # Compute PSDs
    f, psd_source = welch(signal, sr, window="hann", nperseg=nperseg,
                          noverlap=nperseg // 2)
    _, psd_listener = welch(sig_listener, sr, window="hann", nperseg=nperseg,
                            noverlap=nperseg // 2)
    _, psd_target = welch(sig_target, sr, window="hann", nperseg=nperseg,
                          noverlap=nperseg // 2)

    # Convert to dB (relative to source peak)
    ref = np.max(psd_source)
    psd_source_db = 10 * np.log10(psd_source / ref)
    psd_listener_db = 10 * np.log10(psd_listener / ref)
    psd_target_db = 10 * np.log10(psd_target / ref)

    # Mask out DC for cleaner log-scale plot
    mask = f > 0

    d_listen = args.listener_distance
    d_target = args.target_distance

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(f[mask], psd_source_db[mask],
                label="Source (0 m)", color="#2196F3", linewidth=1.5)
    ax.semilogx(f[mask], psd_listener_db[mask],
                label=f"Listener ({d_listen:.0f} m)", color="#4CAF50", linewidth=1.5)
    ax.semilogx(f[mask], psd_target_db[mask],
                label=f"Target ({d_target:.0f} m)", color="#F44336", linewidth=1.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re source peak)")
    ax.set_title(
        f"ISO 9613 Atmospheric Absorption (normalized) \u2014 "
        f"{args.temperature}\u00b0C, {args.humidity}% RH, {args.pressure} kPa"
    )
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(20, sr / 2)

    # Geometric spreading coefficients box
    coeff_text = (
        "Geometric spreading (constant):\n"
        f"  Listener {d_listen:.0f} m: {geo_listener_db:+.1f} dB  "
        f"(k = {geo_listener_lin:.6f})\n"
        f"  Target {d_target:.0f} m:   {geo_target_db:+.1f} dB  "
        f"(k = {geo_target_lin:.6f})"
    )
    ax.text(
        0.98, 0.98, coeff_text,
        transform=ax.transAxes,
        fontsize=8,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
