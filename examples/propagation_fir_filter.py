#!/usr/bin/env python3
"""Apply ISO 9613 atmospheric propagation attenuation via FIR filter.

Designs a linear-phase FIR filter whose magnitude response matches the
ISO 9613 frequency-dependent attenuation curve (atmospheric absorption +
geometric spreading), then convolves it with the input audio.

Compared to the STFT approach in propagation_filter.py, this gives much
finer frequency resolution (~1 Hz when numtaps equals the sample rate)
at the cost of longer computation time.

Usage:
    python examples/propagation_fir_filter.py input.wav --distance 500
    python examples/propagation_fir_filter.py input.wav --distance 200 --numtaps 88200
"""

import argparse
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin2, fftconvolve, welch

from sound_propagation import AtmosphericPropagation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply ISO 9613 propagation attenuation via FIR filter."
    )
    parser.add_argument("input", help="Path to input WAV file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV path (default: <input>_fir_attenuated.wav)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=500.0,
        help="Source-to-evaluation distance in metres (default: 500)",
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
        "--numtaps",
        type=int,
        default=0,
        help="FIR filter length (default: sample rate, giving ~1 Hz resolution). "
             "Must be odd; even values are incremented by 1.",
    )
    return parser.parse_args()


def read_wav(path):
    """Read a WAV file and return (sample_rate, data) with float64 in [-1, 1]."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    elif data.dtype != np.float64:
        raise ValueError(f"Unsupported WAV sample format: {data.dtype}")
    return sr, data


def write_wav(path, sr, data):
    """Write float64 data as 16-bit WAV, clipping to [-1, 1]."""
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(path, sr, (data * 32767).astype(np.int16))


def design_propagation_filter(sr, numtaps, prop, distance):
    """Design an FIR filter matching the ISO 9613 attenuation curve.

    Parameters
    ----------
    sr : int
        Sample rate in Hz.
    numtaps : int
        FIR filter length (odd). Controls frequency resolution:
        resolution ~= sr / numtaps Hz.
    prop : AtmosphericPropagation
        Configured propagation model.
    distance : float
        Evaluation distance in metres.

    Returns
    -------
    ndarray
        FIR filter coefficients (length numtaps).
    ndarray
        Frequency points used for the design (Hz).
    ndarray
        Target gain at each frequency point (dB).
    """
    nyquist = sr / 2.0

    # Dense frequency grid from 0 to Nyquist.  firwin2 wants frequencies
    # normalised to [0, 1] where 1 = Nyquist, so we build both scales.
    n_points = numtaps // 2 + 1
    freqs_hz = np.linspace(0, nyquist, n_points)
    freqs_norm = freqs_hz / nyquist  # [0, 1] for firwin2

    # Compute attenuation at every frequency point
    attenuation_db = np.zeros(n_points)
    mask = freqs_hz > 0
    att = prop.attenuation_at_position(freqs_hz[mask], (distance, 0.0, 0.0))
    attenuation_db[mask] = att["total_dB"]

    # Convert dB to linear amplitude gain for firwin2
    gain_linear = 10.0 ** (attenuation_db / 20.0)

    # firwin2 requires gain[0] at freq 0 and gain[-1] at freq 1 (Nyquist)
    fir = firwin2(numtaps, freqs_norm, gain_linear)

    return fir, freqs_hz, attenuation_db


def main():
    args = parse_args()

    # Resolve output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_fir_attenuated{ext}"
    else:
        output_path = args.output

    # Read input
    sr, data = read_wav(args.input)
    is_mono = data.ndim == 1
    n_channels = 1 if is_mono else data.shape[1]
    n_samples = data.shape[0]

    # Default numtaps to sample rate (giving ~1 Hz resolution)
    numtaps = args.numtaps if args.numtaps > 0 else sr
    # firwin2 requires odd numtaps for type I linear phase
    if numtaps % 2 == 0:
        numtaps += 1

    freq_resolution = sr / numtaps

    print(f"Input:       {args.input}")
    print(f"Sample rate: {sr} Hz")
    print(f"Channels:    {n_channels}")
    print(f"Duration:    {n_samples / sr:.2f} s")
    print()

    # Set up propagation model: source at origin, recording at 1 m (reference)
    prop = AtmosphericPropagation(
        temperature_c=args.temperature,
        relative_humidity_pct=args.humidity,
        pressure_kpa=args.pressure,
        source=(0.0, 0.0, 0.0),
        recording=(1.0, 0.0, 0.0),
    )

    print(f"Temperature: {args.temperature} C")
    print(f"Humidity:    {args.humidity}%")
    print(f"Pressure:    {args.pressure} kPa")
    print(f"Distance:    {args.distance} m")
    print()
    print(f"FIR taps:    {numtaps}")
    print(f"Freq resolution: {freq_resolution:.2f} Hz")
    print()

    # Design the FIR filter
    fir, freqs_hz, attenuation_db = design_propagation_filter(
        sr, numtaps, prop, args.distance
    )

    # Print attenuation at key frequencies
    print("Attenuation at key frequencies:")
    for target_f in [125, 500, 1000, 4000]:
        idx = np.argmin(np.abs(freqs_hz - target_f))
        print(f"  {target_f:5d} Hz: {attenuation_db[idx]:+.1f} dB")
    print()

    # Welch PSD of original (informational)
    nperseg_welch = min(4096, n_samples)
    if is_mono:
        f_welch, psd_before = welch(data, sr, nperseg=nperseg_welch)
    else:
        f_welch, psd_before = welch(data[:, 0], sr, nperseg=nperseg_welch)

    # Apply FIR filter to each channel via FFT-based convolution
    print(f"Filtering (fftconvolve, {n_channels} channel(s))...")
    if is_mono:
        filtered = fftconvolve(data, fir, mode="same")
    else:
        channels = []
        for ch in range(n_channels):
            channels.append(fftconvolve(data[:, ch], fir, mode="same"))
        filtered = np.column_stack(channels)

    # Write output
    write_wav(output_path, sr, filtered)
    print(f"Output:      {output_path}")

    # PSD comparison at 1 kHz
    if is_mono:
        _, psd_after = welch(filtered, sr, nperseg=nperseg_welch)
    else:
        _, psd_after = welch(filtered[:, 0], sr, nperseg=nperseg_welch)

    idx_1k = np.argmin(np.abs(f_welch - 1000))
    if psd_before[idx_1k] > 0 and psd_after[idx_1k] > 0:
        measured = 10 * np.log10(psd_after[idx_1k] / psd_before[idx_1k])
        print(f"Measured PSD change at 1 kHz: {measured:+.1f} dB")


if __name__ == "__main__":
    main()
