#!/usr/bin/env python3
"""Apply ISO 9613 atmospheric propagation attenuation to a WAV file.

Loads a WAV file, computes frequency-dependent attenuation (atmospheric
absorption + geometric spreading) using the sound_propagation library,
applies it as an STFT-domain filter, and writes the attenuated result.

Usage:
    python examples/propagation_filter.py input.wav --distance 500
"""

import argparse
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft, welch

from sound_propagation import AtmosphericPropagation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply ISO 9613 atmospheric sound propagation to a WAV file."
    )
    parser.add_argument("input", help="Path to input WAV file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV path (default: <input>_attenuated.wav)",
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
    return parser.parse_args()


def read_wav(path):
    """Read a WAV file and return (sample_rate, data) with float64 in [-1, 1].

    Returns data with shape (n_samples,) for mono or (n_samples, n_channels)
    for multi-channel.
    """
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


def compute_attenuation_db(freqs, prop, distance):
    """Compute the total dB attenuation for each frequency bin.

    Parameters
    ----------
    freqs : ndarray
        Frequency axis from the STFT (Hz).
    prop : AtmosphericPropagation
        Configured propagation model (source at origin, recording at 1 m).
    distance : float
        Target evaluation distance in metres.

    Returns
    -------
    ndarray
        Attenuation in dB (negative = quieter) for each frequency.
    """
    # DC bin gets no attenuation
    result = np.zeros_like(freqs)
    mask = freqs > 0
    if not np.any(mask):
        return result

    att = prop.attenuation_at_position(freqs[mask], (distance, 0.0, 0.0))
    result[mask] = att["total_dB"]
    return result


def process_channel(signal, sr, attenuation_db, nperseg):
    """Apply frequency-domain attenuation to a single channel via STFT/iSTFT.

    Parameters
    ----------
    signal : ndarray
        1-D time-domain signal.
    sr : int
        Sample rate in Hz.
    attenuation_db : ndarray
        dB gain to apply per frequency bin (same length as STFT freq axis).
    nperseg : int
        STFT segment length.

    Returns
    -------
    ndarray
        Attenuated time-domain signal.
    """
    noverlap = nperseg // 2
    f, t, Zxx = stft(signal, sr, window="hann", nperseg=nperseg, noverlap=noverlap)

    # Convert dB to linear gain and apply
    gain = 10.0 ** (attenuation_db / 20.0)
    Zxx *= gain[:, np.newaxis]

    _, reconstructed = istft(Zxx, sr, window="hann", nperseg=nperseg, noverlap=noverlap)

    # istft may return slightly different length; match original
    if len(reconstructed) < len(signal):
        reconstructed = np.pad(reconstructed, (0, len(signal) - len(reconstructed)))
    else:
        reconstructed = reconstructed[: len(signal)]

    return reconstructed


def main():
    args = parse_args()

    # Resolve output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_attenuated{ext}"
    else:
        output_path = args.output

    # Read input
    sr, data = read_wav(args.input)
    is_mono = data.ndim == 1
    n_channels = 1 if is_mono else data.shape[1]
    n_samples = data.shape[0]

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

    # STFT parameters
    nperseg = 2048
    noverlap = nperseg // 2

    # Compute frequency axis and attenuation curve
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    attenuation_db = compute_attenuation_db(freqs, prop, args.distance)

    # Print attenuation at key frequencies
    print("Attenuation at key frequencies:")
    for target_f in [125, 500, 1000, 4000]:
        idx = np.argmin(np.abs(freqs - target_f))
        print(f"  {target_f:5d} Hz: {attenuation_db[idx]:+.1f} dB")
    print()

    # Welch PSD of original (for informational summary)
    if is_mono:
        f_welch, psd_before = welch(data, sr, window="hann", nperseg=nperseg,
                                    noverlap=noverlap)
    else:
        f_welch, psd_before = welch(data[:, 0], sr, window="hann", nperseg=nperseg,
                                    noverlap=noverlap)

    # Process each channel
    if is_mono:
        output = process_channel(data, sr, attenuation_db, nperseg)
    else:
        channels = []
        for ch in range(n_channels):
            channels.append(process_channel(data[:, ch], sr, attenuation_db, nperseg))
        output = np.column_stack(channels)

    # Write output
    write_wav(output_path, sr, output)
    print(f"Output:      {output_path}")

    # PSD comparison at 1 kHz
    if is_mono:
        _, psd_after = welch(output, sr, window="hann", nperseg=nperseg,
                             noverlap=noverlap)
    else:
        _, psd_after = welch(output[:, 0], sr, window="hann", nperseg=nperseg,
                             noverlap=noverlap)

    idx_1k = np.argmin(np.abs(f_welch - 1000))
    if psd_before[idx_1k] > 0 and psd_after[idx_1k] > 0:
        measured = 10 * np.log10(psd_after[idx_1k] / psd_before[idx_1k])
        print(f"Measured PSD change at 1 kHz: {measured:+.1f} dB")


if __name__ == "__main__":
    main()
