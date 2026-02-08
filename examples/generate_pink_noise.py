#!/usr/bin/env python3
"""Generate a pink noise WAV file for testing propagation_filter.py.

Pink noise has equal energy per octave (power spectral density ~ 1/f),
making it ideal for validating frequency-dependent attenuation since
every octave band carries the same power before filtering.

Usage:
    python examples/generate_pink_noise.py [--output examples/pink_noise.wav]
    python examples/generate_pink_noise.py --duration 5 --sample-rate 48000
"""

import argparse

import numpy as np
from scipy.io import wavfile


def pink_noise(n_samples, rng=None):
    """Generate pink noise (1/f spectrum) via frequency-domain shaping.

    Parameters
    ----------
    n_samples : int
        Number of output samples.
    rng : numpy.random.Generator or None
        Random number generator (for reproducibility).

    Returns
    -------
    ndarray
        Pink noise signal normalised to peak amplitude ~1.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate white noise in the frequency domain
    n_fft = n_samples
    n_freqs = n_fft // 2 + 1
    # Random complex spectrum with unit magnitude, random phase
    phases = rng.uniform(0, 2 * np.pi, n_freqs)
    spectrum = np.exp(1j * phases)

    # Shape by 1/sqrt(f) to get 1/f power spectrum
    freqs = np.arange(n_freqs)
    freqs[0] = 1  # avoid division by zero at DC
    spectrum *= 1.0 / np.sqrt(freqs)
    spectrum[0] = 0  # zero DC component

    # Inverse FFT to time domain
    signal = np.fft.irfft(spectrum, n=n_fft)

    # Normalise to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    return signal


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pink noise WAV file for testing."
    )
    parser.add_argument(
        "--output",
        default="examples/pink_noise.wav",
        help="Output WAV path (default: examples/pink_noise.wav)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration in seconds (default: 3)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Peak amplitude 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    n_samples = int(args.duration * args.sample_rate)
    rng = np.random.default_rng(args.seed)

    signal = pink_noise(n_samples, rng) * args.amplitude

    data_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(args.output, args.sample_rate, data_int16)

    print(f"Generated: {args.output}")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Duration:    {args.duration:.1f} s")
    print(f"  Samples:     {n_samples}")
    print(f"  Amplitude:   {args.amplitude}")
    print(f"  Seed:        {args.seed}")


if __name__ == "__main__":
    main()
