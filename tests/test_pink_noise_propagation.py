"""
Pink noise propagation tests — verify that both STFT and FIR filtering
reproduce the ISO 9613 attenuation predicted by AtmosphericPropagation.

Two scenarios share a common geometry:
  - Source at the origin
  - Listener (recording) at 200 m

Test A evaluates at 100 m (closer to the source → less attenuation than
the listener would experience).
Test B evaluates at 500 m (farther from the source → more attenuation
than the listener would experience).

Both are run with each filter method (STFT from ``propagation_filter.py``
and FIR from ``propagation_fir_filter.py``) and compared against the
theoretical prediction from ``AtmosphericPropagation.attenuation_at_position``.
"""

import sys
import os

import numpy as np
import pytest
from custom_noise import pink_noise
from scipy.signal import welch, fftconvolve

# Make the examples directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "examples"))
from propagation_filter import compute_attenuation_db, process_channel
from propagation_fir_filter import design_propagation_filter

from sound_propagation import AtmosphericPropagation

# ── shared geometry and environment ────────────────────────────────

SAMPLE_RATE = 44100
DURATION = 5.0
NPERSEG = 4096
TEMPERATURE_C = 20.0
HUMIDITY_PCT = 70.0
PRESSURE_KPA = 101.325

SOURCE = (0.0, 0.0, 0.0)
LISTENER = (200.0, 0.0, 0.0)  # shared listening point at 200 m

EVAL_CLOSER = (100.0, 0.0, 0.0)   # 100 m — closer to source
EVAL_FARTHER = (500.0, 0.0, 0.0)  # 500 m — farther from source

# Octave band edges for PSD integration (start at 125 Hz — the 63 Hz band
# is excluded because STFT edge effects at very low frequencies relative
# to the window length make the measurement unreliable)
OCTAVE_BANDS = [
    (125, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]

# FIR filter tap count — same as sample rate for ~1 Hz resolution
FIR_NUMTAPS = SAMPLE_RATE | 1  # ensure odd


# ── helpers ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pink_signal():
    """Generate a reproducible pink noise signal (float64, normalised)."""
    return pink_noise(
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        random_seed=42,
        dtype="float64",
    )


@pytest.fixture(scope="module")
def original_band_power(pink_signal):
    """PSD power per octave band of the unfiltered pink noise."""
    return _octave_band_power(pink_signal, SAMPLE_RATE, NPERSEG)


def _make_prop():
    """Create propagation model with the shared listener position."""
    return AtmosphericPropagation(
        temperature_c=TEMPERATURE_C,
        relative_humidity_pct=HUMIDITY_PCT,
        pressure_kpa=PRESSURE_KPA,
        source=SOURCE,
        recording=LISTENER,
    )


def _octave_band_power(signal, sr, nperseg):
    """Return PSD power integrated per octave band."""
    f, psd = welch(signal, sr, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    df = f[1] - f[0]
    powers = []
    for lo, hi in OCTAVE_BANDS:
        mask = (f >= lo) & (f < hi)
        powers.append(np.sum(psd[mask]) * df)
    return np.array(powers)


def _apply_stft_filter(signal, eval_distance):
    """Run the STFT filter pipeline for a given evaluation distance."""
    prop = _make_prop()
    freqs = np.fft.rfftfreq(NPERSEG, d=1.0 / SAMPLE_RATE)
    att_db = compute_attenuation_db(freqs, prop, eval_distance)
    return process_channel(signal, SAMPLE_RATE, att_db, NPERSEG)


def _apply_fir_filter(signal, eval_distance):
    """Run the FIR filter pipeline for a given evaluation distance."""
    prop = _make_prop()
    fir, _, _ = design_propagation_filter(
        SAMPLE_RATE, FIR_NUMTAPS, prop, eval_distance
    )
    filtered = fftconvolve(signal, fir, mode="same")
    return filtered


def _predicted_band_attenuation_db(eval_pos, freq_grid):
    """Theoretical attenuation (dB) per octave band at eval_pos relative
    to the listener.

    Computed by averaging the linear power gain across all frequency bins
    within each band, then converting back to dB.  The freq_grid should
    match the filter method being validated (STFT bins or FIR design points).
    """
    prop = _make_prop()
    att_db = compute_attenuation_db(freq_grid, prop, eval_pos[0])

    predicted = []
    for lo, hi in OCTAVE_BANDS:
        mask = (freq_grid >= lo) & (freq_grid < hi)
        band_gain_linear = np.mean(10.0 ** (att_db[mask] / 10.0))
        predicted.append(10.0 * np.log10(band_gain_linear))
    return np.array(predicted)


def _stft_freq_grid():
    """Frequency grid matching the STFT filter bins."""
    return np.fft.rfftfreq(NPERSEG, d=1.0 / SAMPLE_RATE)


def _fir_freq_grid():
    """Frequency grid matching the FIR filter design points."""
    n_points = FIR_NUMTAPS // 2 + 1
    return np.linspace(0, SAMPLE_RATE / 2.0, n_points)


# ── parametrised filter methods ────────────────────────────────────

FILTER_METHODS = [
    pytest.param("stft", id="STFT"),
    pytest.param("fir", id="FIR"),
]


def _apply_filter(method, signal, eval_distance):
    """Dispatch to the appropriate filter pipeline."""
    if method == "stft":
        return _apply_stft_filter(signal, eval_distance)
    else:
        return _apply_fir_filter(signal, eval_distance)


def _freq_grid_for(method):
    """Return the frequency grid matching a filter method."""
    if method == "stft":
        return _stft_freq_grid()
    else:
        return _fir_freq_grid()


# ── Test A: evaluation closer to source (100 m) ───────────────────

class TestCloserToSource:
    """Evaluate at 100 m — halfway between source and listener at 200 m.

    Expect amplification relative to the listener (positive dB change)
    because the evaluation point is closer and suffers less atmospheric
    absorption and geometric spreading.
    """

    @pytest.fixture(scope="class", params=FILTER_METHODS)
    def band_power_closer(self, request, pink_signal):
        method = request.param
        filtered = _apply_filter(method, pink_signal, 100.0)
        power = _octave_band_power(filtered, SAMPLE_RATE, NPERSEG)
        freq_grid = _freq_grid_for(method)
        return power, freq_grid

    def test_closer_is_louder(self, band_power_closer, original_band_power):
        """Signal filtered at 100 m should be louder than at 200 m
        (the listener distance, where the filter is identity)."""
        power, _ = band_power_closer
        for i, (lo, hi) in enumerate(OCTAVE_BANDS):
            assert power[i] > original_band_power[i], (
                f"Band {lo}-{hi} Hz: closer point should be louder than listener"
            )

    def test_psd_change_matches_prediction(self, band_power_closer,
                                           original_band_power):
        """Measured PSD change per octave band matches ISO 9613 prediction."""
        power, freq_grid = band_power_closer
        measured_db = 10.0 * np.log10(power / original_band_power)
        predicted_db = _predicted_band_attenuation_db(EVAL_CLOSER, freq_grid)

        for i, (lo, hi) in enumerate(OCTAVE_BANDS):
            assert measured_db[i] == pytest.approx(predicted_db[i], abs=1.0), (
                f"Band {lo}-{hi} Hz: measured {measured_db[i]:.2f} dB, "
                f"predicted {predicted_db[i]:.2f} dB"
            )

    def test_all_bands_positive_gain(self, band_power_closer, original_band_power):
        """Every band should show positive gain (louder) when closer."""
        power, _ = band_power_closer
        measured_db = 10.0 * np.log10(power / original_band_power)
        for i, (lo, hi) in enumerate(OCTAVE_BANDS):
            assert measured_db[i] > 0, (
                f"Band {lo}-{hi} Hz: expected positive gain at closer point"
            )

    def test_geometric_dominates_at_low_freq(self):
        """At low frequencies, atmospheric absorption is small — most of
        the gain comes from geometric spreading (6 dB for halving distance).

        This is a model-level test, independent of filter method.
        """
        prop = _make_prop()
        result = prop.attenuation_at_position(125.0, EVAL_CLOSER)
        # Geometric: 20*log10(100/200) = -6.02 dB → eval is closer → +6.02 dB
        assert result["geometric_dB"] == pytest.approx(6.02, abs=0.1)
        # Atmospheric contribution at 125 Hz over 100 m difference is small
        assert abs(result["atmospheric_dB"]) < 1.0


# ── Test B: evaluation farther from source (500 m) ────────────────

class TestFartherFromSource:
    """Evaluate at 500 m — well beyond the listener at 200 m.

    Expect attenuation relative to the listener (negative dB change)
    with high frequencies losing more energy than low frequencies.
    """

    @pytest.fixture(scope="class", params=FILTER_METHODS)
    def band_power_farther(self, request, pink_signal):
        method = request.param
        filtered = _apply_filter(method, pink_signal, 500.0)
        power = _octave_band_power(filtered, SAMPLE_RATE, NPERSEG)
        freq_grid = _freq_grid_for(method)
        return power, freq_grid

    def test_farther_is_quieter(self, band_power_farther, original_band_power):
        """Signal filtered at 500 m should be quieter than at the listener."""
        power, _ = band_power_farther
        for i, (lo, hi) in enumerate(OCTAVE_BANDS):
            assert power[i] < original_band_power[i], (
                f"Band {lo}-{hi} Hz: farther point should be quieter than listener"
            )

    def test_psd_change_matches_prediction(self, band_power_farther,
                                           original_band_power):
        """Measured PSD change per octave band matches ISO 9613 prediction."""
        power, freq_grid = band_power_farther
        measured_db = 10.0 * np.log10(power / original_band_power)
        predicted_db = _predicted_band_attenuation_db(EVAL_FARTHER, freq_grid)

        for i, (lo, hi) in enumerate(OCTAVE_BANDS):
            assert measured_db[i] == pytest.approx(predicted_db[i], abs=1.0), (
                f"Band {lo}-{hi} Hz: measured {measured_db[i]:.2f} dB, "
                f"predicted {predicted_db[i]:.2f} dB"
            )

    def test_high_freq_attenuates_more(self, band_power_farther,
                                       original_band_power):
        """Higher octave bands should lose more energy than lower ones."""
        power, _ = band_power_farther
        measured_db = 10.0 * np.log10(power / original_band_power)
        for i in range(1, len(measured_db)):
            lo, hi = OCTAVE_BANDS[i]
            assert measured_db[i] <= measured_db[i - 1] + 0.1, (
                f"Band {lo}-{hi} Hz ({measured_db[i]:.2f} dB) should attenuate "
                f"at least as much as band {OCTAVE_BANDS[i-1][0]}-"
                f"{OCTAVE_BANDS[i-1][1]} Hz ({measured_db[i-1]:.2f} dB)"
            )

    def test_all_bands_negative_gain(self, band_power_farther, original_band_power):
        """Every band should show negative gain (quieter) when farther."""
        power, _ = band_power_farther
        measured_db = 10.0 * np.log10(power / original_band_power)
        for i, (lo, hi) in enumerate(OCTAVE_BANDS):
            assert measured_db[i] < 0, (
                f"Band {lo}-{hi} Hz: expected negative gain at farther point"
            )

    def test_atmospheric_contribution_grows_with_freq(self):
        """The atmospheric absorption component should increase with frequency.

        This is a model-level test, independent of filter method.
        """
        prop = _make_prop()
        low = prop.attenuation_at_position(250.0, EVAL_FARTHER)
        high = prop.attenuation_at_position(4000.0, EVAL_FARTHER)
        # More negative atmospheric_dB = more absorption
        assert high["atmospheric_dB"] < low["atmospheric_dB"]
