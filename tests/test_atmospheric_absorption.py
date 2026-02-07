"""
Unit tests for AtmosphericPropagation against ISO 9613-1:1993
Table 1 — Pure-tone atmospheric-absorption attenuation coefficients,
in decibels per kilometre, at an air pressure of one standard
atmosphere (101,325 kPa).

Per Note 5 of Section 6.4, the table was computed using exact midband
frequencies:  f_m = 1000 * 10^(k/10)  Hz,  k ∈ {-13 … +10}
(one-third-octave band centres with bandwidth designator b = 1/3).
"""

import math
import warnings
import numpy as np
import pytest
from sound_propagation.atmospheric_absorption import AtmosphericPropagation
from sound_propagation.ground_attenuation import GroundAttenuation


# ── exact midband frequencies (Note 5, Eq. 6) ──────────────────────
def exact_freq(label: int) -> float:
    """Return the exact 1/3-octave midband frequency for a preferred
    frequency label (50–10 000 Hz)."""
    LABEL_TO_K = {
        50: -13, 63: -12, 80: -11, 100: -10, 125: -9, 160: -8,
        200: -7, 250: -6, 315: -5, 400: -4, 500: -3, 630: -2,
        800: -1, 1000: 0, 1250: 1, 1600: 2, 2000: 3, 2500: 4,
        3150: 5, 4000: 6, 5000: 7, 6300: 8, 8000: 9, 10000: 10,
    }
    k = LABEL_TO_K[label]
    return 1000.0 * 10.0 ** (k / 10.0)


# ── Table 1 reference data ─────────────────────────────────────────
# Each tuple: (temp_C, rh_pct, freq_label, alpha_dB_per_km)
# Values transcribed from the scanned ISO 9613-1:1993 Table 1.
#
# The table uses European decimal notation (comma = decimal point)
# and scientific notation like "1,15 × 10" meaning 11.5.

TABLE_1_DATA = [
    # ── (a) Air temperature: −20 °C ────────────────────────────────
    # Row 50 Hz
    (-20, 10, 50, 5.89e-1),
    (-20, 70, 50, 1.25e-1),
    (-20, 100, 50, 9.92e-2),
    # Row 125 Hz
    (-20, 10, 125, 1.20),
    (-20, 15, 125, 1.43),
    (-20, 30, 125, 1.28),
    (-20, 70, 125, 5.14e-1),
    (-20, 100, 125, 3.44e-1),
    # Row 1000 Hz
    (-20, 10, 1000, 1.65),
    (-20, 15, 1000, 2.34),
    (-20, 50, 1000, 9.14),
    (-20, 70, 1000, 1.15e1),
    (-20, 100, 1000, 1.11e1),
    # Row 4000 Hz
    (-20, 10, 4000, 3.86),
    (-20, 50, 4000, 1.32e1),
    (-20, 70, 4000, 2.02e1),
    (-20, 100, 4000, 3.14e1),
    # Row 8000 Hz
    (-20, 10, 8000, 1.09e1),
    (-20, 70, 8000, 2.78e1),
    (-20, 100, 8000, 4.11e1),

    # ── (b) Air temperature: −15 °C ────────────────────────────────
    (-15, 10, 1000, 2.67),
    (-15, 50, 1000, 1.32e1),
    (-15, 70, 1000, 1.21e1),
    (-15, 10, 4000, 4.92),
    (-15, 70, 4000, 3.79e1),

    # ── (c) Air temperature: −10 °C ────────────────────────────────
    (-10, 10, 125, 1.86),
    (-10, 70, 125, 3.32e-1),
    (-10, 10, 1000, 4.65),
    (-10, 50, 1000, 1.29e1),
    (-10, 70, 1000, 9.18),
    (-10, 100, 1000, 5.91),
    (-10, 10, 8000, 1.42e1),
    (-10, 70, 8000, 8.62e1),

    # ── (d) Air temperature: −5 °C ─────────────────────────────────
    (-5, 10, 1000, 8.29),
    (-5, 50, 1000, 9.68),
    (-5, 70, 1000, 6.38),
    (-5, 10, 4000, 1.11e1),
    (-5, 70, 4000, 6.50e1),

    # ── (e) Air temperature: 0 °C ──────────────────────────────────
    (0, 10, 125, 1.30),
    (0, 30, 125, 4.69e-1),
    (0, 70, 125, 3.90e-1),
    (0, 100, 125, 3.54e-1),
    # 1000 Hz — non-monotonic: peaks near 15 % RH at 0 °C
    (0, 10, 1000, 1.40e1),
    (0, 15, 1000, 1.83e1),
    (0, 20, 1000, 1.77e1),
    (0, 50, 1000, 6.83),
    (0, 70, 1000, 4.64),
    (0, 100, 1000, 3.37),
    (0, 10, 4000, 1.90e1),
    (0, 10, 8000, 2.64e1),

    # ── (f) Air temperature: 5 °C ──────────────────────────────────
    (5, 10, 1000, 2.00e1),
    (5, 50, 1000, 5.08),
    (5, 70, 1000, 3.80),

    # ── (g) Air temperature: 10 °C ─────────────────────────────────
    (10, 10, 125, 7.88e-1),
    (10, 70, 125, 4.11e-1),
    (10, 10, 1000, 2.16e1),
    (10, 20, 1000, 1.10e1),
    (10, 50, 1000, 4.26),
    (10, 70, 1000, 3.66),
    (10, 100, 1000, 3.55),
    (10, 10, 4000, 5.73e1),
    (10, 70, 4000, 3.28e1),
    (10, 100, 4000, 2.35e1),
    (10, 10, 8000, 6.94e1),

    # ── (h) Air temperature: 15 °C ─────────────────────────────────
    (15, 10, 1000, 1.84e1),
    (15, 70, 1000, 4.08),
    (15, 10, 4000, 8.73e1),
    (15, 70, 4000, 2.64e1),

    # ── (i) Air temperature: 20 °C ─────────────────────────────────
    (20, 10, 50, 2.70e-1),
    (20, 50, 50, 7.94e-2),
    (20, 100, 50, 4.03e-2),
    (20, 10, 100, 6.22e-1),
    (20, 50, 100, 2.94e-1),
    (20, 100, 100, 1.58e-1),
    (20, 10, 125, 7.76e-1),
    (20, 50, 125, 4.45e-1),
    (20, 70, 125, 3.39e-1),
    (20, 100, 125, 2.47e-1),
    (20, 10, 160, 9.65e-1),
    (20, 70, 160, 5.18e-1),
    (20, 10, 200, 1.22),
    (20, 70, 200, 7.76e-1),
    (20, 100, 200, 5.91e-1),
    (20, 10, 1000, 1.41e1),
    (20, 70, 1000, 4.98),
    (20, 100, 1000, 5.42),
    (20, 10, 4000, 1.09e2),
    (20, 100, 4000, 1.94e1),

    # ── (j) Air temperature: 25 °C ─────────────────────────────────
    (25, 10, 1000, 1.07e1),
    (25, 70, 1000, 6.19),
    (25, 100, 1000, 6.47),

    # ── (k) Air temperature: 30 °C ─────────────────────────────────
    (30, 10, 125, 9.58e-1),
    (30, 70, 125, 2.56e-1),
    (30, 10, 1000, 8.67),
    (30, 50, 1000, 7.03),
    (30, 70, 1000, 7.41),
    (30, 100, 1000, 7.17),
    (30, 10, 4000, 9.60e1),
    (30, 70, 4000, 2.31e1),
    (30, 100, 4000, 2.40e1),

    # ── (l) Air temperature: 35 °C ─────────────────────────────────
    (35, 10, 1000, 7.71),
    (35, 70, 1000, 8.30),

    # ── (m) Air temperature: 40 °C ─────────────────────────────────
    (40, 10, 1000, 7.68),
    (40, 70, 1000, 8.66),

    # ── (p) Air temperature: 50 °C ─────────────────────────────────
    (50, 10, 100, 5.84e-1),
    (50, 70, 100, 9.13e-2),
    (50, 10, 1000, 9.95),
    (50, 50, 1000, 1.00e1),
    (50, 70, 1000, 8.03),
    (50, 100, 1000, 6.05),
    (50, 10, 4000, 4.64e1),
    (50, 70, 4000, 4.82e1),
    (50, 10, 8000, 1.55e2),
    (50, 70, 8000, 8.03e1),
]


@pytest.mark.parametrize(
    "temp_c, rh_pct, freq_label, expected_dB_km",
    TABLE_1_DATA,
    ids=[
        f"T={t:+d}C_RH={h}%_f={f}Hz"
        for t, h, f, _ in TABLE_1_DATA
    ],
)
def test_absorption_coefficient_against_table1(
    temp_c, rh_pct, freq_label, expected_dB_km
):
    """Compare absorption_coefficient() to ISO 9613-1 Table 1 values."""
    prop = AtmosphericPropagation(
        temperature_c=temp_c,
        relative_humidity_pct=rh_pct,
        source=(0, 0, 0),
        recording=(100, 0, 0),
    )
    freq = exact_freq(freq_label)
    computed_dB_m = prop.absorption_coefficient(freq)
    computed_dB_km = computed_dB_m * 1000.0

    # Table values have 3 significant figures.
    # Use 1 % relative tolerance for values ≥ 1 dB/km,
    # 0.01 dB/km absolute tolerance for smaller values.
    if expected_dB_km >= 1.0:
        assert computed_dB_km == pytest.approx(expected_dB_km, rel=0.01), (
            f"expected {expected_dB_km:.4f} dB/km, "
            f"got {computed_dB_km:.4f} dB/km"
        )
    else:
        assert computed_dB_km == pytest.approx(expected_dB_km, abs=0.01), (
            f"expected {expected_dB_km:.6f} dB/km, "
            f"got {computed_dB_km:.6f} dB/km"
        )


# ── Tests for attenuation_at_offset ────────────────────────────────

class TestAttenuationAtOffset:
    """Verify the propagation method combining absorption + spreading."""

    def _make_prop(self, temp_c=20.0, rh=70.0, d=200.0):
        return AtmosphericPropagation(
            temperature_c=temp_c,
            relative_humidity_pct=rh,
            source=(0, 0, 0),
            recording=(d, 0, 0),
        )

    def test_zero_offset_is_zero(self):
        """At the recording position itself, change must be 0 dB."""
        prop = self._make_prop()
        result = prop.attenuation_at_offset(1000.0, 0.0)
        assert result["total_dB"] == pytest.approx(0.0, abs=1e-12)
        assert result["atmospheric_dB"] == pytest.approx(0.0, abs=1e-12)
        assert result["geometric_dB"] == pytest.approx(0.0, abs=1e-12)

    def test_positive_offset_attenuates(self):
        """Moving farther from source → total_dB < 0 (quieter)."""
        prop = self._make_prop()
        result = prop.attenuation_at_offset(1000.0, 50.0)
        assert result["total_dB"] < 0.0
        assert result["atmospheric_dB"] < 0.0
        assert result["geometric_dB"] < 0.0
        assert result["distance_to_source"] == pytest.approx(250.0)

    def test_negative_offset_amplifies(self):
        """Moving closer to source → total_dB > 0 (louder)."""
        prop = self._make_prop()
        result = prop.attenuation_at_offset(1000.0, -50.0)
        assert result["total_dB"] > 0.0
        assert result["atmospheric_dB"] > 0.0
        assert result["geometric_dB"] > 0.0
        assert result["distance_to_source"] == pytest.approx(150.0)

    def test_symmetry_of_components(self):
        """atmospheric + geometric must equal total."""
        prop = self._make_prop()
        for offset in [-80.0, -20.0, 0.0, 20.0, 80.0]:
            r = prop.attenuation_at_offset(1000.0, offset)
            assert r["total_dB"] == pytest.approx(
                r["atmospheric_dB"] + r["geometric_dB"], abs=1e-12
            )

    def test_geometric_follows_inverse_square(self):
        """Geometric component must equal ∓20·lg(d_eval/d_rec)."""
        prop = self._make_prop(d=100.0)
        r = prop.attenuation_at_offset(1000.0, 100.0)
        expected_geo = -20.0 * math.log10(200.0 / 100.0)  # −6.021 dB
        assert r["geometric_dB"] == pytest.approx(expected_geo, abs=1e-10)

    def test_atmospheric_scales_with_distance(self):
        """Atmospheric component must scale linearly with offset."""
        prop = self._make_prop()
        r1 = prop.attenuation_at_offset(1000.0, 10.0)
        r2 = prop.attenuation_at_offset(1000.0, 20.0)
        assert r2["atmospheric_dB"] == pytest.approx(
            2.0 * r1["atmospheric_dB"], abs=1e-10
        )

    def test_offset_behind_source_raises(self):
        """Evaluation point at or behind source must raise ValueError."""
        prop = self._make_prop(d=100.0)
        with pytest.raises(ValueError, match="at or behind the source"):
            prop.attenuation_at_offset(1000.0, -100.0)
        with pytest.raises(ValueError, match="at or behind the source"):
            prop.attenuation_at_offset(1000.0, -150.0)

    def test_3d_distance(self):
        """Source–recording distance is Euclidean in 3-D."""
        prop = AtmosphericPropagation(
            temperature_c=20.0,
            relative_humidity_pct=50.0,
            source=(0, 0, 0),
            recording=(30, 40, 0),  # distance = 50 m
        )
        assert prop._d_rec == pytest.approx(50.0)
        r = prop.attenuation_at_offset(1000.0, 0.0)
        assert r["distance_to_source"] == pytest.approx(50.0)


# ── Edge-case / validation tests ───────────────────────────────────

class TestValidation:
    def test_zero_humidity_raises(self):
        with pytest.raises(ValueError):
            AtmosphericPropagation(20.0, 0.0)

    def test_negative_frequency_raises(self):
        prop = AtmosphericPropagation(20.0, 50.0)
        with pytest.raises(ValueError):
            prop.absorption_coefficient(-100.0)

    def test_same_position_raises(self):
        with pytest.raises(ValueError):
            AtmosphericPropagation(20.0, 50.0, source=(0, 0, 0), recording=(0, 0, 0))

    def test_negative_frequency_array_raises(self):
        prop = AtmosphericPropagation(20.0, 50.0)
        with pytest.raises(ValueError):
            prop.absorption_coefficient(np.array([100.0, -50.0, 1000.0]))

    def test_repr(self):
        prop = AtmosphericPropagation(20.0, 70.0)
        r = repr(prop)
        assert "20" in r
        assert "70" in r


# ── Vectorisation tests ────────────────────────────────────────────

class TestVectorisation:
    """Verify that scalar and array inputs produce consistent results."""

    def _make_prop(self):
        return AtmosphericPropagation(
            temperature_c=20.0,
            relative_humidity_pct=70.0,
            source=(0, 0, 0),
            recording=(200, 0, 0),
        )

    def test_scalar_returns_scalar(self):
        """A plain float input still returns a numpy scalar (0-d)."""
        prop = self._make_prop()
        result = prop.absorption_coefficient(1000.0)
        assert np.ndim(result) == 0

    def test_array_returns_array(self):
        """An array input returns an array of the same shape."""
        prop = self._make_prop()
        freqs = np.array([125.0, 250.0, 500.0, 1000.0, 4000.0])
        result = prop.absorption_coefficient(freqs)
        assert isinstance(result, np.ndarray)
        assert result.shape == freqs.shape

    def test_array_matches_scalar_loop(self):
        """Each element of the array result must match the scalar call."""
        prop = self._make_prop()
        freqs = np.array([50.0, 125.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        vectorised = prop.absorption_coefficient(freqs)
        for i, f in enumerate(freqs):
            scalar = prop.absorption_coefficient(float(f))
            assert float(vectorised[i]) == pytest.approx(float(scalar), rel=1e-14)

    def test_list_input_accepted(self):
        """A plain Python list should also work via np.asarray."""
        prop = self._make_prop()
        result = prop.absorption_coefficient([500.0, 1000.0])
        assert result.shape == (2,)

    def test_2d_array_shape_preserved(self):
        """A 2-D input should produce a 2-D output."""
        prop = self._make_prop()
        freqs = np.array([[125.0, 250.0], [500.0, 1000.0]])
        result = prop.absorption_coefficient(freqs)
        assert result.shape == (2, 2)

    def test_attenuation_at_offset_vectorised(self):
        """attenuation_at_offset with array frequencies."""
        prop = self._make_prop()
        freqs = np.array([500.0, 1000.0, 4000.0])
        result = prop.attenuation_at_offset(freqs, 50.0)
        assert result["total_dB"].shape == (3,)
        assert result["atmospheric_dB"].shape == (3,)
        # geometric_dB is frequency-independent, stays scalar
        assert np.ndim(result["geometric_dB"]) == 0

        # verify each element matches scalar call
        for i, f in enumerate(freqs):
            scalar = prop.attenuation_at_offset(float(f), 50.0)
            assert float(result["total_dB"][i]) == pytest.approx(
                float(scalar["total_dB"]), rel=1e-14
            )

    def test_monotonic_with_frequency(self):
        """Higher frequency → higher absorption coefficient."""
        prop = self._make_prop()
        freqs = np.array([100.0, 500.0, 1000.0, 4000.0, 8000.0])
        alphas = prop.absorption_coefficient(freqs)
        assert np.all(np.diff(alphas) > 0)


# ── attenuation_at_position tests ──────────────────────────────────

class TestAttenuationAtPosition:
    """Verify attenuation_at_position for arbitrary 3-D evaluation points."""

    def _make_prop(self, src=(0, 0, 0), rec=(200, 0, 0)):
        return AtmosphericPropagation(
            temperature_c=20.0,
            relative_humidity_pct=70.0,
            source=src,
            recording=rec,
        )

    def test_at_recording_is_zero(self):
        """Evaluating at the recording position gives 0 dB change."""
        prop = self._make_prop()
        r = prop.attenuation_at_position(1000.0, (200, 0, 0))
        assert r["total_dB"] == pytest.approx(0.0, abs=1e-12)
        assert r["atmospheric_dB"] == pytest.approx(0.0, abs=1e-12)
        assert r["geometric_dB"] == pytest.approx(0.0, abs=1e-12)
        assert r["distance_to_source"] == pytest.approx(200.0)

    def test_on_axis_matches_offset(self):
        """A point on the source→recording axis must match attenuation_at_offset."""
        prop = self._make_prop()
        # 50 m past recording along x-axis = offset +50
        r_pos = prop.attenuation_at_position(1000.0, (250, 0, 0))
        r_off = prop.attenuation_at_offset(1000.0, 50.0)
        assert r_pos["total_dB"] == pytest.approx(r_off["total_dB"], abs=1e-12)
        assert r_pos["atmospheric_dB"] == pytest.approx(r_off["atmospheric_dB"], abs=1e-12)
        assert r_pos["geometric_dB"] == pytest.approx(r_off["geometric_dB"], abs=1e-12)

        # 50 m before recording = offset -50
        r_pos = prop.attenuation_at_position(1000.0, (150, 0, 0))
        r_off = prop.attenuation_at_offset(1000.0, -50.0)
        assert r_pos["total_dB"] == pytest.approx(r_off["total_dB"], abs=1e-12)

    def test_off_axis_uses_euclidean_distance(self):
        """An off-axis point uses straight-line distance to source."""
        prop = self._make_prop(src=(0, 0, 0), rec=(100, 0, 0))
        # Point at (0, 100, 0) — distance 100 from source, same as recording
        r = prop.attenuation_at_position(1000.0, (0, 100, 0))
        assert r["distance_to_source"] == pytest.approx(100.0)
        # Same distance as recording → 0 dB geometric, 0 dB atmospheric
        assert r["total_dB"] == pytest.approx(0.0, abs=1e-12)

    def test_farther_off_axis_attenuates(self):
        """A point farther from source than the recording should attenuate."""
        prop = self._make_prop(src=(0, 0, 0), rec=(100, 0, 0))
        # Point at (200, 200, 0) — distance ~283 m, farther than 100 m
        r = prop.attenuation_at_position(1000.0, (200, 200, 0))
        assert r["distance_to_source"] > 100.0
        assert r["total_dB"] < 0.0

    def test_closer_off_axis_amplifies(self):
        """A point closer to source than the recording should amplify."""
        prop = self._make_prop(src=(0, 0, 0), rec=(100, 0, 0))
        # Point at (10, 10, 0) — distance ~14.1 m, closer than 100 m
        r = prop.attenuation_at_position(1000.0, (10, 10, 0))
        assert r["distance_to_source"] < 100.0
        assert r["total_dB"] > 0.0

    def test_3d_distance(self):
        """Full 3-D Euclidean distance is used."""
        prop = self._make_prop(src=(0, 0, 0), rec=(100, 0, 0))
        r = prop.attenuation_at_position(1000.0, (30, 40, 0))
        assert r["distance_to_source"] == pytest.approx(50.0)
        r = prop.attenuation_at_position(1000.0, (10, 20, 20))
        assert r["distance_to_source"] == pytest.approx(30.0)

    def test_at_source_raises(self):
        """Evaluating at the source position must raise ValueError."""
        prop = self._make_prop()
        with pytest.raises(ValueError, match="coincides with the source"):
            prop.attenuation_at_position(1000.0, (0, 0, 0))

    def test_vectorised_frequency(self):
        """Array frequencies work with attenuation_at_position."""
        prop = self._make_prop()
        freqs = np.array([500.0, 1000.0, 4000.0])
        r = prop.attenuation_at_position(freqs, (250, 0, 0))
        assert r["total_dB"].shape == (3,)
        assert r["atmospheric_dB"].shape == (3,)
        # geometric_dB is frequency-independent
        assert np.ndim(r["geometric_dB"]) == 0

    def test_symmetry_of_components(self):
        """atmospheric + geometric must equal total at any position."""
        prop = self._make_prop()
        for pos in [(50, 0, 0), (200, 0, 0), (0, 300, 0), (100, 100, 100)]:
            r = prop.attenuation_at_position(1000.0, pos)
            assert r["total_dB"] == pytest.approx(
                r["atmospheric_dB"] + r["geometric_dB"], abs=1e-12
            )


# ── Pressure None default tests ──────────────────────────────────

class TestPressureNone:
    """Verify that pressure_kpa=None defaults to 101.325 kPa."""

    def test_none_matches_explicit(self):
        """None pressure produces identical results to explicit 101.325."""
        prop_none = AtmosphericPropagation(20.0, 50.0, pressure_kpa=None)
        prop_explicit = AtmosphericPropagation(20.0, 50.0, pressure_kpa=101.325)
        freqs = np.array([125.0, 1000.0, 4000.0])
        alpha_none = prop_none.absorption_coefficient(freqs)
        alpha_explicit = prop_explicit.absorption_coefficient(freqs)
        np.testing.assert_array_equal(alpha_none, alpha_explicit)

    def test_none_is_default(self):
        """Omitting pressure_kpa entirely uses 101.325."""
        prop_default = AtmosphericPropagation(20.0, 50.0)
        prop_none = AtmosphericPropagation(20.0, 50.0, pressure_kpa=None)
        assert prop_default.pressure_kpa == prop_none.pressure_kpa == 101.325

    def test_none_stores_numeric(self):
        """pressure_kpa attribute is float, not None."""
        prop = AtmosphericPropagation(20.0, 50.0, pressure_kpa=None)
        assert prop.pressure_kpa == 101.325


# ── ISO §7 accuracy warning tests ────────────────────────────────

class TestAccuracyWarnings:
    """Verify warnings for conditions outside ISO 9613-1 §7 bounds."""

    def test_no_warning_in_range(self):
        """No warning for typical conditions well within §7.1 bounds."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AtmosphericPropagation(20.0, 50.0)

    def test_cold_temperature_warns(self):
        """Temperature below -20 °C triggers a warning."""
        with pytest.warns(UserWarning, match="Temperature.*outside"):
            AtmosphericPropagation(-25.0, 50.0)

    def test_hot_temperature_warns(self):
        """Temperature above +50 °C triggers a warning."""
        with pytest.warns(UserWarning, match="Temperature.*outside"):
            AtmosphericPropagation(55.0, 50.0)

    def test_boundary_temperature_no_warning(self):
        """Exact boundary temperatures (-20, +50) should not warn for temp.
        Use humidity values that keep h within [0.05, 5]%."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # At 20°C, 50% RH → h well within range, no warnings at all
            AtmosphericPropagation(20.0, 50.0)
            # At 50°C, 15% RH → h ≈ 1.83%, within range
            AtmosphericPropagation(50.0, 15.0)

    def test_low_humidity_section72_warns(self):
        """Very low RH producing h in [0.005, 0.05)% triggers §7.2 warning."""
        # At 20°C, RH ~0.5% gives h ≈ 0.012% (in [0.005, 0.05)%)
        with pytest.warns(UserWarning, match="§7.2"):
            AtmosphericPropagation(20.0, 0.5)

    def test_high_humidity_section72_warns(self):
        """Very high RH producing h > 5% triggers §7.2 warning."""
        # At 50°C, RH 100% gives h well above 5%
        with pytest.warns(UserWarning, match="§7.2"):
            AtmosphericPropagation(50.0, 100.0)

    def test_very_low_humidity_section73_warns(self):
        """Extremely low RH producing h < 0.005% triggers §7.3 warning."""
        # At -20°C, RH ~0.5% gives very low h
        # We need to find conditions where h < 0.005%
        # At -20°C, psat_ratio is very small, so low RH will do it
        with pytest.warns(UserWarning, match="§7.3"):
            AtmosphericPropagation(-20.0, 0.5)


# ── Octave-band convenience method tests ──────────────────────────

class TestOctaveBand:
    """Verify absorption_coefficient_octave convenience method."""

    def _make_prop(self):
        return AtmosphericPropagation(
            temperature_c=20.0,
            relative_humidity_pct=70.0,
            source=(0, 0, 0),
            recording=(200, 0, 0),
        )

    def test_default_bands(self):
        """Default call returns 8-element array for standard octave bands."""
        prop = self._make_prop()
        result = prop.absorption_coefficient_octave()
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)
        assert np.all(result > 0)

    def test_default_matches_explicit_bands(self):
        """Default call matches explicit OCTAVE_BANDS call."""
        prop = self._make_prop()
        default = prop.absorption_coefficient_octave()
        explicit = prop.absorption_coefficient_octave(
            np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        )
        np.testing.assert_array_equal(default, explicit)

    def test_custom_bands(self):
        """Custom centre frequencies are accepted."""
        prop = self._make_prop()
        custom = np.array([250, 500, 1000])
        result = prop.absorption_coefficient_octave(custom)
        assert result.shape == (3,)

    def test_consistent_with_absorption_coefficient(self):
        """Each element matches a scalar absorption_coefficient call."""
        prop = self._make_prop()
        octave = prop.absorption_coefficient_octave()
        bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        for i, f in enumerate(bands):
            scalar = prop.absorption_coefficient(float(f))
            assert float(octave[i]) == pytest.approx(float(scalar), rel=1e-14)

    def test_monotonic(self):
        """Higher octave bands have higher absorption at 20 °C / 70% RH."""
        prop = self._make_prop()
        result = prop.absorption_coefficient_octave()
        assert np.all(np.diff(result) > 0)

    def test_octave_bands_class_constant(self):
        """OCTAVE_BANDS is accessible as a class attribute."""
        expected = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        np.testing.assert_array_equal(AtmosphericPropagation.OCTAVE_BANDS, expected)

    def test_list_input(self):
        """A plain list is accepted as center_frequencies."""
        prop = self._make_prop()
        result = prop.absorption_coefficient_octave([500, 1000, 2000])
        assert result.shape == (3,)


# ── total_attenuation tests ──────────────────────────────────────

class TestTotalAttenuation:
    """Verify total_attenuation combining atmospheric + geometric + ground."""

    def _make_prop(self, src=(0, 0, 0), rec=(200, 0, 0)):
        return AtmosphericPropagation(20.0, 70.0, source=src, recording=rec)

    def _make_ground(self, hs=1.0, hr=2.0, dp=200.0):
        return GroundAttenuation(
            source_height=hs, receiver_height=hr, distance=dp,
            G_source=1.0, G_receiver=1.0, G_middle=1.0,
        )

    def test_without_ground_matches_attenuation_at_position(self):
        """Without ground, result equals attenuation_at_position."""
        prop = self._make_prop()
        pos = (250, 0, 0)
        expected = prop.attenuation_at_position(1000.0, pos)
        result = prop.total_attenuation(1000.0, pos)
        assert result["total_dB"] == pytest.approx(expected["total_dB"], abs=1e-12)
        assert result["atmospheric_dB"] == pytest.approx(expected["atmospheric_dB"], abs=1e-12)
        assert result["geometric_dB"] == pytest.approx(expected["geometric_dB"], abs=1e-12)
        assert result["distance_to_source"] == pytest.approx(expected["distance_to_source"])

    def test_with_ground_adds_ground_component(self):
        """With ground, total_dB includes ground_dB."""
        prop = self._make_prop()
        ground = self._make_ground()
        pos = (250, 0, 0)
        without = prop.total_attenuation(1000.0, pos)
        with_ground = prop.total_attenuation(1000.0, pos, ground=ground)
        agr = ground.ground_attenuation(1000.0)
        assert with_ground["total_dB"] == pytest.approx(
            without["total_dB"] + agr, abs=1e-12
        )

    def test_ground_dB_key_absent_without_ground(self):
        """ground_dB key should not appear when ground=None."""
        prop = self._make_prop()
        result = prop.total_attenuation(1000.0, (250, 0, 0))
        assert "ground_dB" not in result

    def test_ground_dB_key_present_with_ground(self):
        """ground_dB key appears when ground is provided."""
        prop = self._make_prop()
        ground = self._make_ground()
        result = prop.total_attenuation(1000.0, (250, 0, 0), ground=ground)
        assert "ground_dB" in result

    def test_total_equals_sum_of_components(self):
        """total_dB = atmospheric_dB + geometric_dB + ground_dB."""
        prop = self._make_prop()
        ground = self._make_ground()
        result = prop.total_attenuation(1000.0, (250, 0, 0), ground=ground)
        expected = result["atmospheric_dB"] + result["geometric_dB"] + result["ground_dB"]
        assert result["total_dB"] == pytest.approx(expected, abs=1e-12)

    def test_non_octave_frequency_with_ground_raises(self):
        """Passing non-octave frequency when ground is set raises ValueError."""
        prop = self._make_prop()
        ground = self._make_ground()
        with pytest.raises(ValueError):
            prop.total_attenuation(777.0, (250, 0, 0), ground=ground)

    def test_vectorised_frequencies(self):
        """Array of octave-band frequencies works with ground."""
        prop = self._make_prop()
        ground = self._make_ground()
        freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
        result = prop.total_attenuation(freqs, (250, 0, 0), ground=ground)
        assert result["total_dB"].shape == (8,)
        assert result["ground_dB"].shape == (8,)

    def test_distance_to_source_present(self):
        """distance_to_source is always included."""
        prop = self._make_prop()
        result = prop.total_attenuation(1000.0, (250, 0, 0))
        assert "distance_to_source" in result
        assert result["distance_to_source"] == pytest.approx(250.0)
