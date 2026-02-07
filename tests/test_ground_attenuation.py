"""Tests for ISO 9613-2:1996 §7.3.1 ground attenuation (Agr)."""

import math
import numpy as np
import pytest

from sound_propagation.ground_attenuation import GroundAttenuation

OCTAVE_BANDS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


# ------------------------------------------------------------------
# 1. Hard ground (G=0 everywhere)
# ------------------------------------------------------------------

class TestHardGround:
    """With G=0 on all regions, As and Ar = -1.5 for all bands,
    and Am = -3q for 63 Hz, 0 for others."""

    @pytest.fixture
    def ga(self):
        return GroundAttenuation(
            source_height=2.0, receiver_height=4.0, distance=200.0,
            G_source=0.0, G_receiver=0.0, G_middle=0.0,
        )

    @pytest.mark.parametrize("freq", OCTAVE_BANDS)
    def test_As_and_Ar_minus_1_5(self, ga, freq):
        """For G=0: As = Ar = -1.5 at all frequencies."""
        As = ga._source_or_receiver_atten(freq, 0.0, ga.hs)
        Ar = ga._source_or_receiver_atten(freq, 0.0, ga.hr)
        assert As == pytest.approx(-1.5)
        assert Ar == pytest.approx(-1.5)

    def test_Am_63Hz(self, ga):
        """Am at 63 Hz = -3q regardless of Gm."""
        Am = ga._middle_atten(63)
        assert Am == pytest.approx(-3.0 * ga.q)

    @pytest.mark.parametrize("freq", [125, 250, 500, 1000, 2000, 4000, 8000])
    def test_Am_other_bands_hard(self, ga, freq):
        """Am at 125-8000 Hz with Gm=0: -3q(1-0) = -3q."""
        Am = ga._middle_atten(freq)
        assert Am == pytest.approx(-3.0 * ga.q)

    @pytest.mark.parametrize("freq", OCTAVE_BANDS)
    def test_total_hard(self, ga, freq):
        """Agr = As + Ar + Am for hard ground."""
        Agr = ga.ground_attenuation(freq)
        expected_Am = -3.0 * ga.q  # Gm=0, so -3q(1-0) = -3q for all bands
        assert Agr == pytest.approx(-3.0 + expected_Am)


# ------------------------------------------------------------------
# 2. Porous ground (G=1 everywhere)
# ------------------------------------------------------------------

class TestPorousGround:
    """With G=1, verify each band uses the correct helper function."""

    @pytest.fixture
    def ga(self):
        return GroundAttenuation(
            source_height=1.0, receiver_height=2.0, distance=100.0,
            G_source=1.0, G_receiver=1.0, G_middle=1.0,
        )

    def test_63Hz(self, ga):
        """63 Hz: As = Ar = -1.5 always."""
        assert ga.ground_attenuation(63) == pytest.approx(
            -1.5 + -1.5 + -3.0 * ga.q
        )

    def test_125Hz(self, ga):
        As = -1.5 + 1.0 * ga._a_prime(ga.hs)
        Ar = -1.5 + 1.0 * ga._a_prime(ga.hr)
        Am = -3.0 * ga.q * (1.0 - 1.0)  # Gm=1 → Am=0
        assert ga.ground_attenuation(125) == pytest.approx(As + Ar + Am)

    def test_250Hz(self, ga):
        As = -1.5 + ga._b_prime(ga.hs)
        Ar = -1.5 + ga._b_prime(ga.hr)
        assert ga.ground_attenuation(250) == pytest.approx(As + Ar + 0.0)

    def test_500Hz(self, ga):
        As = -1.5 + ga._c_prime(ga.hs)
        Ar = -1.5 + ga._c_prime(ga.hr)
        assert ga.ground_attenuation(500) == pytest.approx(As + Ar + 0.0)

    def test_1000Hz(self, ga):
        As = -1.5 + ga._d_prime(ga.hs)
        Ar = -1.5 + ga._d_prime(ga.hr)
        assert ga.ground_attenuation(1000) == pytest.approx(As + Ar + 0.0)

    @pytest.mark.parametrize("freq", [2000, 4000, 8000])
    def test_high_freq(self, ga, freq):
        """2000-8000 Hz with G=1: As = Ar = -1.5(1-1) = 0, Am = 0."""
        assert ga.ground_attenuation(freq) == pytest.approx(0.0)


# ------------------------------------------------------------------
# 3. Mixed ground (G=0.5)
# ------------------------------------------------------------------

class TestMixedGround:
    """G=0.5 on all regions tests interpolation between hard and porous."""

    @pytest.fixture
    def ga(self):
        return GroundAttenuation(
            source_height=1.5, receiver_height=1.5, distance=150.0,
            G_source=0.5, G_receiver=0.5, G_middle=0.5,
        )

    def test_high_freq_mixed(self, ga):
        """2000 Hz: As = Ar = -1.5(1-0.5) = -0.75."""
        As = -1.5 * (1.0 - 0.5)
        Ar = -1.5 * (1.0 - 0.5)
        Am = -3.0 * ga.q * (1.0 - 0.5)
        assert ga.ground_attenuation(2000) == pytest.approx(As + Ar + Am)

    def test_125Hz_mixed(self, ga):
        As = -1.5 + 0.5 * ga._a_prime(ga.hs)
        Ar = -1.5 + 0.5 * ga._a_prime(ga.hr)
        Am = -3.0 * ga.q * 0.5
        assert ga.ground_attenuation(125) == pytest.approx(As + Ar + Am)


# ------------------------------------------------------------------
# 4. No middle region (dp ≤ 30(hs+hr) → q=0 → Am=0)
# ------------------------------------------------------------------

class TestNoMiddleRegion:
    """When dp ≤ 30(hs+hr), q=0 and Am=0."""

    def test_q_zero(self):
        # hs=2, hr=3, 30*(2+3)=150, dp=100 < 150 → q=0
        ga = GroundAttenuation(2.0, 3.0, 100.0)
        assert ga.q == 0.0

    def test_q_zero_boundary(self):
        # dp exactly = 30(hs+hr) → q=0
        ga = GroundAttenuation(1.0, 1.0, 60.0)
        assert ga.q == 0.0

    def test_Am_zero_when_q_zero(self):
        ga = GroundAttenuation(2.0, 3.0, 100.0)
        for freq in OCTAVE_BANDS:
            assert ga._middle_atten(freq) == 0.0


# ------------------------------------------------------------------
# 5. Large distance (q → 1)
# ------------------------------------------------------------------

class TestLargeDistance:
    """At very large dp, q approaches 1."""

    def test_q_approaches_1(self):
        ga = GroundAttenuation(0.5, 0.5, 100000.0)
        assert ga.q == pytest.approx(1.0, abs=0.001)

    def test_Am_approaches_limit(self):
        ga = GroundAttenuation(0.5, 0.5, 100000.0, G_middle=0.0)
        # Am at 125 Hz with Gm=0: -3q(1-0) ≈ -3
        assert ga._middle_atten(125) == pytest.approx(-3.0, abs=0.01)

    def test_Am_porous_large_distance(self):
        ga = GroundAttenuation(0.5, 0.5, 100000.0, G_middle=1.0)
        # Am at 125 Hz with Gm=1: -3q(1-1) = 0
        assert ga._middle_atten(125) == pytest.approx(0.0)


# ------------------------------------------------------------------
# 6. Zero height edge case
# ------------------------------------------------------------------

class TestZeroHeight:
    """h=0 should work correctly in the helper functions."""

    def test_helper_functions_at_zero_height(self):
        ga = GroundAttenuation(0.0, 0.0, 200.0)
        # Should not raise; verify helpers return finite values
        for fn in [ga._a_prime, ga._b_prime, ga._c_prime, ga._d_prime]:
            val = fn(0.0)
            assert math.isfinite(val)

    def test_a_prime_zero_height(self):
        ga = GroundAttenuation(0.0, 0.0, 200.0)
        h = 0.0
        dp = 200.0
        expected = (1.5
                    + 3.0 * math.exp(-0.12 * (0.0 - 5.0) ** 2)
                      * (1.0 - math.exp(-dp / 50.0))
                    + 5.7 * math.exp(-0.09 * 0.0)
                      * (1.0 - math.exp(-2.8e-6 * dp ** 2)))
        assert ga._a_prime(h) == pytest.approx(expected)

    def test_q_zero_height(self):
        """With hs=hr=0, 30*(0+0)=0, so q=1-0/dp=1 for any dp>0."""
        ga = GroundAttenuation(0.0, 0.0, 100.0)
        assert ga.q == pytest.approx(1.0)


# ------------------------------------------------------------------
# 7. Vectorized input
# ------------------------------------------------------------------

class TestVectorized:
    """Verify ground_attenuation accepts arrays and returns arrays."""

    def test_array_input_returns_array(self):
        ga = GroundAttenuation(1.0, 1.5, 200.0)
        freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        result = ga.ground_attenuation(freqs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)
        assert np.all(np.isfinite(result))

    def test_list_input_returns_array(self):
        ga = GroundAttenuation(1.0, 1.5, 200.0)
        result = ga.ground_attenuation([63, 500, 4000])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_scalar_input_returns_float(self):
        ga = GroundAttenuation(1.0, 1.5, 200.0)
        result = ga.ground_attenuation(500)
        assert isinstance(result, float)

    def test_array_matches_scalar(self):
        """Array output must match individual scalar calls."""
        ga = GroundAttenuation(1.0, 2.0, 200.0, G_source=0.5, G_receiver=0.7)
        freqs = list(OCTAVE_BANDS)
        array_result = ga.ground_attenuation(freqs)
        for i, f in enumerate(freqs):
            scalar_result = ga.ground_attenuation(f)
            assert array_result[i] == pytest.approx(scalar_result)

    def test_single_element_array(self):
        ga = GroundAttenuation(1.0, 1.0, 100.0)
        result = ga.ground_attenuation(np.array([250]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_invalid_in_array_raises(self):
        ga = GroundAttenuation(1.0, 1.0, 100.0)
        with pytest.raises(ValueError, match="octave-band"):
            ga.ground_attenuation([63, 100, 250])


# ------------------------------------------------------------------
# 8. Invalid frequency raises ValueError
# ------------------------------------------------------------------

class TestInvalidFrequency:
    def test_non_octave_band(self):
        ga = GroundAttenuation(1.0, 1.0, 100.0)
        with pytest.raises(ValueError, match="octave-band"):
            ga.ground_attenuation(100)

    def test_non_octave_band_float(self):
        ga = GroundAttenuation(1.0, 1.0, 100.0)
        with pytest.raises(ValueError, match="octave-band"):
            ga.ground_attenuation(300.0)


# ------------------------------------------------------------------
# 9. Symmetry: swapping source/receiver
# ------------------------------------------------------------------

class TestSymmetry:
    """Swapping (hs, Gs) with (hr, Gr) should swap As and Ar."""

    def test_swap_source_receiver(self):
        ga1 = GroundAttenuation(1.0, 3.0, 200.0, G_source=0.3, G_receiver=0.7)
        ga2 = GroundAttenuation(3.0, 1.0, 200.0, G_source=0.7, G_receiver=0.3)

        for freq in OCTAVE_BANDS:
            assert ga1.ground_attenuation(freq) == pytest.approx(
                ga2.ground_attenuation(freq)
            )

    def test_As_Ar_swap(self):
        """Explicitly check that As of ga1 equals Ar of ga2."""
        ga1 = GroundAttenuation(1.0, 3.0, 200.0, G_source=0.3, G_receiver=0.7)
        ga2 = GroundAttenuation(3.0, 1.0, 200.0, G_source=0.7, G_receiver=0.3)

        for freq in OCTAVE_BANDS:
            As1 = ga1._source_or_receiver_atten(freq, ga1.Gs, ga1.hs)
            Ar2 = ga2._source_or_receiver_atten(freq, ga2.Gr, ga2.hr)
            assert As1 == pytest.approx(Ar2)


# ------------------------------------------------------------------
# 10. Known reference values
# ------------------------------------------------------------------

class TestKnownValues:
    """Cross-check against hand-computed values from Table 3 formulas."""

    def test_hard_ground_all_bands(self):
        """Hard ground, typical geometry: known exact result."""
        ga = GroundAttenuation(2.0, 4.0, 500.0,
                               G_source=0.0, G_receiver=0.0, G_middle=0.0)
        # q = 1 - 30*(2+4)/500 = 1 - 180/500 = 0.64
        assert ga.q == pytest.approx(0.64)
        # All bands: As=Ar=-1.5, Am=-3*0.64*(1-0)=-1.92
        for freq in OCTAVE_BANDS:
            assert ga.ground_attenuation(freq) == pytest.approx(-3.0 - 1.92)

    def test_porous_ground_2000Hz(self):
        """Porous ground at 2000 Hz: As=Ar=0, Am=0 (Gm=1)."""
        ga = GroundAttenuation(1.0, 1.0, 500.0,
                               G_source=1.0, G_receiver=1.0, G_middle=1.0)
        assert ga.ground_attenuation(2000) == pytest.approx(0.0)

    def test_porous_250Hz_specific(self):
        """Porous ground, hs=hr=0, dp=200: check b'(0) value."""
        ga = GroundAttenuation(0.0, 0.0, 200.0,
                               G_source=1.0, G_receiver=1.0, G_middle=1.0)
        # b'(0) = 1.5 + 8.6 * exp(0) * (1 - exp(-200/50))
        #       = 1.5 + 8.6 * 1.0 * (1 - exp(-4))
        #       = 1.5 + 8.6 * 0.981684... = 1.5 + 8.4425 = 9.9425
        b_prime = 1.5 + 8.6 * math.exp(-0.09 * 0) * (1 - math.exp(-200 / 50))
        As = -1.5 + 1.0 * b_prime
        Ar = As  # same h
        Am = 0.0  # Gm=1
        assert ga.ground_attenuation(250) == pytest.approx(As + Ar + Am, rel=1e-6)


# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------

class TestConstructorValidation:
    def test_negative_distance(self):
        with pytest.raises(ValueError, match="distance"):
            GroundAttenuation(1.0, 1.0, -10.0)

    def test_zero_distance(self):
        with pytest.raises(ValueError, match="distance"):
            GroundAttenuation(1.0, 1.0, 0.0)

    def test_G_out_of_range(self):
        with pytest.raises(ValueError, match="G_source"):
            GroundAttenuation(1.0, 1.0, 100.0, G_source=1.5)

    def test_G_negative(self):
        with pytest.raises(ValueError, match="G_receiver"):
            GroundAttenuation(1.0, 1.0, 100.0, G_receiver=-0.1)

    def test_repr(self):
        ga = GroundAttenuation(1.0, 2.0, 100.0, 0.5, 0.5, 0.5)
        assert "GroundAttenuation" in repr(ga)
        assert "hs=1.0" in repr(ga)
