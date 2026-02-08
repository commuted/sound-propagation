"""
ISO 9613-1:1993 — Acoustics — Attenuation of sound during propagation
outdoors — Part 1: Calculation of the absorption coefficient.

Implements the pure-tone atmospheric absorption coefficient α (dB/m)
from Section 4, and a propagation method that combines atmospheric
absorption with geometric (inverse-square) spreading to give the
change in sound-pressure level at an arbitrary point along the
source-to-recording axis.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np

Position = tuple[float, float, float] | Sequence[float] | np.ndarray


class AtmosphericPropagation:
    """Compute atmospheric absorption per ISO 9613-1:1993 and propagation
    attenuation/amplification relative to a recording position.

    Parameters
    ----------
    temperature_c : float
        Ambient temperature in degrees Celsius.
    relative_humidity_pct : float
        Relative humidity in percent (0–100).
    pressure_kpa : float or None
        Ambient atmospheric pressure in kPa (default 101.325 if None).
    source : tuple, list, or np.ndarray
        (x, y, z) position of the sound source in metres.
    recording : tuple, list, or np.ndarray
        (x, y, z) position of the recording microphone in metres.
    copy : bool
        If True, ndarray positions are copied before storing.
        Lists and tuples are always converted to tuples (new objects).
        Default False.
    warn_frequency : bool
        If True, emit a warning when frequencies passed to
        ``absorption_coefficient`` fall outside the ISO 9613-1
        validated range of 50 Hz – 10 kHz.  Default False.
    """

    # ISO reference values
    T_REF = 293.15        # reference temperature  [K]
    T_TRIPLE = 273.16     # triple-point isotherm  [K]
    P_REF = 101.325       # reference pressure     [kPa]

    # Standard octave-band centre frequencies (Hz)
    OCTAVE_BANDS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])

    # Approximate A-weighting corrections (dB) for OCTAVE_BANDS
    A_WEIGHT = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 0.9, -1.1])

    def __init__(
        self,
        temperature_c: float,
        relative_humidity_pct: float,
        pressure_kpa: float | None = None,
        source: Position = (0.0, 0.0, 0.0),
        recording: Position = (100.0, 0.0, 0.0),
        copy: bool = False,
        warn_frequency: bool = False,
    ):
        if not 0 < relative_humidity_pct <= 100:
            raise ValueError("relative_humidity_pct must be in (0, 100]")
        if pressure_kpa is None:
            pressure_kpa = 101.325

        self.temperature_c = temperature_c
        self.relative_humidity_pct = relative_humidity_pct
        self.pressure_kpa = pressure_kpa
        self.source = self._validate_position(source, "source", copy)
        self.recording = self._validate_position(recording, "recording", copy)
        self._warn_frequency = warn_frequency

        self._T = temperature_c + 273.15  # absolute temperature [K]
        self._d_rec = self._distance(source, recording)
        if np.isclose(self._d_rec, 0.0):
            raise ValueError("source and recording positions must differ")

        # pre-compute humidity-dependent relaxation frequencies
        self._h = self._molar_concentration_h2o()
        self._frO = self._relaxation_freq_oxygen()
        self._frN = self._relaxation_freq_nitrogen()

        # classical + molecular rotational absorption (frequency-independent)
        pa_pr = self.pressure_kpa / self.P_REF
        self._alpha_cr = 1.84e-11 * (1.0 / pa_pr) * (self._T / self.T_REF) ** 0.5

        # ISO 9613-1 §7 accuracy warnings
        self._check_accuracy_bounds()

    # ------------------------------------------------------------------
    # ISO 9613-1 §4 internals
    # ------------------------------------------------------------------

    def _molar_concentration_h2o(self) -> float:
        """Molar concentration of water vapour, *h* (%)."""
        T = self._T
        pa = self.pressure_kpa
        pr = self.P_REF
        hr = self.relative_humidity_pct

        # Saturation vapour pressure ratio  (ISO 9613-1 Eq. 2)
        #   p_sat / p_ref = 10^(−6.8346 (T_01/T)^1.261 + 4.6151)
        psat_ratio = 10.0 ** (-6.8346 * (self.T_TRIPLE / T) ** 1.261 + 4.6151)

        return hr * psat_ratio / (pa / pr)

    def _relaxation_freq_oxygen(self) -> float:
        """Relaxation frequency of oxygen, *frO* (Hz)."""
        h = self._h
        pa_pr = self.pressure_kpa / self.P_REF
        return pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))

    def _relaxation_freq_nitrogen(self) -> float:
        """Relaxation frequency of nitrogen, *frN* (Hz)."""
        h = self._h
        T = self._T
        pa_pr = self.pressure_kpa / self.P_REF
        T_ratio = T / self.T_REF
        return pa_pr * T_ratio ** (-0.5) * (
            9.0 + 280.0 * h * math.exp(-4.170 * (T_ratio ** (-1.0 / 3.0) - 1.0))
        )

    def _check_accuracy_bounds(self) -> None:
        """Emit warnings when conditions fall outside ISO 9613-1 §7 bounds."""
        T_c = self.temperature_c
        h = self._h  # molar concentration of water vapour (%)

        # §7.1 ±10% accuracy: T ∈ [-20, +50] °C, h ∈ [0.05, 5] %
        # §7.2 ±20% accuracy: same T range, h outside [0.05, 5] but
        #       still within [0.005, 5] or h > 5 %
        # §7.3 ±50% accuracy: T > -73 °C with h < 0.005 %

        if T_c < -20.0 or T_c > 50.0:
            warnings.warn(
                f"Temperature {T_c} °C is outside the ISO 9613-1 §7.1 "
                f"validated range [-20, +50] °C; accuracy may exceed ±10%.",
                stacklevel=2,
            )

        if h < 0.05 or h > 5.0:
            if 0.005 <= h < 0.05 or h > 5.0:
                warnings.warn(
                    f"Molar concentration of water vapour h={h:.4f}% is "
                    f"outside [0.05, 5]%; accuracy degrades to ±20% "
                    f"(ISO 9613-1 §7.2).",
                    stacklevel=2,
                )
            elif h < 0.005:
                warnings.warn(
                    f"Molar concentration of water vapour h={h:.6f}% is "
                    f"below 0.005%; accuracy degrades to ±50% "
                    f"(ISO 9613-1 §7.3).",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def absorption_coefficient(
        self, frequency: np.ndarray | float
    ) -> np.ndarray | float:
        """Pure-tone absorption coefficient **α** in dB m⁻¹.

        Implements ISO 9613-1:1993 Section 4, Equation 1.

        Parameters
        ----------
        frequency : float or array_like
            Sound frequency in Hz (≥ 0).  Scalars and arrays are both
            accepted; an array input produces an array output.
            Zero returns 0 dB/m (no absorption at DC).

        Returns
        -------
        numpy scalar or ndarray
            α in dB per metre, same shape as *frequency*.
        """
        f = np.asarray(frequency, dtype=float)
        if np.any(f < 0):
            raise ValueError("frequency must be non-negative")
        if self._warn_frequency and (np.any(f < 50.0) or np.any(f > 10000.0)):
            warnings.warn(
                "One or more frequencies are outside the ISO 9613-1 "
                "validated range [50 Hz, 10 kHz]; results may be inaccurate.",
                stacklevel=2,
            )

        T = self._T
        T_ratio = T / self.T_REF

        # Vibrational relaxation — oxygen
        alpha_vib_O = (
            0.01275
            * math.exp(-2239.1 / T)
            * self._frO / (self._frO ** 2 + f ** 2)
        )

        # Vibrational relaxation — nitrogen
        alpha_vib_N = (
            0.1068
            * math.exp(-3352.0 / T)
            * self._frN / (self._frN ** 2 + f ** 2)
        )

        alpha = 8.686 * f ** 2 * (
            self._alpha_cr + T_ratio ** (-2.5) * (alpha_vib_O + alpha_vib_N)
        )

        return alpha

    def absorption_coefficient_octave(
        self, center_frequencies: np.ndarray | None = None
    ) -> np.ndarray:
        """Pure-tone absorption coefficients for octave-band centre frequencies.

        Convenience wrapper around :meth:`absorption_coefficient` that
        defaults to the eight standard octave bands (63–8000 Hz).

        Parameters
        ----------
        center_frequencies : array_like or None, optional
            Octave-band centre frequencies in Hz.  If *None* (default),
            uses ``OCTAVE_BANDS`` (63, 125, 250, 500, 1000, 2000, 4000,
            8000 Hz).

        Returns
        -------
        numpy.ndarray
            α in dB per metre, one element per frequency.
        """
        if center_frequencies is None:
            center_frequencies = self.OCTAVE_BANDS
        return self.absorption_coefficient(np.asarray(center_frequencies, dtype=float))

    def attenuation_at_offset(
        self,
        frequency: np.ndarray | float,
        distance_offset: float,
    ) -> dict[str, np.ndarray | float]:
        """Sound-pressure-level change at a point offset from the recording.

        The evaluation point lies on the ray from source → recording,
        displaced by *distance_offset* metres measured **from the
        recording position**:

        * **positive** offset → farther from the source (more attenuation)
        * **negative** offset → closer to the source (amplification)

        Both atmospheric absorption (ISO 9613-1) and geometric spreading
        (inverse-square law, 1/r²) are included.

        Parameters
        ----------
        frequency : float or array_like
            Sound frequency in Hz (≥ 0).  Scalars and arrays are both
            accepted; array input produces array-valued dict entries
            for the frequency-dependent keys.
        distance_offset : float
            Displacement from the recording position in metres.

        Returns
        -------
        dict with keys
            ``'total_dB'``          – net SPL change (positive = louder),
            ``'atmospheric_dB'``    – contribution from absorption only,
            ``'geometric_dB'``      – contribution from spreading only,
            ``'distance_to_source'``– distance from source to evaluation point [m],
            ``'distance_to_receiver'``– distance from recording to evaluation point [m].
        """
        d_rec = self._d_rec
        d_eval = d_rec + distance_offset

        if d_eval <= 0.0:
            raise ValueError(
                f"evaluation point is at or behind the source "
                f"(d_eval={d_eval:.2f} m); offset must be > {-d_rec:.2f}"
            )

        alpha = self.absorption_coefficient(frequency)

        # Atmospheric absorption difference  (positive → more loss)
        delta_absorption = alpha * (d_eval - d_rec)

        # Geometric spreading difference  (positive → more loss)
        delta_geometric = 20.0 * math.log10(d_eval / d_rec)

        # SPL change: negative totals = quieter, positive = louder
        total = -(delta_absorption + delta_geometric)

        return {
            "total_dB": total,
            "atmospheric_dB": -delta_absorption,
            "geometric_dB": -delta_geometric,
            "distance_to_source": d_eval,
            "distance_to_receiver": abs(distance_offset),
        }

    def attenuation_at_position(
        self,
        frequency: np.ndarray | float,
        eval_pos: Position,
    ) -> dict[str, np.ndarray | float]:
        """Sound-pressure-level change at an arbitrary 3-D position.

        Like :meth:`attenuation_at_offset`, but the evaluation point is
        given as an explicit (x, y, z) coordinate rather than a scalar
        offset along the source→recording axis.  This allows computing
        the SPL change at any point in space — not just on-axis.

        The returned SPL change is **relative to the recording position**:
        positive means louder than the recording, negative means quieter.

        Parameters
        ----------
        frequency : float or array_like
            Sound frequency in Hz (≥ 0).
        eval_pos : tuple, list, or np.ndarray
            (x, y, z) position of the evaluation point in metres.

        Returns
        -------
        dict with keys
            ``'total_dB'``          – net SPL change (positive = louder),
            ``'atmospheric_dB'``    – contribution from absorption only,
            ``'geometric_dB'``      – contribution from spreading only,
            ``'distance_to_source'``– distance from source to eval point [m],
            ``'distance_to_receiver'``– distance from recording to eval point [m].
        """
        eval_pos = self._validate_position(eval_pos, "eval_pos", False)
        d_eval = self._distance(self.source, eval_pos)
        if np.isclose(d_eval, 0.0):
            raise ValueError("evaluation position coincides with the source")

        d_rec = self._d_rec
        alpha = self.absorption_coefficient(frequency)

        delta_absorption = alpha * (d_eval - d_rec)
        delta_geometric = 20.0 * math.log10(d_eval / d_rec)
        total = -(delta_absorption + delta_geometric)

        return {
            "total_dB": total,
            "atmospheric_dB": -delta_absorption,
            "geometric_dB": -delta_geometric,
            "distance_to_source": d_eval,
            "distance_to_receiver": self._distance(self.recording, eval_pos),
        }

    def total_attenuation(
        self,
        frequency: np.ndarray | float,
        eval_pos: Position,
        ground: "GroundAttenuation | None" = None,
    ) -> dict[str, np.ndarray | float]:
        """Combined attenuation relative to the recording position.

        Sums atmospheric absorption, geometric spreading, and (optionally)
        ground attenuation into a single result dict.

        Parameters
        ----------
        frequency : float or array_like
            Sound frequency in Hz (≥ 0).  When *ground* is provided,
            frequencies must be standard octave-band centre frequencies
            (63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz).
        eval_pos : tuple, list, or np.ndarray
            (x, y, z) position of the evaluation point in metres.
        ground : GroundAttenuation or None, optional
            If provided, its ``ground_attenuation(frequency)`` result is
            added to the total.  The caller is responsible for constructing
            the ``GroundAttenuation`` with geometry matching this scenario.

        Returns
        -------
        dict with keys
            ``'total_dB'``          – net SPL change (positive = louder),
            ``'atmospheric_dB'``    – atmospheric absorption component,
            ``'geometric_dB'``      – geometric spreading component,
            ``'ground_dB'``         – ground attenuation (only if *ground* given),
            ``'distance_to_source'``– distance from source to eval point [m],
            ``'distance_to_receiver'``– distance from recording to eval point [m].
        """
        result = self.attenuation_at_position(frequency, eval_pos)
        if ground is not None:
            agr = ground.ground_attenuation(frequency)
            result["ground_dB"] = agr
            result["total_dB"] = result["total_dB"] + agr
        return result

    def a_weighted_attenuation(
        self,
        eval_pos: Position,
        ground: "GroundAttenuation | None" = None,
    ) -> np.ndarray:
        """A-weighted total attenuation for the eight standard octave bands.

        Calls :meth:`total_attenuation` with ``OCTAVE_BANDS`` and adds
        the approximate A-weighting corrections from ``A_WEIGHT``.

        Parameters
        ----------
        eval_pos : tuple, list, or np.ndarray
            (x, y, z) position of the evaluation point in metres.
        ground : GroundAttenuation or None, optional
            If provided, ground attenuation is included.

        Returns
        -------
        numpy.ndarray
            A-weighted SPL change in dB, one element per octave band.
        """
        result = self.total_attenuation(self.OCTAVE_BANDS, eval_pos, ground=ground)
        return result["total_dB"] + self.A_WEIGHT

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_position(pos, name: str, copy: bool) -> tuple | np.ndarray:
        if isinstance(pos, np.ndarray):
            if pos.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {pos.shape}")
            return pos.copy() if copy else pos
        pos = tuple(pos)
        if len(pos) != 3:
            raise ValueError(f"{name} must have exactly 3 elements, got {len(pos)}")
        return pos

    @staticmethod
    def _distance(a: tuple, b: tuple) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def __call__(
        self,
        frequency: np.ndarray | float,
        eval_pos: Position,
    ) -> dict[str, np.ndarray | float]:
        """Shorthand for :meth:`attenuation_at_position`."""
        return self.attenuation_at_position(frequency, eval_pos)

    def __repr__(self) -> str:
        return (
            f"AtmosphericPropagation("
            f"T={self.temperature_c} °C, "
            f"RH={self.relative_humidity_pct}%, "
            f"P={self.pressure_kpa} kPa, "
            f"d_src_rec={self._d_rec:.2f} m)"
        )
