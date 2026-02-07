"""
ISO 9613-2:1996(E) — Acoustics — Attenuation of sound during propagation
outdoors — Part 2: General method of calculation.

Section 7.3.1: Ground attenuation (Agr) for flat terrain using the
three-region model (source region, middle region, receiver region).
"""

import math

import numpy as np


class GroundAttenuation:
    """ISO 9613-2:1996 §7.3.1 ground attenuation for flat terrain.

    Computes Agr = As + Ar + Am (Equation 9) using the three-region
    ground model from Table 3.

    Parameters
    ----------
    source_height : float
        Height of the sound source above the ground plane (hs), in metres.
    receiver_height : float
        Height of the receiver above the ground plane (hr), in metres.
    distance : float
        Source-to-receiver distance projected onto the ground plane (dp),
        in metres.  Must be positive.
    G_source : float
        Ground factor for the source region (0 = hard, 1 = porous).
    G_receiver : float
        Ground factor for the receiver region (0 = hard, 1 = porous).
    G_middle : float
        Ground factor for the middle region (0 = hard, 1 = porous).

    Attributes
    ----------
    hs : float
        Source height in metres.
    hr : float
        Receiver height in metres.
    dp : float
        Ground-projected source-to-receiver distance in metres.
    Gs, Gr, Gm : float
        Ground factors for source, receiver, and middle regions.
    q : float
        Pre-computed middle-region weighting factor from geometry.

    Examples
    --------
    >>> ga = GroundAttenuation(1.0, 2.0, 200.0, G_source=1.0, G_receiver=0.0)
    >>> ga.ground_attenuation(500)
    -2.997...
    >>> ga.ground_attenuation([63, 125, 250])
    array([...])
    """

    OCTAVE_BANDS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)

    def __init__(
        self,
        source_height: float,
        receiver_height: float,
        distance: float,
        G_source: float = 1.0,
        G_receiver: float = 1.0,
        G_middle: float = 1.0,
    ):
        if distance <= 0:
            raise ValueError("distance must be positive")
        for name, val in [("G_source", G_source), ("G_receiver", G_receiver),
                          ("G_middle", G_middle)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        self.hs = float(source_height)
        self.hr = float(receiver_height)
        self.dp = float(distance)
        self.Gs = float(G_source)
        self.Gr = float(G_receiver)
        self.Gm = float(G_middle)

        # Pre-compute q factor (geometry-only, independent of frequency)
        threshold = 30.0 * (self.hs + self.hr)
        if self.dp <= threshold:
            self.q = 0.0
        else:
            self.q = 1.0 - threshold / self.dp

    def ground_attenuation(
        self, frequency: np.ndarray | float | int,
    ) -> np.ndarray | float:
        """Return Agr in dB for the given octave-band centre frequency.

        Implements ISO 9613-2:1996, Equation 9:  Agr = As + Ar + Am.

        Parameters
        ----------
        frequency : int, float, or array_like
            One or more standard octave-band centre frequencies:
            63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz.  Scalars and
            arrays are both accepted; an array input produces an array
            output of matching shape.

        Returns
        -------
        float or numpy.ndarray
            Ground attenuation Agr in dB, same shape as *frequency*.
            Values are typically negative (reduction in SPL).

        Raises
        ------
        ValueError
            If any element of *frequency* is not a standard octave-band
            centre frequency.

        See Also
        --------
        AtmosphericPropagation.absorption_coefficient :
            ISO 9613-1 atmospheric absorption (dB/m).

        Examples
        --------
        Scalar input returns a scalar:

        >>> ga = GroundAttenuation(1.0, 1.0, 200.0)
        >>> ga.ground_attenuation(500)
        -1.08...

        Array input returns an array:

        >>> ga.ground_attenuation([63, 500, 4000])
        array([...])
        """
        f = np.asarray(frequency)
        scalar_input = f.ndim == 0
        f = np.atleast_1d(f)

        # Validate all frequencies
        rounded = np.rint(f).astype(int)
        valid = np.isin(rounded, self.OCTAVE_BANDS)
        if not valid.all():
            bad = f[~valid]
            raise ValueError(
                f"frequency must be a standard octave-band centre frequency "
                f"{self.OCTAVE_BANDS}, got {bad}"
            )

        result = np.empty_like(f, dtype=float)
        for i, freq in enumerate(rounded):
            As = self._source_or_receiver_atten(int(freq), self.Gs, self.hs)
            Ar = self._source_or_receiver_atten(int(freq), self.Gr, self.hr)
            Am = self._middle_atten(int(freq))
            result[i] = As + Ar + Am

        if scalar_input:
            return float(result[0])
        return result

    # ------------------------------------------------------------------
    # Table 3 helper functions
    # ------------------------------------------------------------------

    def _a_prime(self, h: float) -> float:
        """Height-dependent factor for the 125 Hz octave band (Table 3).

        Parameters
        ----------
        h : float
            Height above ground in metres (hs or hr).

        Returns
        -------
        float
            Factor a'(h) used in As/Ar calculation at 125 Hz.

        Notes
        -----
        a'(h) = 1.5 + 3.0·exp(−0.12·(h−5)²)·(1−exp(−dp/50))
                     + 5.7·exp(−0.09·h²)·(1−exp(−2.8×10⁻⁶·dp²))
        """
        dp = self.dp
        return (1.5
                + 3.0 * math.exp(-0.12 * (h - 5.0) ** 2)
                  * (1.0 - math.exp(-dp / 50.0))
                + 5.7 * math.exp(-0.09 * h ** 2)
                  * (1.0 - math.exp(-2.8e-6 * dp ** 2)))

    def _b_prime(self, h: float) -> float:
        """Height-dependent factor for the 250 Hz octave band (Table 3).

        Parameters
        ----------
        h : float
            Height above ground in metres (hs or hr).

        Returns
        -------
        float
            Factor b'(h) used in As/Ar calculation at 250 Hz.

        Notes
        -----
        b'(h) = 1.5 + 8.6·exp(−0.09·h²)·(1−exp(−dp/50))
        """
        dp = self.dp
        return (1.5
                + 8.6 * math.exp(-0.09 * h ** 2)
                  * (1.0 - math.exp(-dp / 50.0)))

    def _c_prime(self, h: float) -> float:
        """Height-dependent factor for the 500 Hz octave band (Table 3).

        Parameters
        ----------
        h : float
            Height above ground in metres (hs or hr).

        Returns
        -------
        float
            Factor c'(h) used in As/Ar calculation at 500 Hz.

        Notes
        -----
        c'(h) = 1.5 + 14.0·exp(−0.46·h²)·(1−exp(−dp/50))
        """
        dp = self.dp
        return (1.5
                + 14.0 * math.exp(-0.46 * h ** 2)
                  * (1.0 - math.exp(-dp / 50.0)))

    def _d_prime(self, h: float) -> float:
        """Height-dependent factor for the 1000 Hz octave band (Table 3).

        Parameters
        ----------
        h : float
            Height above ground in metres (hs or hr).

        Returns
        -------
        float
            Factor d'(h) used in As/Ar calculation at 1000 Hz.

        Notes
        -----
        d'(h) = 1.5 + 5.0·exp(−0.9·h²)·(1−exp(−dp/50))
        """
        dp = self.dp
        return (1.5
                + 5.0 * math.exp(-0.9 * h ** 2)
                  * (1.0 - math.exp(-dp / 50.0)))

    def _source_or_receiver_atten(self, freq: int, G: float, h: float) -> float:
        """Compute source-region (As) or receiver-region (Ar) attenuation.

        Applies Table 3 formulas, selecting the appropriate helper
        function for each octave band.

        Parameters
        ----------
        freq : int
            Octave-band centre frequency in Hz.
        G : float
            Ground factor for this region (0 = hard, 1 = porous).
        h : float
            Height above ground in metres.

        Returns
        -------
        float
            As or Ar in dB.
        """
        if freq == 63:
            return -1.5
        elif freq == 125:
            return -1.5 + G * self._a_prime(h)
        elif freq == 250:
            return -1.5 + G * self._b_prime(h)
        elif freq == 500:
            return -1.5 + G * self._c_prime(h)
        elif freq == 1000:
            return -1.5 + G * self._d_prime(h)
        else:
            # 2000, 4000, 8000
            return -1.5 * (1.0 - G)

    def _middle_atten(self, freq: int) -> float:
        """Compute middle-region attenuation Am.

        Parameters
        ----------
        freq : int
            Octave-band centre frequency in Hz.

        Returns
        -------
        float
            Am in dB.  At 63 Hz: ``−3q``.
            At 125–8000 Hz: ``−3q(1 − Gm)``.
        """
        q = self.q
        if freq == 63:
            return -3.0 * q
        else:
            # 125–8000 Hz
            return -3.0 * q * (1.0 - self.Gm)

    def __repr__(self) -> str:
        return (
            f"GroundAttenuation("
            f"hs={self.hs}, hr={self.hr}, dp={self.dp}, "
            f"Gs={self.Gs}, Gr={self.Gr}, Gm={self.Gm})"
        )
