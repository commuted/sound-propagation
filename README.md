# ISO 9613 Propagation Toolkit

A small, pure-Python library that implements the core calculations from the ISO 9613 series:

- **ISO 9613-1:1993** -- Atmospheric absorption and geometric spreading of sound outdoors.
- **ISO 9613-2:1996** -- Ground-attenuation model for flat terrain (three-region model).

The toolkit provides two high-level classes:

| Class                    | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `AtmosphericPropagation` | Compute the frequency-dependent atmospheric absorption coefficient **α** (dB/m) and the resulting sound-pressure-level (SPL) change between a source and a receiver, optionally adding ground attenuation. |
| `GroundAttenuation`      | Calculate the ground-attenuation term **A<sub>gr</sub>** (dB) for the standard octave-band centre frequencies (63 Hz -- 8 kHz) using the three-region model defined in ISO 9613-2. |

Both classes are deliberately lightweight, have no external dependencies beyond **NumPy**, and are fully typed for IDE support. Requires **Python 3.10+** (uses `X | Y` union syntax).

Validated against **102 data points from ISO 9613-1:1993 Table 1** spanning -20 °C to +50 °C, 10--100 % RH, and 50--8000 Hz.

------

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed API](#detailed-api)
   - [`AtmosphericPropagation`](#atmosphericpropagation)
   - [`GroundAttenuation`](#groundattenuation)
4. [Examples](#examples)
5. [Testing & Development](#testing--development)
6. [License](#license)

------

## Installation

```bash
pip install sound-propagation
```

Or install from source:

```bash
git clone https://github.com/commuted/sound-propagation.git
cd sound-propagation
pip install -e .
```

------

## Quick Start

```python
from sound_propagation import AtmosphericPropagation, GroundAttenuation

# 1. Define atmospheric conditions
prop = AtmosphericPropagation(
    temperature_c=20.0,           # °C
    relative_humidity_pct=50.0,   # %
    pressure_kpa=101.325,         # kPa (standard atmosphere)
    source=(0, 0, 0),             # source position (m)
    recording=(100, 0, 0),        # mic position (m)
    warn_frequency=True,          # optional warnings for out-of-range freqs
)

# 2. Compute atmospheric absorption for a single frequency
alpha = prop.absorption_coefficient(1000)   # dB/m at 1 kHz
print(f"Absorption coefficient at 1 kHz: {alpha:.4f} dB/m")

# 3. Get attenuation at an arbitrary 3-D point (shorthand via __call__)
result = prop(1000, (150, 0, 0))
print("Total SPL change (dB):", result["total_dB"])

# 4. Get attenuation at an on-axis offset from the recording position
result = prop.attenuation_at_offset(
    frequency=[500, 1000, 2000],
    distance_offset=20.0,         # 20 m beyond mic
)
print("Total SPL change (dB):", result["total_dB"])

# 5. Add ground attenuation (optional)
ground = GroundAttenuation(
    source_height=2.0,
    receiver_height=1.5,
    distance=120.0,
    G_source=0.8,
    G_receiver=0.5,
    G_middle=0.6,
)

# 6. Combine everything (atmosphere + geometry + ground)
full = prop.total_attenuation(
    frequency=AtmosphericPropagation.OCTAVE_BANDS,
    eval_pos=(30, 0, 0),
    ground=ground,
)
print("Total attenuation per octave band (dB):", full["total_dB"])

# 7. A-weighted convenience
a_weighted = prop.a_weighted_attenuation((30, 0, 0), ground=ground)
print("A-weighted attenuation (dB):", a_weighted)
```

------

## Detailed API

### `AtmosphericPropagation`

#### Constructor

```python
AtmosphericPropagation(
    temperature_c: float,
    relative_humidity_pct: float,
    pressure_kpa: float | None = None,
    source: Position = (0.0, 0.0, 0.0),
    recording: Position = (100.0, 0.0, 0.0),
    copy: bool = False,
    warn_frequency: bool = False,
)
```

| Parameter               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `temperature_c`         | Ambient temperature (°C).                                    |
| `relative_humidity_pct` | Relative humidity (0 % -- 100 %).                            |
| `pressure_kpa`          | Atmospheric pressure (kPa). Defaults to standard 101.325 kPa. |
| `source` / `recording`  | 3-D position as a tuple, list, or `np.ndarray` of shape `(3,)`. |
| `copy`                  | If `True`, NumPy position arrays are copied; otherwise stored by reference. |
| `warn_frequency`        | Emit a warning when a frequency falls outside the ISO-validated 50 Hz -- 10 kHz range. |

#### Core Methods

| Method                                                   | Return                  | Description                                                  |
| -------------------------------------------------------- | ----------------------- | ------------------------------------------------------------ |
| `absorption_coefficient(frequency)`                      | `float` or `np.ndarray` | Pure-tone atmospheric absorption **α** (dB/m) per ISO 9613-1 §4. Accepts f = 0 (returns 0). |
| `absorption_coefficient_octave(center_frequencies=None)` | `np.ndarray`            | Same as above but for the eight standard octave-band centre frequencies. |
| `attenuation_at_offset(frequency, distance_offset)`      | `dict`                  | SPL change at a point displaced *distance_offset* m from the recording position along the source-to-receiver line. |
| `attenuation_at_position(frequency, eval_pos)`           | `dict`                  | SPL change at any arbitrary 3-D point.                       |
| `total_attenuation(frequency, eval_pos, ground=None)`    | `dict`                  | Combines atmospheric, geometric, and optional ground attenuation. |
| `a_weighted_attenuation(eval_pos, ground=None)`          | `np.ndarray`            | A-weighted total attenuation for the eight standard octave bands. |
| `__call__(frequency, eval_pos)`                          | `dict`                  | Shortcut for `attenuation_at_position`.                      |

#### Return Dict Keys

Methods returning `dict` include these keys:

| Key                    | Type                    | Description                                          |
| ---------------------- | ----------------------- | ---------------------------------------------------- |
| `total_dB`             | `float` or `np.ndarray` | Net SPL change (positive = louder, negative = quieter). |
| `atmospheric_dB`       | `float` or `np.ndarray` | Contribution from atmospheric absorption only.       |
| `geometric_dB`         | `float`                 | Contribution from geometric (1/r²) spreading only.   |
| `ground_dB`            | `float` or `np.ndarray` | Ground attenuation (only present when `ground` is provided). |
| `distance_to_source`   | `float`                 | Distance from source to evaluation point (m).        |
| `distance_to_receiver` | `float`                 | Distance from recording to evaluation point (m).     |

#### Class Constants

| Constant       | Value                                                    |
| -------------- | -------------------------------------------------------- |
| `OCTAVE_BANDS` | `np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])` |
| `A_WEIGHT`     | `np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 0.9, -1.1])` |

------

### `GroundAttenuation`

#### Constructor

```python
GroundAttenuation(
    source_height: float,
    receiver_height: float,
    distance: float,
    G_source: float = 1.0,
    G_receiver: float = 1.0,
    G_middle: float = 1.0,
)
```

| Parameter                            | Meaning                                        |
| ------------------------------------ | ---------------------------------------------- |
| `source_height` (`hs`)               | Height of the source above ground (m).         |
| `receiver_height` (`hr`)             | Height of the receiver above ground (m).       |
| `distance` (`dp`)                    | Horizontal source-to-receiver distance (m).    |
| `G_source`, `G_receiver`, `G_middle` | Ground factors (0 = hard surface, 1 = porous). |

#### Core Method

| Method                          | Return                  | Description                                                  |
| ------------------------------- | ----------------------- | ------------------------------------------------------------ |
| `ground_attenuation(frequency)` | `float` or `np.ndarray` | Calculates **A<sub>gr</sub>** (dB) for the supplied octave-band centre frequencies using the three-region model (ISO 9613-2 §7.3.1). |

------

## Examples

### 1. Plotting atmospheric absorption vs. frequency

```python
import matplotlib.pyplot as plt
import numpy as np
from sound_propagation import AtmosphericPropagation

prop = AtmosphericPropagation(
    temperature_c=25,
    relative_humidity_pct=60,
    source=(0, 0, 0),
    recording=(150, 0, 0),
)

freqs = np.logspace(np.log10(50), np.log10(10000), 200)
alpha = prop.absorption_coefficient(freqs)

plt.semilogx(freqs, alpha)
plt.title("Atmospheric Absorption Coefficient (ISO 9613-1)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("α (dB/m)")
plt.grid(True, which="both")
plt.show()
```

### 2. Comparing ground attenuation for different surfaces

```python
from sound_propagation import GroundAttenuation

freqs = GroundAttenuation.OCTAVE_BANDS

hard   = GroundAttenuation(1.5, 1.5, 300, G_source=0.0, G_receiver=0.0, G_middle=0.0)
porous = GroundAttenuation(1.5, 1.5, 300, G_source=1.0, G_receiver=1.0, G_middle=1.0)

print("Hard ground:  ", hard.ground_attenuation(freqs))
print("Porous ground:", porous.ground_attenuation(freqs))
```

### 3. Full ISO 9613 prediction (atmosphere + ground)

```python
from sound_propagation import AtmosphericPropagation, GroundAttenuation

prop = AtmosphericPropagation(20, 55, source=(0, 0, 0), recording=(80, 0, 0))
ground = GroundAttenuation(2.0, 1.0, 80, G_source=0.8, G_receiver=0.5, G_middle=0.6)

total = prop.total_attenuation(
    frequency=AtmosphericPropagation.OCTAVE_BANDS,
    eval_pos=(120, 0, 0),
    ground=ground,
)

print("Total attenuation per octave (dB):", total["total_dB"])
```

------

## Testing & Development

The test suite contains **256 tests** including validation against all 102 data points from **ISO 9613-1:1993 Table 1**.

```bash
# Run all tests
./env-sound_prop/bin/pytest tests/ -v

# Run only Table 1 validation (102 parametrized cases)
./env-sound_prop/bin/pytest tests/test_atmospheric_absorption.py -v -k "test_absorption_coefficient_against_table1"

# Run ground attenuation tests
./env-sound_prop/bin/pytest tests/test_ground_attenuation.py -v
```

Tolerances: 1 % relative for values >= 1 dB/km, 0.01 dB/km absolute for smaller values.

------

## License

This code is released under the **MIT License**. See the accompanying [`LICENSE`](LICENSE) file for the full text.

------

### Contributing

Open an issue or pull request on the repository.
