# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ISO 9613-1/2 atmospheric sound absorption, propagation, and ground attenuation library. Python with numpy dependency.

## Setup

```bash
# Install in editable mode (from the venv)
./env-sound_prop/bin/pip install -e .
```

## Commands

```bash
# Run all tests (use the venv's pytest)
./env-sound_prop/bin/pytest tests/ -v

# Run a single parametrized test by keyword
./env-sound_prop/bin/pytest tests/test_atmospheric_absorption.py -v -k "T=+20C_RH=70%_f=1000Hz"

# Run only the Table 1 validation tests
./env-sound_prop/bin/pytest tests/test_atmospheric_absorption.py -v -k "test_absorption_coefficient_against_table1"

# Run only the attenuation_at_offset tests
./env-sound_prop/bin/pytest tests/test_atmospheric_absorption.py -v -k "TestAttenuationAtOffset"

# Run ground attenuation tests
./env-sound_prop/bin/pytest tests/test_ground_attenuation.py -v
```

No build step, linter, or formatter is configured. The venv is at `env-sound_prop/` (Python 3.12, pytest installed).

## Architecture

**Package:** `src/sound_propagation/` (installed as `sound_propagation`).

Two modules:
- `atmospheric_absorption.py` — class `AtmosphericPropagation` (ISO 9613-1)
- `ground_attenuation.py` — class `GroundAttenuation` (ISO 9613-2 §7.3.1)

Both are re-exported from `sound_propagation.__init__`.

### AtmosphericPropagation

**Constructor** takes environmental conditions (temperature, humidity, pressure) and two 3D positions (sound source, recording microphone). Position arguments (`source`, `recording`) accept tuples, lists, or `np.ndarray` (shape `(3,)`); a `copy` flag controls whether ndarrays are copied on storage. It pre-computes humidity-dependent O₂ and N₂ relaxation frequencies on init.

**Public methods:**
- `absorption_coefficient(frequency)` — returns the ISO 9613-1 §6.2 pure-tone absorption coefficient α in **dB/m**
- `absorption_coefficient_octave(center_frequencies=None)` — convenience wrapper returning α for standard octave bands (63–8000 Hz by default)
- `attenuation_at_offset(frequency, distance_offset)` — returns a dict with the SPL change (total, atmospheric, geometric components) at a point offset along the source→recording ray
- `attenuation_at_position(frequency, eval_pos)` — SPL change at an arbitrary 3-D position
- `total_attenuation(frequency, eval_pos, ground=None)` — combined atmospheric + geometric + optional ground attenuation in a single call

### GroundAttenuation

Computes ground attenuation Agr = As + Ar + Am per ISO 9613-2 Table 3, for the eight standard octave bands.

## Test data

`tests/test_atmospheric_absorption.py` validates against **ISO 9613-1:1993 Table 1** (Section 8.1.1) — 102 parametrized cases spanning -20 °C to +50 °C, 10–100% RH, 50–8000 Hz. Table values are in dB/km; the test converts by multiplying the code's dB/m output by 1000.

The table uses **exact midband frequencies** per Equation 6: `f_m = 1000 × 10^(k/10)` (not the rounded preferred labels like 125, 4000, etc.). The helper `exact_freq()` does this conversion.

Tolerances: 1% relative for values ≥ 1 dB/km, 0.01 dB/km absolute for smaller values.
