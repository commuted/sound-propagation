# Propagation Filter

Apply ISO 9613-1 atmospheric absorption to a WAV file using a high-resolution
FIR filter.

## What it does

Given an input audio file and a propagation distance, this tool designs a
linear-phase FIR filter whose magnitude response matches the ISO 9613
frequency-dependent attenuation curve, then convolves it with the audio.
The result is a WAV file that sounds as the original would after travelling
the specified distance through the atmosphere.

The filter is built with `scipy.signal.firwin2` and applied via
`scipy.signal.fftconvolve`, giving fine frequency resolution (typically ~1 Hz
when the default filter length equals the sample rate).

## Scripts

### `propagation_fir_filter.py`

The main filtering tool. Reads a WAV file, applies the propagation model,
and writes the attenuated output.

```bash
python propagation_fir_filter.py input.wav --distance 500
python propagation_fir_filter.py input.wav --distance 200 --output nearby.wav
python propagation_fir_filter.py input.wav --distance 1000 --normalize
```

### `generate_pink_noise.py`

A helper to generate pink noise test signals. Pink noise has equal energy
per octave, making it ideal for evaluating frequency-dependent attenuation
since every octave band starts with the same power.

```bash
python generate_pink_noise.py
python generate_pink_noise.py --duration 10 --sample-rate 48000 --output test.wav
```

## Options

### propagation_fir_filter.py

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | Path to input WAV file. Accepts 16-bit, 32-bit integer, or 32/64-bit float formats. Multi-channel files are supported. |
| `--output FILE` | `<input>_fir_attenuated.wav` | Output WAV file path. If omitted, appends `_fir_attenuated` to the input filename. Output is always 16-bit PCM. |
| `--distance METRES` | `500` | Distance from the sound source to the evaluation point, in metres. This is the total propagation path length. The model uses a 1-metre reference point, so attenuation is computed relative to how the sound would measure at 1 m from the source. |
| `--temperature CELSIUS` | `20.0` | Ambient air temperature. Affects the speed of sound and the relaxation frequencies of oxygen and nitrogen, which in turn shift the absorption peaks. |
| `--humidity PERCENT` | `70.0` | Relative humidity. Has a strong, non-linear effect on absorption. Very dry air and very humid air both absorb differently; the relationship is not monotonic at all frequencies. |
| `--pressure KPA` | `101.325` | Atmospheric pressure. Standard sea-level pressure is 101.325 kPa. Higher altitudes (lower pressure) generally increase absorption per unit distance. |
| `--numtaps N` | sample rate | FIR filter length. Controls the frequency resolution of the filter: resolution is approximately `sample_rate / numtaps` Hz. The default uses the sample rate itself (e.g., 44101 taps for 44.1 kHz audio), giving ~1 Hz resolution. Must be odd; even values are automatically incremented by 1. Larger values give finer resolution at the cost of computation time. |
| `--normalize` | off | When set, the FIR filter models only the frequency-dependent atmospheric absorption. Geometric spreading (the distance-dependent volume loss from the inverse-square law) is factored out and reported as a separate scalar gain coefficient. Without this flag, the filter includes both effects. This is useful when you want to hear or analyse the spectral shaping independently from the overall volume reduction. |

### generate_pink_noise.py

| Option | Default | Description |
|--------|---------|-------------|
| `--output FILE` | `examples/pink_noise.wav` | Output WAV file path. |
| `--duration SECONDS` | `3.0` | Duration of the generated noise in seconds. |
| `--sample-rate HZ` | `44100` | Sample rate. |
| `--amplitude FLOAT` | `0.5` | Peak amplitude, from 0 to 1. The noise is normalized to this level before conversion to 16-bit PCM. |
| `--seed INT` | `42` | Random seed for reproducibility. Using the same seed and parameters always produces the identical output file. |

## Understanding `--normalize`

By default, the filter applies both atmospheric absorption (frequency-dependent)
and geometric spreading (frequency-independent inverse-square law). At short
distances, geometric spreading dominates — for example, at 10 m the geometric
loss is -20 dB while atmospheric absorption is fractions of a dB. This makes
the output very quiet without revealing the spectral character.

With `--normalize`, the tool separates these two effects:

- The **FIR filter** applies only atmospheric absorption (the interesting,
  frequency-dependent part).
- The **geometric spreading** is reported as a constant coefficient that you
  can apply separately if needed.

Example output with `--normalize` at 500 m:

```
Geometric spreading: -54.0 dB (gain coefficient: 0.002000)

Atmospheric attenuation at key frequencies:
    125 Hz: -0.2 dB
    500 Hz: -1.4 dB
   1000 Hz: -2.5 dB
   4000 Hz: -11.5 dB
```

The coefficient `k = 0.002` means that to reconstruct the true SPL at 500 m,
you would multiply the filtered signal by 0.002 (or equivalently subtract
54 dB).

## Atmospheric conditions and their effects

The ISO 9613-1 model is sensitive to environmental conditions:

- **Temperature** shifts the molecular relaxation frequencies of O2 and N2.
  Higher temperatures move the absorption peak to higher frequencies.
- **Humidity** has a strong non-linear effect. Moderate humidity (~40-60%)
  often produces the highest absorption at mid frequencies. Both very dry
  and very humid conditions can reduce absorption at some frequencies.
- **Pressure** scales inversely with absorption. At altitude (lower pressure),
  absorption per metre increases.

At short distances (<100 m), atmospheric absorption is negligible for most
audio frequencies. The effect becomes significant above 1 kHz at distances
of several hundred metres, and dominates the spectral balance at kilometre
scales.
