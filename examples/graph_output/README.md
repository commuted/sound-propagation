# Graph Output

Visualize the frequency-dependent effect of atmospheric absorption on a sound
signal at multiple distances.

## What it does

This script generates a power spectral density (PSD) plot comparing a source
signal at three positions: the origin, a listener distance, and a target
distance. It uses normalized filtering — only the frequency-dependent
atmospheric absorption is applied, with geometric spreading factored out and
displayed as a constant coefficient on the plot.

The result clearly shows how the atmosphere acts as a natural low-pass filter:
low frequencies pass through with minimal loss, while high frequencies are
progressively absorbed with distance.

## Usage

```bash
# Default: generate pink noise, plot at 200 m and 500 m, show interactively
python plot_propagation_psd.py

# Custom distances
python plot_propagation_psd.py --listener-distance 100 --target-distance 1000

# Save to file instead of displaying
python plot_propagation_psd.py --output propagation.png

# Use your own audio file as the source signal
python plot_propagation_psd.py --input recording.wav

# Different atmospheric conditions
python plot_propagation_psd.py --temperature 35 --humidity 30 --target-distance 1000
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input FILE` | *(generate pink noise)* | Input WAV file to use as the source signal. If omitted, the script generates 5 seconds of pink noise internally (using `custom_noise` if installed, otherwise a built-in fallback). Pink noise is the ideal test signal because its equal-energy-per-octave spectrum makes absorption effects easy to read on a log-frequency plot. Multi-channel files use the first channel only. |
| `--listener-distance METRES` | `200.0` | Distance from the source to the listener position. This is the middle curve on the plot. |
| `--target-distance METRES` | `500.0` | Distance from the source to the target position. This is the farthest curve on the plot, showing the most absorption. |
| `--temperature CELSIUS` | `20.0` | Ambient air temperature. Affects molecular relaxation frequencies and shifts where absorption peaks occur in the spectrum. |
| `--humidity PERCENT` | `70.0` | Relative humidity. Has a strong, non-linear influence on absorption. The effect is not monotonic — moderate humidity can produce more absorption at certain frequencies than either very dry or very humid air. |
| `--pressure KPA` | `101.325` | Atmospheric pressure. Standard sea-level value. Lower pressure (higher altitude) generally increases absorption per unit distance. |
| `--output FILE` | *(show window)* | Save the plot to a file (PNG, PDF, SVG, etc. — any format matplotlib supports) instead of opening an interactive window. |

## Reading the plot

The plot shows three PSD curves on a log-frequency axis, all referenced to the
peak PSD of the source signal:

- **Source (0 m)** — the original signal with no propagation applied.
- **Listener** — the signal after atmospheric absorption at the listener distance.
- **Target** — the signal after atmospheric absorption at the target distance.

Because `--normalize` is used internally, the curves show only the spectral
shaping from atmospheric absorption. The flat, frequency-independent volume
loss from geometric spreading (inverse-square law) is displayed separately
in a text box in the upper-right corner of the plot, showing both the dB
value and the linear gain coefficient `k` for each distance.

At low frequencies (below ~200 Hz at typical distances), the three curves
nearly overlap — the atmosphere is essentially transparent. Above 1-2 kHz,
the curves diverge as absorption increases with both frequency and distance.
By 10-20 kHz at 500 m, absorption can exceed 100 dB.

## Dependencies

This script imports `design_propagation_filter` from the sibling
`propagation_filter/` directory. Both example directories must be present
for the import to work.

Required packages: `numpy`, `scipy`, `matplotlib`, and `sound_propagation`.
