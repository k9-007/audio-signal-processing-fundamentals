"""Demo 2 — Nyquist limit and aliasing.

Covers (see notes sections 8-9):
  A 40 Hz tone sampled at 200 samples/s (Nyquist = 100 Hz, so it fits).
  - Downsample by 4 -> 50 samples/s -> Nyquist = 25 Hz < 40 Hz
    => the 40 Hz energy FOLDS BACK and shows up at |50 - 40| = 10 Hz. Aliasing!
  - Downsample by 2 -> 100 samples/s -> Nyquist = 50 Hz > 40 Hz
    => the spike stays at 40 Hz. No aliasing.

NOTE: we downsample by naive slicing (y[::k]) ON PURPOSE, to expose the
problem. Real code must low-pass filter first, e.g.
torchaudio.transforms.Resample, which has the anti-aliasing filter built in.

Run:  python src/aliasing_demo.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signals import ensure_output_dir, magnitude_spectrum, make_time_axis, sinusoid

ORIGINAL_RATE = 200  # samples per second
SIGNAL_FREQ = 40     # cycles per second


def downsample_and_plot(factor: int, out_dir) -> None:
    t_high = make_time_axis(1.0, ORIGINAL_RATE)
    y_high = sinusoid(SIGNAL_FREQ, t_high)

    # Naive decimation: keep every `factor`-th sample (no anti-aliasing filter!)
    t_low, y_low = t_high[::factor], y_high[::factor]
    low_rate = ORIGINAL_RATE // factor
    nyquist_low = low_rate / 2

    freqs_high, mags_high = magnitude_spectrum(y_high, ORIGINAL_RATE)
    freqs_low, mags_low = magnitude_spectrum(y_low, low_rate)

    aliased = nyquist_low < SIGNAL_FREQ
    verdict = (f"ALIASED: {SIGNAL_FREQ} Hz folds to {abs(low_rate - SIGNAL_FREQ)} Hz"
               if aliased else "no aliasing, spike stays at 40 Hz")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(t_high, y_high)
    axes[0, 0].set(title=f"Original: {SIGNAL_FREQ} Hz @ {ORIGINAL_RATE} samples/s")
    axes[0, 1].plot(freqs_high, mags_high)
    axes[0, 1].set(title="Original FFT (spike at 40 Hz)", xlabel="Frequency (Hz)")

    axes[1, 0].plot(t_low, y_low, "o-", color="tab:red", markersize=3)
    axes[1, 0].set(title=f"Downsampled x{factor}: {low_rate} samples/s (Nyquist = {nyquist_low:.0f} Hz)",
                   xlabel="Time (s)")
    axes[1, 1].plot(freqs_low, mags_low, color="tab:red")
    axes[1, 1].set(title=f"Downsampled FFT — {verdict}", xlabel="Frequency (Hz)")

    for ax in axes.flat:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / f"05_aliasing_downsample_x{factor}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"factor {factor}: new rate {low_rate} Hz, Nyquist {nyquist_low:.0f} Hz -> {verdict}")


if __name__ == "__main__":
    out_dir = ensure_output_dir()
    downsample_and_plot(factor=4, out_dir=out_dir)  # aliases
    downsample_and_plot(factor=2, out_dir=out_dir)  # safe
    print(f"Saved plots 05 to {out_dir}")
