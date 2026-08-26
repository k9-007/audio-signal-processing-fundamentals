"""Demo 1 — Frequency basics and the Fourier transform.

Covers (see notes sections 3, 5-7, 10, 11):
  1. Sinusoids at different frequencies (what "Hz" means visually)
  2. FFT of a pure 20 Hz tone -> a single spike at 20 Hz
  3. FFT of a sum of sinusoids -> one spike per ingredient ("un-mixing paint")
  4. Inverse FFT -> perfect reconstruction of the original signal

Run:  python src/fourier_demo.py
Plots are saved to outputs/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signals import ensure_output_dir, magnitude_spectrum, make_time_axis, sinusoid, sum_of_sinusoids

SAMPLING_RATE = 1000  # samples per second
DURATION_S = 1.0


def plot_sinusoid_basics(out_dir) -> None:
    t = make_time_axis(DURATION_S, SAMPLING_RATE)
    plt.figure(figsize=(10, 5))
    for f in [1, 4, 8]:
        plt.plot(t, sinusoid(f, t), label=f"{f} Hz = {f} cycles/second")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Frequency = how many cycles fit into one second")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "01_sinusoid_basics.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_fft_of_pure_tone(out_dir) -> None:
    t = make_time_axis(DURATION_S, SAMPLING_RATE)
    y = sinusoid(20, t)
    freqs, mags = magnitude_spectrum(y, SAMPLING_RATE)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, y)
    axes[0].set(title="Time domain: 20 Hz sinusoid", xlabel="Time (s)", ylabel="Amplitude")
    axes[1].plot(freqs, mags)
    axes[1].set(title="Frequency domain: FFT finds the single 20 Hz spike",
                xlabel="Frequency (Hz)", ylabel="Magnitude")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "02_fft_pure_tone.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fft_of_mixture(out_dir) -> None:
    t = make_time_axis(DURATION_S, SAMPLING_RATE)
    ingredients = [5, 20, 40, 250]
    y = sum_of_sinusoids(ingredients, t)
    freqs, mags = magnitude_spectrum(y, SAMPLING_RATE)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, y)
    axes[0].set(title=f"Mixed signal: sum of sinusoids at {ingredients} Hz",
                xlabel="Time (s)", ylabel="Amplitude")
    axes[1].plot(freqs, mags)
    axes[1].set(title="FFT un-mixes the paint: one spike per ingredient",
                xlabel="Frequency (Hz)", ylabel="Magnitude")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "03_fft_sum_of_sinusoids.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_inverse_fft(out_dir) -> None:
    t = make_time_axis(DURATION_S, SAMPLING_RATE)
    y = sum_of_sinusoids([5, 20, 40], t)
    y_back = np.fft.ifft(np.fft.fft(y)).real

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, y)
    axes[0].set(title="Original signal", ylabel="Amplitude")
    axes[1].plot(t, y_back, color="tab:orange")
    axes[1].set(title="After FFT then inverse FFT — identical (lossless round trip)",
                xlabel="Time (s)", ylabel="Amplitude")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "04_inverse_fft.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    max_err = np.max(np.abs(y - y_back))
    print(f"Max reconstruction error after FFT -> iFFT: {max_err:.2e} (floating-point dust)")


if __name__ == "__main__":
    out_dir = ensure_output_dir()
    plot_sinusoid_basics(out_dir)
    plot_fft_of_pure_tone(out_dir)
    plot_fft_of_mixture(out_dir)
    plot_inverse_fft(out_dir)
    print(f"Saved plots 01-04 to {out_dir}")
