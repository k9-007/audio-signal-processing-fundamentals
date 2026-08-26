"""Small helpers shared by the demo scripts."""

from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def make_time_axis(duration_s: float, sampling_rate: int) -> np.ndarray:
    """Time stamps for `duration_s` seconds sampled at `sampling_rate` samples/s."""
    n_samples = int(duration_s * sampling_rate)
    return np.linspace(0.0, duration_s, n_samples, endpoint=False)


def sinusoid(freq_hz: float, t: np.ndarray, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """y(t) = A * sin(2*pi*f*t + phase)"""
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)


def sum_of_sinusoids(freqs_hz: list[float], t: np.ndarray) -> np.ndarray:
    """Mix several pure tones into one signal (the 'mixed paint')."""
    return np.sum([sinusoid(f, t) for f in freqs_hz], axis=0)


def magnitude_spectrum(y: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """FFT of `y`, returning (positive frequencies, magnitudes) only.

    Negative frequencies are conjugate mirrors of the positive ones for
    real signals, so we drop them.
    """
    y_fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=1.0 / sampling_rate)
    half = len(freqs) // 2
    return freqs[:half], np.abs(y_fft)[:half]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR
