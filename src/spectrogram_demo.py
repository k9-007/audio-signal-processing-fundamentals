"""Demo 3 — STFT, spectrograms, and the time-frequency trade-off.

Covers (see notes sections 12-16):
  1. STFT with scipy -> magnitude AND phase images (phase looks like noise)
  2. The window-size trade-off: tiny window vs huge window vs sweet spot
  3. torchaudio Spectrogram + AmplitudeToDB (why dB scaling is essential)

Requires the sample audio file. Get it with:
  python scripts/download_sample_audio.py

Run:  python src/spectrogram_demo.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchaudio.transforms as T
from scipy.signal import stft

from audio_io import load_audio
from signals import ensure_output_dir

AUDIO_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "sample_audio.flac"


def scipy_stft_magnitude_and_phase(audio, sr, out_dir) -> None:
    f, t_spec, zxx = stft(audio.numpy(), fs=sr, nperseg=400, noverlap=200)
    magnitude, phase = np.abs(zxx), np.angle(zxx)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    m0 = axes[0].pcolormesh(t_spec, f, 20 * np.log10(magnitude + 1e-10), shading="gouraud")
    axes[0].set(title="Magnitude (dB) — clear structure, this is what we use",
                xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(m0, ax=axes[0], format="%+2.0f dB")

    m1 = axes[1].pcolormesh(t_spec, f, phase, shading="gouraud")
    axes[1].set(title="Phase — looks like noise, so analysis ignores it",
                xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(m1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(out_dir / "06_stft_magnitude_vs_phase.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def window_size_tradeoff(audio, sr, out_dir) -> None:
    configs = [
        (16, "Tiny window (16): great time res, only 9 freq bins"),
        (400, "Default window (400): the sweet spot for speech"),
        (10000, "Huge window (10000): great freq res, time smeared"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (nperseg, title) in zip(axes, configs):
        f, t_spec, zxx = stft(audio.numpy(), fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
        ax.pcolormesh(t_spec, f, 20 * np.log10(np.abs(zxx) + 1e-10), shading="auto")
        ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.suptitle("The time-frequency resolution trade-off")
    fig.tight_layout()
    fig.savefig(out_dir / "07_window_size_tradeoff.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def torchaudio_spectrogram_db(audio, out_dir) -> None:
    spec = T.Spectrogram(n_fft=400, hop_length=200)(audio)
    db_spec = T.AmplitudeToDB()(spec)
    print(f"Spectrogram shape: {tuple(spec.shape)}  "
          f"(n_fft//2 + 1 = {400 // 2 + 1} frequency bins x time frames)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(spec.numpy(), cmap="inferno", origin="lower", aspect="auto", interpolation="none")
    axes[0].set(title="Raw amplitudes — nearly black, useless to look at",
                xlabel="Time frame", ylabel="Frequency bin")
    im = axes[1].imshow(db_spec.numpy(), cmap="inferno", origin="lower", aspect="auto", interpolation="none")
    axes[1].set(title="Same data in decibels — structure appears",
                xlabel="Time frame", ylabel="Frequency bin")
    fig.colorbar(im, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_dir / "08_db_scaling.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if not AUDIO_PATH.exists():
        raise SystemExit(f"Missing {AUDIO_PATH}. Run: python scripts/download_sample_audio.py")
    out_dir = ensure_output_dir()
    audio, sr = load_audio(AUDIO_PATH)
    scipy_stft_magnitude_and_phase(audio, sr, out_dir)
    window_size_tradeoff(audio, sr, out_dir)
    torchaudio_spectrogram_db(audio, out_dir)
    print(f"Saved plots 06-08 to {out_dir}")
