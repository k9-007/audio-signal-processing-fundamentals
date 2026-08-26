"""Demo 4 — Mel spectrograms and waveform reconstruction with Griffin-Lim.

Covers (see notes sections 17-19):
  1. The mel scale curve (Hz -> mel): perception is logarithmic
  2. Mel spectrogram with torchaudio
  3. Reconstruction: mel -> linear spectrogram (InverseMelScale)
     -> waveform (Griffin-Lim iterative phase recovery)
  4. Saves the reconstructed audio so you can hear the "slightly robotic"
     quality that motivates neural vocoders like HiFi-GAN.

Requires the sample audio file. Get it with:
  python scripts/download_sample_audio.py

Run:  python src/mel_reconstruction_demo.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchaudio.transforms as T

from audio_io import load_audio, save_audio
from signals import ensure_output_dir

AUDIO_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "sample_audio.flac"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128


def plot_mel_scale_curve(out_dir) -> None:
    freqs = np.linspace(20, 20000, 500)
    mels = 2595 * np.log10(1 + freqs / 700)

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, mels)
    plt.title("Mel scale: equal mel steps = equal *perceived* pitch steps")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mel")
    plt.grid(True)
    plt.annotate("steep here:\nears are sharp at low freqs", xy=(1500, 1180), fontsize=9)
    plt.annotate("flat here:\nears barely notice changes", xy=(11000, 3000), fontsize=9)
    plt.savefig(out_dir / "09_mel_scale_curve.png", dpi=120, bbox_inches="tight")
    plt.close()


def mel_spectrogram_and_reconstruction(out_dir) -> None:
    audio, sr = load_audio(AUDIO_PATH)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, mel_scale="slaney",
    )
    mel_spec = mel_transform(audio)
    mel_db = T.AmplitudeToDB()(mel_spec)
    print(f"Mel spectrogram shape: {tuple(mel_spec.shape)}  ({N_MELS} mel bins x time frames)")

    # Step 1: mel -> linear spectrogram (approximate: mel binning was many-to-one)
    inverse_mel = T.InverseMelScale(
        n_stft=N_FFT // 2 + 1, n_mels=N_MELS,
        sample_rate=SAMPLE_RATE, mel_scale="slaney",
    )
    linear_spec = inverse_mel(mel_spec)

    # Step 2: magnitude-only spectrogram -> waveform via iterative phase recovery
    griffin_lim = T.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=100)
    reconstructed = griffin_lim(linear_spec)

    out_wav = out_dir / "reconstructed_griffin_lim.wav"
    save_audio(out_wav, reconstructed, SAMPLE_RATE)
    print(f"Wrote {out_wav} — compare it by ear with the original; expect a slightly robotic sound.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(mel_db.numpy(), cmap="inferno", origin="lower", aspect="auto")
    axes[0].set(title=f"Mel spectrogram ({N_MELS} bins, dB)", xlabel="Time frame", ylabel="Mel bin")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(T.AmplitudeToDB()(linear_spec).numpy(), cmap="inferno",
                         origin="lower", aspect="auto")
    axes[1].set(title=f"Recovered linear spectrogram ({N_FFT // 2 + 1} bins)",
                xlabel="Time frame", ylabel="Frequency bin")
    fig.colorbar(im1, ax=axes[1])

    axes[2].plot(reconstructed.numpy(), linewidth=0.5)
    axes[2].set(title="Griffin-Lim reconstructed waveform", xlabel="Sample", ylabel="Amplitude")
    fig.tight_layout()
    fig.savefig(out_dir / "10_mel_and_reconstruction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if not AUDIO_PATH.exists():
        raise SystemExit(f"Missing {AUDIO_PATH}. Run: python scripts/download_sample_audio.py")
    out_dir = ensure_output_dir()
    plot_mel_scale_curve(out_dir)
    mel_spectrogram_and_reconstruction(out_dir)
    print(f"Saved plots 09-10 and reconstructed audio to {out_dir}")
