"""Audio file I/O that works everywhere.

torchaudio >= 2.9 delegates load/save to torchcodec, which requires system
FFmpeg libraries. `soundfile` reads/writes WAV and FLAC natively with no
extra system dependencies, so we use it here and convert to torch tensors.
"""

from pathlib import Path

import soundfile as sf
import torch


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load an audio file -> (1-D float32 tensor, sampling_rate)."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    return torch.from_numpy(data), sr


def save_audio(path: str | Path, waveform: torch.Tensor, sampling_rate: int) -> None:
    """Save a 1-D tensor as an audio file (format inferred from extension)."""
    sf.write(str(path), waveform.squeeze().numpy(), sampling_rate)
