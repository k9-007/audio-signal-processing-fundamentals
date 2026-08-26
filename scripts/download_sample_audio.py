"""Download a public-domain speech sample and save it as
notebooks/sample_audio.flac (16 kHz mono), which the notebook and the
src/ demos expect.

Usage:  python scripts/download_sample_audio.py
"""

import ssl
import sys
import tempfile
import urllib.request
from pathlib import Path

import certifi
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from audio_io import load_audio, save_audio

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "sample_audio.flac"
TARGET_RATE = 16000

# A short clean speech clip used by the official torchaudio tutorials.
ASSET_URL = (
    "https://download.pytorch.org/torchaudio/tutorial-assets/"
    "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)


def main() -> None:
    if TARGET.exists():
        print(f"{TARGET} already exists, nothing to do.")
        return

    context = ssl.create_default_context(cafile=certifi.where())
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        print(f"Downloading {ASSET_URL}")
        with urllib.request.urlopen(ASSET_URL, context=context) as response:
            tmp.write(response.read())
        tmp.flush()
        audio, sr = load_audio(tmp.name)

    if sr != TARGET_RATE:
        # Proper resampling (with anti-aliasing filter) — never naive slicing!
        audio = T.Resample(orig_freq=sr, new_freq=TARGET_RATE)(audio)

    TARGET.parent.mkdir(exist_ok=True)
    save_audio(TARGET, audio, TARGET_RATE)
    duration = len(audio) / TARGET_RATE
    print(f"Saved {TARGET}  ({duration:.1f} s at {TARGET_RATE} Hz)")


if __name__ == "__main__":
    main()
