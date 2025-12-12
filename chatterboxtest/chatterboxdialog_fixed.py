#!/usr/bin/env python3
"""
chatterboxdialog.py
-------------------
Minimal self‑contained demo that generates a two‑speaker conversation
with Chatterbox‑TTS.

Prerequisites
~~~~~~~~~~~~~
pip install chatterbox-tts torch torchaudio pydub

Put two short reference clips (≈ 5 s each) in the same directory:

    voice_m.wav   # male‑sounding speaker
    voice_f.wav   # female‑sounding speaker

Run:

    python chatterboxdialog.py

The script will output `conversation.wav` and optionally `conversation.mp3`.
"""

from pathlib import Path
import argparse
import time
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# ── Patch torch.load ───────────────────────────────────────────
# Some Chatterbox checkpoints were saved on CUDA devices.  On a
# Mac (M‑series / MPS) or any CPU‑only environment, attempting to
# load them will raise:
#   RuntimeError: Attempting to deserialize object on a CUDA device …
#
# We intercept torch.load early and force all storages to map to
# CPU first; the model is moved to the desired device afterwards.
_orig_torch_load = torch.load

def _torch_load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_cpu


def parse_dialog(path: Path) -> list[tuple[str, str, float]]:
    """
    Parse a dialog file where each non‑blank line looks like:
        [male 0.4] Platform three. Midnight express pulls in ten minutes.
    Returns a list of tuples: (text, speaker, exaggeration).
    Lines starting with '#' are treated as comments.
    """
    items: list[tuple[str, str, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if not (raw.startswith("[") and "]" in raw):
                raise ValueError(f"Malformed dialog line: {raw}")
            tag, text = raw.split("]", 1)
            tag = tag[1:].strip()          # drop leading '['
            try:
                speaker, exag_str = tag.split()
                exag = float(exag_str)
            except ValueError:             # speaker or exag missing
                raise ValueError(f"Bad tag format in line: {raw}")
            items.append((text.strip(), speaker.lower(), exag))
    return items


def pick_device() -> str:
    """Select CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_segments(model: ChatterboxTTS, lines: list[tuple[str, str, float]]) -> list[torch.Tensor]:
    """Generate each utterance and return a list of wave tensors."""
    voices = {
        "male": "voice_m.wav",
        "female": "voice_f.wav",
    }

    cfg_weight = 0.4
    temperature = 0.6

    segments: list[torch.Tensor] = []
    total_synthesis_time = 0.0
    total_audio_duration = 0.0
    
    print(f"🎭 Starting dialog synthesis with {len(lines)} utterances...")
    
    for i, (text, speaker, exag) in enumerate(lines, 1):
        print(f"🎯 Synthesizing utterance {i}/{len(lines)} ({speaker})...")
        start_time = time.time()
        
        wav = model.generate(
            text,
            audio_prompt_path=voices[speaker],
            exaggeration=exag,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        
        end_time = time.time()
        utterance_time = end_time - start_time
        total_synthesis_time += utterance_time
        
        # Calculate audio duration for this segment
        audio_duration = wav.shape[-1] / model.sr
        total_audio_duration += audio_duration
        
        print(f"⏱️  Utterance {i} completed in {utterance_time:.2f}s (audio: {audio_duration:.2f}s)")
        segments.append(wav)
    
    # Print summary statistics
    avg_synthesis_time = total_synthesis_time / len(lines)
    real_time_factor = total_synthesis_time / total_audio_duration
    
    print(f"📊 Dialog synthesis complete:")
    print(f"   Total synthesis time: {total_synthesis_time:.2f}s")
    print(f"   Total audio duration: {total_audio_duration:.2f}s")
    print(f"   Average per utterance: {avg_synthesis_time:.2f}s")
    print(f"   Real-time factor: {real_time_factor:.2f}x")
    
    return segments


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> bool:
    """
    Convert WAV file to MP3 using pydub.
    
    Parameters
    ----------
    wav_path : Path
        Path to the input WAV file
    mp3_path : Path
        Path for the output MP3 file
    bitrate : str
        MP3 bitrate (e.g., "128k", "192k", "320k")
    
    Returns
    -------
    bool
        True if conversion was successful, False otherwise
    """
    if not HAS_PYDUB:
        print("⚠️  pydub not available. Install with: pip install pydub")
        return False
    
    try:
        print(f"🎵 Converting {wav_path.name} to MP3...")
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(mp3_path), format="mp3", bitrate=bitrate)
        print(f"✅ MP3 saved: {mp3_path}")
        return True
    except Exception as e:
        print(f"❌ MP3 conversion failed: {e}")
        return False


def main() -> None:
    import sys
    parser = argparse.ArgumentParser(description="Generate a two‑speaker Chatterbox dialog")
    parser.add_argument("--dialog", default="dialog.txt", help="Path to dialog markup file")
    parser.add_argument("--mp3", action="store_true", help="Also create MP3 version")
    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (e.g., 128k, 192k, 320k)")
    parser.add_argument("--out", default="conversation", help="Output filename (without extension)")
    args = parser.parse_args()

    device = pick_device()
    print(f"🖥️  Using device: {device}")

    # Start overall timing
    script_start_time = time.time()

    model = ChatterboxTTS.from_pretrained(device=device)
    sr = model.sr  # sample‑rate (e.g. 24 000 Hz)

    dialog_lines = parse_dialog(Path(args.dialog))
    segments = build_segments(model, dialog_lines)

    # Insert 250 ms silence between turns
    pause = torch.zeros(1, int(0.25 * sr))
    dialogue = torch.cat([torch.cat([seg, pause], dim=1) for seg in segments], dim=1)

    # Save WAV file
    wav_outfile = Path(f"{args.out}.wav")
    ta.save(str(wav_outfile), dialogue, sr)
    
    # Calculate final statistics
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    final_audio_duration = dialogue.shape[-1] / sr
    
    print(f"✅ Saved WAV: {wav_outfile.absolute()}")
    
    # Convert to MP3 if requested
    if args.mp3:
        mp3_outfile = Path(f"{args.out}.mp3")
        mp3_success = convert_wav_to_mp3(wav_outfile, mp3_outfile, args.bitrate)
        if mp3_success:
            # Get file sizes for comparison
            wav_size = wav_outfile.stat().st_size / (1024 * 1024)  # MB
            mp3_size = mp3_outfile.stat().st_size / (1024 * 1024)  # MB
            compression_ratio = wav_size / mp3_size if mp3_size > 0 else 0
            print(f"📊 File sizes: WAV {wav_size:.1f}MB → MP3 {mp3_size:.1f}MB (compression: {compression_ratio:.1f}x)")
    
    print(f"🎬 Complete dialog: {final_audio_duration:.2f}s audio generated in {total_script_time:.2f}s total time")


if __name__ == "__main__":
    main()
