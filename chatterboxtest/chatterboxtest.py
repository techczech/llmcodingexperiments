#!/usr/bin/env python3
"""
Generate speech with ChatterboxTTS and save it as a .wav file.

Examples
--------
# Default text, default voice, saves output.wav
python tts_generate.py

# Custom text
python tts_generate.py --text "Hello, Oxford!"

# Voice-cloning from a 3-5 s reference clip
python tts_generate.py --voice my_voice.wav

# Custom output file name
python tts_generate.py --out welcome.wav
"""

import argparse
import os
import time
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def get_device(force: str | None = None) -> str:
    """
    Pick a sensible torch device.

    Parameters
    ----------
    force : {"cpu", "cuda", "mps"} or None
        If given, return this value as‑is.
    """
    if force:
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    # ---------- CLI ----------
    parser = argparse.ArgumentParser(description="ChatterboxTTS one‑shot synthesiser")
    parser.add_argument("--text", type=str, default=(
        "Today is the day. I want to move like a titan at dawn, "
        "sweat like a god forging lightning. No more excuses. "
        "From now on, my mornings will be temples of discipline. "
        "I am going to work out like the gods… every damn day."
    ), help="Text to synthesise, or path to a UTF‑8 text file")
    parser.add_argument("--voice", type=str, default=None,
                        help="Path to a short reference WAV for voice cloning (optional)")
    parser.add_argument("--out", type=str, default="output.wav",
                        help="Filename for the generated audio")
    # NEW knobs -------------------------------------
    parser.add_argument("--exaggeration", type=float, default=2.0,
                        help="Expressiveness control (≈0–4)")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        dest="cfg_weight",
                        help="Contrastive guidance strength (0–1)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0–1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--sr", type=int, default=None,
                        help="Override output sample‑rate (Hz)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"],
                        default=None, help="Force computation device")
    # ------------------------------------------------
    args = parser.parse_args()

    # ---------- Device selection ----------
    device = get_device(args.device)

    # ---------- Text source ----------
    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            args.text = f.read()

    # ---------- Deterministic seeding ----------
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ---------- Patch torch.load so everything lands on the right device ----------
    original_torch_load = torch.load

    def patched_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", device)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = patched_load

    # ---------- Model ----------
    model = ChatterboxTTS.from_pretrained(device=device)

    # ---------- Resolve voice prompt (optional) ----------
    prompt_path = args.voice
    if prompt_path and not os.path.isfile(prompt_path):
        print(f"[WARN] '{prompt_path}' not found – using default voice instead.")
        prompt_path = None

    # ---------- Synthesis ----------
    print(f"🎯 Starting synthesis (device: {device})...")
    start_time = time.time()
    
    wav_tensor = model.generate(
        args.text,
        audio_prompt_path=prompt_path,  # None ⇢ default voice
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    
    end_time = time.time()
    synthesis_duration = end_time - start_time
    print(f"⏱️  Synthesis completed in {synthesis_duration:.2f} seconds")

    # ---------- Save ----------
    sample_rate = args.sr or model.sr
    ta.save(args.out, wav_tensor, sample_rate)
    
    # Calculate audio duration
    audio_duration = wav_tensor.shape[-1] / sample_rate
    
    print(f"✅ Saved speech to {args.out} (device: {device})")
    print(f"📊 Audio duration: {audio_duration:.2f}s | Synthesis time: {synthesis_duration:.2f}s | Real-time factor: {synthesis_duration/audio_duration:.2f}x")


if __name__ == "__main__":
    main()