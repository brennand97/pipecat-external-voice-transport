"""Reproducible synthetic kitchen microphone corpus. No personal recordings.

Uses the inventoried billable OpenAI test key only to synthesize fixed public
phrases. Distortions and labels are deterministic, not actual tablet calibration.
"""

import argparse
import io
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

RATE = 16000


def read_wav(data):
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getsampwidth() == 2
        audio = (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                np.float64
            )
            / 32768
        )
        audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1)
        return resample_poly(audio, RATE, wav.getframerate())


def save(path, audio):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def rms(audio):
    return float(np.sqrt(np.mean(audio**2)))


def kitchen(n, seed=1729):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    fan = sosfilt(butter(2, 350, fs=RATE, output="sos"), rng.normal(size=n))
    water = sosfilt(
        butter(2, 800, fs=RATE, btype="highpass", output="sos"), rng.normal(size=n)
    )
    result = fan + 0.3 * water + 0.15 * np.sin(2 * np.pi * 100 * t)
    for second in (2, 4, 6, 9, 11, 13):
        start = second * RATE
        if start < n:
            length = min(1600, n - start)
            result[start : start + length] += (
                rng.normal(size=length) * np.exp(-np.arange(length) / 250) * 3
            )
    return result / max(rms(result), 1e-12)


def corpus(directory, key_file, continuous=False, source_directory=None):
    from openai import OpenAI

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    client = None
    source_directory = source_directory or directory
    source_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    phrases = {
        "wake": "Hey Jarvis.",
        "command": "Set a twenty second pasta timer.",
        "background": "Could you pass the plate? The refrigerator door is open.",
        "end": "Please end this conversation.",
    }
    spoken = {}
    for name, phrase in phrases.items():
        path = source_directory / (name + "-source.wav")
        if not path.exists():
            if client is None:
                client = OpenAI(api_key=key_file.read_text().strip())
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=phrase,
                response_format="wav",
            )
            path.write_bytes(response.content)
        spoken[name] = read_wav(path.read_bytes())
    if continuous:
        for name in ("wake", "command"):
            audio = spoken[name]
            voiced = np.flatnonzero(abs(audio) > 0.003)
            spoken[name] = audio[
                max(0, voiced[0] - 320) : min(len(audio), voiced[-1] + 321)
            ]
    # Allow the real browser/worker/model to initialize before the utterance.
    # SNR uses spoken-region RMS, not whole-clip RMS diluted by padding.
    clean = np.concatenate(
        [
            np.zeros(RATE * 8),
            spoken["wake"],
            np.zeros(int(RATE * (0.12 if continuous else 1))),
            spoken["command"],
            np.zeros(RATE * 4),
        ]
    )
    reference = rms(np.concatenate([spoken["wake"], spoken["command"]]))
    variants = {
        "clean": clean,
        "distant": clean * 0.1 + kitchen(len(clean)) * reference * 0.01,
        "kitchen20": clean + kitchen(len(clean)) * reference * 0.1,
        "kitchen10": clean + kitchen(len(clean)) * reference / (10**0.5),
        "kitchen0": clean + kitchen(len(clean)) * reference,
        "clipped": np.clip(clean * 4, -0.08, 0.08),
        "bad_tablet": np.round(
            sosfilt(
                butter(3, [300, 3500], fs=RATE, btype="bandpass", output="sos"), clean
            )
            * 0.2
            * 128
        )
        / 128
        + kitchen(len(clean)) * reference * 0.02,
    }
    reverberant = clean.copy()
    for delay, gain in [(480, 0.5), (1440, 0.3), (2400, 0.15)]:
        reverberant[delay:] += clean[:-delay] * gain
    variants["reverb_kitchen10"] = reverberant + kitchen(len(clean)) * rms(
        reverberant
    ) / (10**0.5)
    competing = np.resize(spoken["background"], len(clean))
    competing *= reference / max(rms(competing), 1e-12) / (10**0.5)
    variants["interfering_voice"] = clean + competing
    variants["noise_only"] = kitchen(len(clean)) * 0.03
    variants["unrelated_speech"] = np.concatenate(
        [np.zeros(RATE * 8), spoken["background"], np.zeros(RATE * 6)]
    )
    manifest = []
    for name, audio in variants.items():
        path = directory / (name + ".wav")
        save(path, audio)
        manifest.append(
            {
                "name": name,
                "path": str(path),
                "wake_expected": name not in ("noise_only", "unrelated_speech"),
                "wake_start_seconds": 8,
                "rms_dbfs": round(20 * np.log10(max(rms(audio), 1e-12)), 2),
                "duration_seconds": round(len(audio) / RATE, 2),
            }
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Generated", len(manifest), "synthetic acoustic cases in", directory, flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="120ms wake/command gap with trimmed synthesis padding",
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        help="Reuse cached public synthesized phrases without billable TTS requests",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".local/state/agents/.openai-test-key",
    )
    args = parser.parse_args()
    corpus(args.directory, args.api_key_file, args.continuous, args.source_directory)
