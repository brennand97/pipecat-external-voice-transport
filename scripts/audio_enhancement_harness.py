#!/usr/bin/env python3
"""Measure the production audio enhancer against 16 kHz mono PCM fixtures.

Examples:
  python scripts/audio_enhancement_harness.py benchmark \
    --input /secure/fixtures/far-field.pcm --enhancer gtcrn
  python scripts/audio_enhancement_harness.py enhance \
    --input /secure/fixtures/far-field.pcm --enhancer gtcrn --output /tmp/out.pcm

Fixtures are intentionally supplied outside the repository. This script never
contacts OpenAI, Home Assistant, or Music Assistant and contains no credential
handling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import time
from pathlib import Path

from voice_transport.audio_enhancement import (
    AudioInputEnhancementConfig,
    create_audio_enhancer,
)

FRAME_BYTES = 2_560  # 80 ms, signed 16-bit mono at 16 kHz


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "enhance"):
        command = subcommands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--enhancer", choices=("disabled", "gtcrn"), required=True)
        command.add_argument(
            "--model", type=Path, default=Path("models/gtcrn_simple.onnx")
        )
        command.add_argument("--gain-db", type=float, default=0.0)
        command.add_argument("--limiter-dbfs", type=float, default=-3.0)
        command.add_argument("--json", type=Path)
        if name == "benchmark":
            command.add_argument("--repeat", type=int, default=10)
        else:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--force", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> tuple[bytes, dict[str, object]]:
    source = args.input.read_bytes()
    if len(source) % 2:
        raise ValueError("input must be signed 16-bit PCM")
    config = AudioInputEnhancementConfig(
        enhancer=args.enhancer,
        output_gain_db=args.gain_db,
        limiter_dbfs=args.limiter_dbfs,
    )
    enhancer = create_audio_enhancer(config, gtcrn_model_path=str(args.model))
    timings: list[float] = []
    output = bytearray()
    try:
        for offset in range(0, len(source), FRAME_BYTES):
            frame = source[offset : offset + FRAME_BYTES]
            started = time.perf_counter()
            processed, _ = await enhancer.process(frame, 16_000, 1)
            timings.append((time.perf_counter() - started) * 1000)
            output.extend(processed)
    finally:
        await enhancer.close()
    ordered = sorted(timings)

    def percentile(fraction: float) -> float:
        if not ordered:
            return 0.0
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]

    elapsed_audio_ms = len(source) / 32  # 16 kHz x 2 bytes = 32 bytes/ms
    return bytes(output), {
        "enhancer": args.enhancer,
        "input_bytes": len(source),
        "output_bytes": len(output),
        "frames": len(timings),
        "inference_ms_p50": percentile(0.50),
        "inference_ms_p95": percentile(0.95),
        "inference_ms_p99": percentile(0.99),
        "realtime_factor": sum(timings) / elapsed_audio_ms if elapsed_audio_ms else 0.0,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


async def main() -> None:
    args = arguments()
    if args.command == "benchmark" and args.repeat < 1:
        raise ValueError("--repeat must be positive")
    runs = []
    output = b""
    for _ in range(getattr(args, "repeat", 1)):
        output, result = await run(args)
        runs.append(result)
    if args.command == "enhance":
        if args.output.exists() and not args.force:
            raise FileExistsError("refusing to overwrite output without --force")
        args.output.write_bytes(output)
    result: dict[str, object] = {"runs": runs, "input": str(args.input)}
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    asyncio.run(main())
