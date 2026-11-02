"""VOXEL-SAM AR — Phase 5: deterministic demo assets + golden geometry.

Renders synthetic 360-degree keyframes (numpy rectangles on black, written as
lossless PNG), runs the Phase 2 silhouette fallback on them exactly like the
real pipeline (PNG files -> load_keyframes -> estimate) and writes or checks
`assets/expected_geometry.json`.

Usage:
  python scripts/make_demo_assets.py                  # (re)generate the golden file
  python scripts/make_demo_assets.py --check           # CI golden test (no writes)
  python scripts/make_demo_assets.py --video           # also build assets/demo.mp4
  python scripts/make_demo_assets.py --keyframes-dir ./kf  # keep PNGs for reuse

Only needs numpy+PIL+pydantic; --video additionally needs an `ffmpeg` binary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseten.infer import estimate_from_silhouettes, load_keyframes

DEMO_VIEWS = 12
DEMO_W = 160
DEMO_H = 120
DEMO_HEIGHT_M = 0.20
DEMO_CX = DEMO_W // 2
NECK_ROWS = (18, 40)
BODY_ROWS = (40, 102)
FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-12
MAX_VIDEO_BYTES = 25 * 1024 * 1024


def render_view(index: int, num_views: int = DEMO_VIEWS) -> np.ndarray:
    """White bottle-like silhouette on black; width oscillates with azimuth."""
    theta = 2.0 * np.pi * index / num_views
    body_hw = round(30 + 8 * np.cos(2 * theta))
    neck_hw = round(12 + 4 * np.cos(2 * theta + 1.0))
    img = np.zeros((DEMO_H, DEMO_W, 3), dtype=np.uint8)
    for (r0, r1), hw in ((NECK_ROWS, neck_hw), (BODY_ROWS, body_hw)):
        img[r0:r1, DEMO_CX - hw : DEMO_CX + hw + 1] = 255
    return img


def write_keyframes(out_dir: Path, num_views: int = DEMO_VIEWS) -> list[int]:
    """Writes keyframe_XX.png + meta.json; returns the frame indices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_views):
        Image.fromarray(render_view(i, num_views)).save(out_dir / f"keyframe_{i:02d}.png")
    indices = list(range(num_views))
    (out_dir / "meta.json").write_text(
        json.dumps({"frames": indices, "synthetic": True, "views": num_views}), encoding="utf-8"
    )
    return indices


def generate_geometry(keyframes_dir: Path, height_m: float = DEMO_HEIGHT_M) -> dict[str, Any]:
    """Runs the real Phase 2 fallback over PNG keyframes; returns parsed JSON."""
    frames, indices = load_keyframes(keyframes_dir, keyframes_dir / "meta.json")
    estimate = estimate_from_silhouettes(frames, indices, height_m)
    return json.loads(estimate.model_dump_json())


def _max_abs_diff(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return float("inf")
    return float(np.abs(aa - bb).max(initial=0.0))


def compare_golden(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Returns human-readable mismatches ([] means the golden test passes)."""
    problems: list[str] = []
    for key in ("source", "units", "coordinate_frame", "notes", "base_color"):
        if expected.get(key) != actual.get(key):
            problems.append(f"{key}: {expected.get(key)!r} != {actual.get(key)!r}")
    for key in ("vertices", "normals", "edges", "faces", "source_keyframes"):
        exp, act = expected.get(key), actual.get(key)
        if (exp is None) != (act is None):
            problems.append(f"{key}: None mismatch ({exp is None} vs {act is None})")
        elif exp is not None and len(exp) != len(act):
            problems.append(f"{key}: len {len(exp)} != {len(act)}")
    if problems:
        return problems
    for key in ("confidence", "estimated_height_m"):
        diff = abs(float(expected[key]) - float(actual[key]))
        tol = FLOAT_ATOL + FLOAT_RTOL * abs(float(expected[key]))
        if diff > tol:
            problems.append(f"{key}: {expected[key]} != {actual[key]} (diff {diff:.3g})")
    for key in ("vertices", "normals"):
        diff = _max_abs_diff(expected[key], actual[key])
        scale = float(np.abs(np.asarray(expected[key], dtype=np.float64)).max(initial=0.0))
        if diff > FLOAT_ATOL + FLOAT_RTOL * scale:
            problems.append(f"{key}: max abs diff {diff:.3g} (scale {scale:.3g})")
    for key in ("edges", "faces", "source_keyframes"):
        if expected[key] != actual[key]:
            problems.append(f"{key}: integer arrays differ")
    return problems


def build_video(keyframes_dir: Path, out_mp4: Path, fps: int = 8) -> Path:
    """Assembles demo.mp4 from the keyframes with ffmpeg (libx264, else mpeg4)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot build demo.mp4.")
    pattern = str(keyframes_dir / "keyframe_%02d.png")
    base = [
        ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
        "-vf", "scale=640:480:flags=neighbor", "-pix_fmt", "yuv420p",
    ]
    attempts = [base + ["-c:v", "libx264"], base + ["-c:v", "mpeg4"]]
    errors: list[str] = []
    for cmd in attempts:
        proc = subprocess.run(
            [*cmd, str(out_mp4)], capture_output=True, text=True, check=False
        )
        if proc.returncode == 0 and out_mp4.exists():
            break
        errors.append(f"{cmd[-1]}: {proc.stderr[-300:]}")
    else:
        raise RuntimeError(f"ffmpeg failed: {' | '.join(errors)}")
    size = out_mp4.stat().st_size
    if size > MAX_VIDEO_BYTES:
        raise RuntimeError(f"demo.mp4 too large: {size} bytes.")
    return out_mp4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate/check Phase 5 demo assets.")
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "expected_geometry.json")
    p.add_argument("--check", action="store_true", help="golden test: compare, do not write")
    p.add_argument("--video", action="store_true", help="also build assets/demo.mp4")
    p.add_argument("--mp4", type=Path, default=ROOT / "assets" / "demo.mp4")
    p.add_argument("--keyframes-dir", type=Path, default=None, help="keep/inspect PNGs")
    p.add_argument("--num-views", type=int, default=DEMO_VIEWS)
    p.add_argument("--height-m", type=float, default=DEMO_HEIGHT_M)
    p.add_argument("--fps", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_views < 4 or args.num_views > 64:
        print("error: --num-views must be in [4, 64].", file=sys.stderr)
        return 2

    if args.keyframes_dir:
        kf_dir = args.keyframes_dir
        write_keyframes(kf_dir, args.num_views)
        actual = generate_geometry(kf_dir, args.height_m)
    else:
        with tempfile.TemporaryDirectory(prefix="voxel_demo_") as tmp:
            kf_dir = Path(tmp)
            write_keyframes(kf_dir, args.num_views)
            if args.video:
                args.mp4.parent.mkdir(parents=True, exist_ok=True)
                try:
                    mp4 = build_video(kf_dir, args.mp4, args.fps)
                except RuntimeError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                print(f"video: {mp4} ({mp4.stat().st_size} bytes)")
            actual = generate_geometry(kf_dir, args.height_m)

    if args.check:
        if not args.out.exists():
            print(f"error: golden file missing: {args.out}", file=sys.stderr)
            return 1
        expected = json.loads(args.out.read_text(encoding="utf-8"))
        problems = compare_golden(expected, actual)
        if problems:
            print("GOLDEN MISMATCH:")
            for line in problems:
                print(f"  - {line}")
            return 1
        n = len(actual["vertices"])
        print(f"golden OK: {n} vertices, {len(actual['faces'] or [])} faces match {args.out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(actual, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(actual['vertices'])} vertices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
