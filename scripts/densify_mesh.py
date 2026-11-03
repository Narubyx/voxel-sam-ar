"""Densify + smooth a watertight run mesh for display.

Takes an agent-loop run's model.glb (often coarse low-poly from the VLM),
applies Loop subdivision + Taubin smoothing, re-enforces the exact Y span,
re-verifies watertight/winding, and re-exports .glb + .usdz with the same
PBR material. Purely a display refinement: topology intent comes from the run.

Usage:
    python scripts/densify_mesh.py --glb <run>/model.glb --out-dir <dir> \
        --height-m 0.0762 --base-color 0.6431 0.1608 0.2588
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.mesh_tools import check, repair, save_glb, to_usdz


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--height-m", type=float, required=True)
    p.add_argument("--base-color", type=float, nargs=3, default=None)
    p.add_argument("--subdiv", type=int, default=2)
    p.add_argument("--smooth-iters", type=int, default=4)
    args = p.parse_args(argv)

    import numpy as np
    import open3d as o3d
    import trimesh

    mesh = trimesh.load(args.glb, force="mesh")
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    # Midpoint (not Loop): splits preserve triangle aspect, so the sharp rim
    # edge never grows needles; Taubin then rounds the facets.
    for _ in range(max(0, args.subdiv)):
        tm = tm.subdivide_midpoint(number_of_iterations=1)
    tm = tm.filter_smooth_taubin(number_of_iterations=max(0, args.smooth_iters))
    v = np.asarray(tm.vertices)
    span = v[:, 1].max() - v[:, 1].min()
    if not span > 0:
        print("zero height span after smoothing.", file=sys.stderr)
        return 1
    v = v * (args.height_m / span)
    out_mesh = trimesh.Trimesh(v, np.asarray(tm.triangles), process=True)
    out_mesh = repair(out_mesh)

    report = check(out_mesh)
    stats = {"verts": len(out_mesh.vertices), "faces": len(out_mesh.faces)}
    print(json.dumps({"ok": report.ok, "errors": report.errors, **stats}))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "model.glb"
    save_glb(out_mesh, glb_path, base_color=args.base_color)
    try:
        converter = to_usdz(glb_path, out_dir / "model.usdz", base_color=args.base_color)
    except Exception as exc:  # noqa: BLE001 - report converter failure honestly
        print(json.dumps({"usdz_ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"usdz_ok": True, "converter": converter, "glb": str(glb_path)}))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
