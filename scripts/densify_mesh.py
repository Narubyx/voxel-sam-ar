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


def _ordered_boundary_loop(mesh):
    """Single boundary cycle as an ordered vertex list. Raises if not one."""
    import numpy as np
    from collections import Counter

    edges = [tuple(sorted(e)) for e in np.asarray(mesh.edges_sorted)]
    bnd = [e for e, n in Counter(edges).items() if n == 1]
    adj: dict[int, list[int]] = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj or any(len(v) != 2 for v in adj.values()):
        raise ValueError(f"rim is not a single cycle ({len(adj)} verts).")
    start = next(iter(adj))
    loop, prev, cur = [start], -1, start
    while True:
        nxt = adj[cur][0] if adj[cur][1] == prev else adj[cur][1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    if len(loop) != len(adj):
        raise ValueError("rim boundary has multiple components.")
    return loop


def graft_hollow_top(mesh, wall_m=0.004, floor_lift_m=0.012):
    """Swaps the flat top cap for a cup profile stitched to the rim loop."""
    import numpy as np
    import trimesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    y_top, y_base = v[:, 1].max(), v[:, 1].min()
    cap = (v[f][:, :, 1] > y_top - 0.0005).all(axis=1)
    if not cap.any():
        raise ValueError("no flat top cap found to replace.")
    open_mesh = trimesh.Trimesh(v, f[~cap], process=False)
    loop = _ordered_boundary_loop(open_mesh)

    ov = np.asarray(open_mesh.vertices)
    ring = ov[loop]
    rr = np.linalg.norm(ring[:, [0, 2]], axis=1, keepdims=True)
    rr = np.maximum(rr, 1e-9)
    inward = ring[:, [0, 2]] / rr
    inner_xz = ring[:, [0, 2]] - wall_m * inward
    y_floor = y_base + floor_lift_m
    n = len(loop)
    base = len(ov)
    # Wall in 4 stacked bands (one 64 mm-tall quad ring would be needles).
    levels = 5
    ys = np.linspace(y_top, y_floor, levels)
    wall_rings = []
    for lv in range(levels):
        wall_rings.append(np.column_stack([inner_xz, np.full(n, ys[lv])]))
    new_v = np.vstack([ov] + wall_rings + [[[0.0, y_floor, 0.0]]])
    floor_idx = base + levels * n
    faces = [list(tri) for tri in np.asarray(open_mesh.faces)]
    for k in range(n):
        k2 = (k + 1) % n
        o0, o1 = loop[k], loop[k2]
        t0, t1 = base + k, base + k2
        faces.append([t0, o0, o1])  # lip, faces up
        faces.append([t0, o1, t1])
        for lv in range(levels - 1):
            a0, a1 = base + lv * n + k, base + lv * n + k2
            b0, b1 = base + (lv + 1) * n + k, base + (lv + 1) * n + k2
            faces.append([a0, a1, b1])  # inner wall, faces inward
            faces.append([a0, b1, b0])
        f0 = base + (levels - 1) * n + k
        f1 = base + (levels - 1) * n + k2
        faces.append([floor_idx, f0, f1])  # floor, faces up
    return trimesh.Trimesh(new_v, faces, process=True)


def drill_handle_hole(mesh, center=(0.040, 0.0, 0.0), radius_m=0.007):
    """Boolean-drills a Z-axis tunnel for the C-handle opening."""
    import numpy as np
    import trimesh

    v = np.asarray(mesh.vertices)
    half = float(v[:, 2].max() - v[:, 2].min()) / 2 + 0.010
    # trimesh cylinders already span Z; just center on the handle bulge.
    cyl = trimesh.creation.cylinder(radius=radius_m, height=2 * half, sections=24)
    cyl.apply_translation(np.asarray(center, dtype=float))
    try:
        out = trimesh.boolean.difference([mesh, cyl], engine="manifold")
    except Exception as exc:
        raise RuntimeError(f"handle-hole boolean failed: {exc}") from exc
    if out is None or len(getattr(out, "faces", [])) == 0:
        raise RuntimeError("handle-hole boolean produced an empty mesh.")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--height-m", type=float, required=True)
    p.add_argument("--base-color", type=float, nargs=3, default=None)
    p.add_argument("--subdiv", type=int, default=2)
    p.add_argument("--smooth-iters", type=int, default=6)
    p.add_argument("--hollow-top", action="store_true",
                   help="replace the flat top cap with a cup profile (lip+wall+floor)")
    p.add_argument("--wall-mm", type=float, default=4.0)
    p.add_argument("--floor-lift-mm", type=float, default=12.0)
    p.add_argument("--handle-hole", action="store_true",
                   help="boolean-drill the C-handle hole (tuned constants below)")
    p.add_argument("--hole-center", type=float, nargs=3, default=[0.045, 0.0, 0.0])
    p.add_argument("--hole-radius-mm", type=float, default=10.0)
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
    # Graft AFTER repair: merging/filling post-graft collapses near-coincident
    # handle verts and re-patches them as needles. The graft preserves
    # watertightness by construction (stitched rings).
    if args.hollow_top:
        out_mesh = graft_hollow_top(
            out_mesh, wall_m=args.wall_mm / 1000.0,
            floor_lift_m=args.floor_lift_mm / 1000.0)
    if args.handle_hole:
        out_mesh = drill_handle_hole(
            out_mesh, center=tuple(args.hole_center),
            radius_m=args.hole_radius_mm / 1000.0)

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
