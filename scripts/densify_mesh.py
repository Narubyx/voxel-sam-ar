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
    from collections import Counter

    import numpy as np

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
    """Cuts the cup bore with a manifold boolean (wall + flat floor).

    Replaces the old stitched pocket: deleting cap faces and evacuating the
    cylinder left ragged, fill-resistant boundaries and nested the VLM's crude
    inner cup inside the graft ("two intersected cups"). A single difference
    with a Y-axis cylinder removes the cap, the solid core, and any VLM inner
    cup in one clean cut; the surviving rim annulus is the lip.
    """
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
    y_floor = y_base + floor_lift_m
    print(json.dumps({"graft": {"y_top": round(float(y_top), 5),
                                "y_base": round(float(y_base), 5),
                                "y_floor": round(float(y_floor), 5),
                                "loop": len(loop), "cap_faces": int(cap.sum())}}))

    ov = np.asarray(open_mesh.vertices)
    # The body is a tapered cone (wide rim, narrow base): the bore must be a
    # frustum hugging the wall, not a cylinder (a cylinder punches through
    # the cone = side breach = euler -2). Measure the body profile with rays
    # (away from the handle), center the axis on the body, keep wall_m inside.
    def _body_r(y, dx, dz):
        o = np.array([[cx, y, cz]])
        n = float(np.hypot(dx, dz))
        d = np.array([[dx / n, 0.0, dz / n]])
        locs, _, _ = mesh.ray.intersects_location(o, d, multiple_hits=True)
        if not len(locs):
            raise ValueError(f"body profile ray missed at y={y}.")
        return float(np.hypot(locs[:, 0], locs[:, 2]).min())

    _of = np.asarray(open_mesh.faces)
    _oc = ov[_of].mean(axis=1)
    band = ((np.abs(_oc[:, 2]) < 0.015) & (np.abs(_oc[:, 1]) < 0.020)
            & (_oc[:, 0] < 0.0335))
    if not band.any():
        raise ValueError("no body band found for bore centering.")
    cx = float(_oc[band][:, 0].mean())
    cz = float(_oc[band][:, 2].mean())
    top = y_top + 0.005
    r_top = _body_r(y_top - 0.008, -1, 0) - wall_m + 0.008 * 0.25
    r_bot = min(_body_r(y_floor + 0.002, -1, 0),
                _body_r(y_floor + 0.002, 0, 1),
                _body_r(y_floor + 0.002, 0, -1)) - wall_m
    if not r_bot > 0.012:
        raise ValueError(f"bore bottom too narrow ({r_bot:.4f}).")
    n_sec = 24
    th = np.linspace(0, 2 * np.pi, n_sec, endpoint=False)
    ring_top = np.column_stack([cx + r_top * np.cos(th),
                                np.full(n_sec, top),
                                cz + r_top * np.sin(th)])
    ring_bot = np.column_stack([cx + r_bot * np.cos(th),
                                np.full(n_sec, y_floor),
                                cz + r_bot * np.sin(th)])
    fc = np.array([[cx, y_floor, cz]])
    tc = np.array([[cx, top, cz]])
    fv = np.vstack([ring_top, ring_bot, fc, tc])
    ff = []
    for k in range(n_sec):
        k2 = (k + 1) % n_sec
        ff.append([k, n_sec + k, n_sec + k2])
        ff.append([k, n_sec + k2, k2])
        ff.append([2 * n_sec, n_sec + k2, n_sec + k])
        ff.append([2 * n_sec + 1, k, k2])
    cyl = trimesh.Trimesh(fv, ff, process=True)
    if not cyl.is_watertight or not cyl.is_winding_consistent:
        raise ValueError("frustum tool is not a clean volume.")
    if cyl.volume < 0:
        cyl.invert()
    print(json.dumps({"bore": {"r_top": round(r_top, 5),
                               "r_bot": round(r_bot, 5),
                               "cx": round(cx, 5), "cz": round(cz, 5)}}))
    try:
        out = trimesh.boolean.difference([mesh, cyl], engine="manifold")
    except Exception as exc:
        raise RuntimeError(f"bore boolean failed: {exc}") from exc
    if out is None or len(getattr(out, "faces", [])) == 0:
        raise RuntimeError("bore boolean produced an empty mesh.")
    # Drop zero-area slivers the boolean leaves at the bore wall seam, then
    # heal their zero-area slits with an exact merge (rounding is sub-GLB-
    # quantization, so lossless) and patch any residual micro-gap.
    out.update_faces(np.asarray(out.area_faces) > 1e-8)
    out.vertices = np.round(np.asarray(out.vertices), 9)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(out)
    return out


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
    # Both holes are manifold booleans on closed input: tunnel (Z cylinder
    # through the handle, disjoint from the bore) then bore (Y cylinder from
    # above the rim down to the floor level). Each output stays closed.
    if args.handle_hole:
        out_mesh = drill_handle_hole(
            out_mesh, center=tuple(args.hole_center),
            radius_m=args.hole_radius_mm / 1000.0)
    if args.hollow_top:
        out_mesh = graft_hollow_top(
            out_mesh, wall_m=args.wall_mm / 1000.0,
            floor_lift_m=args.floor_lift_mm / 1000.0)

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
