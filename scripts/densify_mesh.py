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

    # The shell is an origin-centered tapered cone; the handle root (+X solid
    # reaching the rim) pollutes every +X measurement, so profile the shell
    # from the clean -X side only and mirror it. The bore is a vertical,
    # origin-centered multi-ring tool hugging shell-minus-wall at each level.
    def _shell_r(y):
        o = np.array([[0.0, y, 0.0]])
        locs, _, _ = mesh.ray.intersects_location(
            o, np.array([[-1.0, 0.0, 0.0]]), multiple_hits=True)
        if not len(locs):
            raise ValueError(f"shell profile ray missed at y={y}.")
        return float(-locs[:, 0].min())

    top = y_top + 0.005
    levels = [top, y_top - 0.008, 0.0, -0.015, y_floor]
    radii = []
    for i, yl in enumerate(levels):
        if i == 0:
            radii.append(_shell_r(levels[1]) - wall_m
                         + (levels[0] - levels[1]) * 0.25 + 0.003)
        else:
            radii.append(_shell_r(yl) - wall_m)
    if not radii[-1] > 0.010:
        raise ValueError(f"bore bottom too narrow ({radii[-1]:.4f}).")
    print(json.dumps({"bore_levels":
                      [[round(y, 5), round(r, 5)]
                       for y, r in zip(levels, radii)]}))
    n_sec = 24
    th = np.linspace(0, 2 * np.pi, n_sec, endpoint=False)
    rings = [np.column_stack([r * np.cos(th), np.full(n_sec, yl),
                              r * np.sin(th)])
             for yl, r in zip(levels, radii)]
    n_lv = len(levels)
    fc = np.array([[0.0, y_floor, 0.0]])
    tc = np.array([[0.0, top, 0.0]])
    fv = np.vstack(rings + [fc, tc])
    ff = []
    for k in range(n_sec):
        k2 = (k + 1) % n_sec
        for lv in range(n_lv - 1):
            a0, a1 = lv * n_sec + k, lv * n_sec + k2
            b0, b1 = (lv + 1) * n_sec + k, (lv + 1) * n_sec + k2
            ff.append([a0, b0, b1])
            ff.append([a0, b1, a1])
        ff.append([n_lv * n_sec, (n_lv - 1) * n_sec + k2,
                   (n_lv - 1) * n_sec + k])
        ff.append([n_lv * n_sec + 1, k, k2])
    cyl = trimesh.Trimesh(fv, ff, process=True)
    if not cyl.is_watertight or not cyl.is_winding_consistent:
        raise ValueError("frustum tool is not a clean volume.")
    if cyl.volume < 0:
        cyl.invert()
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
