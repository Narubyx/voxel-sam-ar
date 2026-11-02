#!/usr/bin/env bash
# VOXEL-SAM AR — Phase 5: end-to-end pipeline check.
#
#   Remote mode (default): --video demo.mp4 --x 320 --y 240 (needs MODAL_URL)
#     video -> POST /segment -> keyframes -> baseten/infer -> geometry.json
#     -> agent/loop (local, baseline) -> model.glb + model.usdz
#   Local mode (--local): synthetic keyframes instead of Modal (fully offline).
#
# Any failing step aborts with a non-zero exit. The mesh step needs trimesh;
# without it the step is SKIPPED (offline smoke) unless --require-mesh.
#
# Usage:
#   scripts/e2e.sh --local
#   scripts/e2e.sh --video assets/demo.mp4 --x 320 --y 240
#   MODAL_URL=https://<app>.modal.run scripts/e2e.sh --video demo.mp4 --x 80 --y 60
#
# Env: MODAL_URL, BASETEN_API_KEY (optional, else silhouette fallback),
#      META_MODEL_API_KEY/BASE_URL (optional, only if --max-cycles > 1),
#      PYTHON (default python3).
#
set -euo pipefail

LOCAL=0
REQUIRE_MESH=0
NO_REMOTE=0
VIDEO=""
X="320"
Y="240"
MODAL_URL="${MODAL_URL:-}"
WORKDIR=""
NUM_KEYFRAMES=12
HEIGHT_M="0.20"
MAX_CYCLES=1
PYTHON="${PYTHON:-python3}"

usage() {
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d; s/^# \{0,1\}//'
  echo "Options: --local --require-mesh --no-remote --video PATH --x N --y N"
  echo "  --modal-url URL --workdir DIR --num-keyframes N --height-m M --max-cycles N"
}

log() { printf '[e2e %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[e2e %s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
need_file() {
  [ -s "$1" ] || die "missing or empty artifact: $1"
  log "ok: $1 ($(wc -c <"$1") bytes)"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --local) LOCAL=1; shift ;;
    --require-mesh) REQUIRE_MESH=1; shift ;;
    --no-remote) NO_REMOTE=1; shift ;;
    --video) VIDEO="$2"; shift 2 ;;
    --x) X="$2"; shift 2 ;;
    --y) Y="$2"; shift 2 ;;
    --modal-url) MODAL_URL="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --num-keyframes) NUM_KEYFRAMES="$2"; shift 2 ;;
    --height-m) HEIGHT_M="$2"; shift 2 ;;
    --max-cycles) MAX_CYCLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$LOCAL" -eq 0 ]; then # validate before creating any directory
  [ -n "$VIDEO" ] || die "remote mode needs --video PATH (or use --local)"
  [ -n "$MODAL_URL" ] || die "remote mode needs MODAL_URL env or --modal-url"
  [ -f "$VIDEO" ] || die "video not found: $VIDEO"
  command -v curl >/dev/null || die "curl not found (remote mode)"
fi
[ -n "$WORKDIR" ] || WORKDIR="$ROOT/e2e_out/e2e_$(date +%Y%m%d_%H%M%S)"
KF_DIR="$WORKDIR/kf"
GEOMETRY="$WORKDIR/geometry.json"
RUNS_DIR="$WORKDIR/runs"
mkdir -p "$WORKDIR" "$KF_DIR"

if ! command -v "$PYTHON" >/dev/null 2>&1 || ! "$PYTHON" --version >/dev/null 2>&1; then
  if [ "$PYTHON" = "python3" ] && command -v python >/dev/null 2>&1 \
      && python --version >/dev/null 2>&1; then
    PYTHON=python # Windows shells where python3 is a store stub
  else
    die "working python not found: $PYTHON (override with PYTHON=...)"
  fi
fi
"$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
  || die "$PYTHON is older than 3.11"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------- step 1: frames
if [ "$LOCAL" -eq 1 ]; then
  log "step 1/3: synthetic keyframes (offline, $NUM_KEYFRAMES views)"
  "$PYTHON" "$ROOT/scripts/make_demo_assets.py" \
    --keyframes-dir "$KF_DIR" --num-views "$NUM_KEYFRAMES" --height-m "$HEIGHT_M" \
    --out "$WORKDIR/synthetic_golden.json"
  META_ARG=(--meta "$KF_DIR/meta.json")
else
  log "step 1/3: POST /segment ($VIDEO @ $X,$Y, $NUM_KEYFRAMES keyframes)"
  MODAL_URL="${MODAL_URL%/}"
  # -L follows Modal's 303 __modal_attempt_token for long requests (GET the token).
  if ! curl -sS -L --fail-with-body --max-time 900 -X POST "$MODAL_URL/segment" \
      -F "file=@${VIDEO};type=video/mp4" -F "x=${X}" -F "y=${Y}" \
      -F "num_keyframes=${NUM_KEYFRAMES}" -o "$WORKDIR/response.json"; then
    head -c 2000 "$WORKDIR/response.json" >&2 || true
    die "POST /segment failed (see body above)"
  fi
  "$PYTHON" - "$WORKDIR/response.json" <<'EOF' \
    || die "POST /segment returned invalid/empty JSON (transient?); retry e2e.sh"
import json, sys
resp = json.load(open(sys.argv[1], encoding="utf-8"))
assert resp.get("keyframes"), "0 keyframes in response"
print(f"e2e: /segment ok: job={resp.get('job_id')} "
      f"kf={len(resp['keyframes'])} backend={resp.get('meta', {}).get('sam_backend')}")
EOF
  "$PYTHON" - "$WORKDIR" <<'EOF'
import base64, json, sys
work = sys.argv[1]
resp = json.load(open(f"{work}/response.json", encoding="utf-8"))
kfs = resp.get("keyframes", [])
if not kfs:
    sys.exit("e2e: /segment returned 0 keyframes")
for rank, kf in enumerate(kfs):
    open(f"{work}/kf/keyframe_{rank:02d}.png", "wb").write(
        base64.b64decode(kf["png_b64"]))
meta = {"job_id": resp.get("job_id"), "frames": [k["index"] for k in kfs],
        "meta": resp.get("meta", {})}
json.dump(meta, open(f"{work}/kf/meta.json", "w", encoding="utf-8"))
print(f"e2e: extracted {len(kfs)} keyframes "
      f"(backend={meta['meta'].get('sam_backend')})")
EOF
  META_ARG=(--meta "$KF_DIR/meta.json")
fi
ls "$KF_DIR"/keyframe_*.png >/dev/null || die "no keyframe_*.png in $KF_DIR"
need_file "$KF_DIR/meta.json"

# ------------------------------------------------------------- step 2: geometry
log "step 2/3: baseten/infer -> geometry.json"
INFER_ARGS=(--keyframes-dir "$KF_DIR" "${META_ARG[@]}" --height-m "$HEIGHT_M"
  --out "$GEOMETRY")
[ "$NO_REMOTE" -eq 1 ] && INFER_ARGS+=(--no-remote)
"$PYTHON" -m baseten.infer "${INFER_ARGS[@]}" || die "baseten/infer failed"
need_file "$GEOMETRY"
"$PYTHON" - "$GEOMETRY" <<'EOF'
import json, sys
from baseten.schema import GeometryEstimate
g = GeometryEstimate.model_validate(json.load(open(sys.argv[1], encoding="utf-8")))
print(f"e2e: geometry OK: {g.summary()}")
EOF

# ----------------------------------------------------------------- step 3: mesh
log "step 3/3: agent/loop (local baseline) -> model.glb + model.usdz"
if "$PYTHON" -c "import trimesh" 2>/dev/null; then
  "$PYTHON" -m agent.loop --geometry "$GEOMETRY" --exec local \
    --max-cycles "$MAX_CYCLES" --out-dir "$RUNS_DIR" \
    || die "agent/loop failed (see manifest in $RUNS_DIR)"
  RUN_DIR="$(ls -dt "$RUNS_DIR"/*/ | head -n 1)"
  need_file "$RUN_DIR/model.glb"
  need_file "$RUN_DIR/model.usdz"
  "$PYTHON" - "$RUN_DIR" <<'EOF'
import json, sys
run = sys.argv[1]
manifest = json.load(open(f"{run}/manifest.json", encoding="utf-8"))
assert manifest["status"] == "success", f"status={manifest['status']}"
rep = manifest["cycles"][-1]["report"]
assert rep["ok"] and not rep["errors"], f"errors={rep['errors']}"
s = rep["stats"]
print(f"e2e: mesh OK: watertight={s['watertight']} "
      f"winding={s['winding_consistent']} v={s['vertices']} f={s['faces']}")
EOF
  log "run dir: $RUN_DIR"
elif [ "$REQUIRE_MESH" -eq 1 ]; then
  die "trimesh not installed and --require-mesh was given"
else
  log "SKIP mesh step: trimesh not installed (pip install \"trimesh[easy]\");"
  log "SKIP rerun with --require-mesh to turn this into an error."
fi

log "E2E DONE: $WORKDIR"
