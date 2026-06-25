#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_libs.sh  —  Copy custom OneEye libs into their DeepStream paths
#
# Run this INSIDE the deepstream-analytics container:
#   docker exec -it deepstream-analytics bash /app/scripts/deploy_libs.sh
#
# The project root is mounted at /app via docker-compose, so all local
# files under libs/ are already present at /app/libs/ inside the container.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DS_ROOT="/opt/nvidia/deepstream/deepstream-7.1"
LIBS_SRC="/app/libs"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -d "$LIBS_SRC" ]]  || die "Libs source not found: $LIBS_SRC  (is the volume mounted?)"
[[ -d "$DS_ROOT" ]]   || die "DeepStream root not found: $DS_ROOT"

log "Source  : $LIBS_SRC"
log "DS root : $DS_ROOT"
echo ""

# ── Helper: copy with backup ──────────────────────────────────────────────────
deploy() {
    local src="$1"
    local dst="$2"

    if [[ ! -f "$src" ]]; then
        warn "Skipping (not found): $src"
        return
    fi

    # Backup original only once (don't overwrite an existing backup)
    if [[ -f "$dst" && ! -f "${dst}.orig" ]]; then
        cp "$dst" "${dst}.orig"
        warn "Backed up original → ${dst}.orig"
    fi

    mkdir -p "$(dirname "$dst")"
    cp -v "$src" "$dst"
}

# ── Runtime .so libraries ─────────────────────────────────────────────────────
log "=== Deploying runtime libraries (.so) ==="
deploy "$LIBS_SRC/libnvds_msgconv.so" \
       "$DS_ROOT/lib/libnvds_msgconv.so"

deploy "$LIBS_SRC/libnvdsgst_nvmultiurisrcbin.so" \
       "$DS_ROOT/lib/gst-plugins/libnvdsgst_nvmultiurisrcbin.so"

deploy "$LIBS_SRC/libnvdsgst_nvurisrcbin.so" \
       "$DS_ROOT/lib/gst-plugins/libnvdsgst_nvurisrcbin.so"

# ── Headers ───────────────────────────────────────────────────────────────────
echo ""
log "=== Deploying headers (.h) ==="
deploy "$LIBS_SRC/nvdsmeta_schema.h" \
       "$DS_ROOT/sources/includes/nvdsmeta_schema.h"

deploy "$LIBS_SRC/gst-nvdscommonconfig.h" \
       "$DS_ROOT/sources/gst-plugins/gst-nvmultiurisrcbin/gst-nvdscommonconfig.h"

deploy "$LIBS_SRC/gstdsnvurisrcbin.h" \
       "$DS_ROOT/sources/gst-plugins/gst-nvurisrcbin/gstdsnvurisrcbin.h"

# ── Sources ───────────────────────────────────────────────────────────────────
echo ""
log "=== Deploying sources (.c / .cpp) ==="
deploy "$LIBS_SRC/deepstream_source_bin.c" \
       "$DS_ROOT/sources/apps/apps-common/src/deepstream_source_bin.c"

deploy "$LIBS_SRC/eventmsg_payload.cpp" \
       "$DS_ROOT/sources/libs/nvmsgconv/eventmsg_payload.cpp"

deploy "$LIBS_SRC/gstdsnvmultiurisrcbin.cpp" \
       "$DS_ROOT/sources/gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp"

deploy "$LIBS_SRC/gstdsnvurisrcbin.cpp" \
       "$DS_ROOT/sources/gst-plugins/gst-nvurisrcbin/gstdsnvurisrcbin.cpp"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
log "✓ All libs deployed. GStreamer plugins reload automatically on next pipeline start."
log "  To verify .so overrides are active:"
echo "     ldconfig -p | grep nvds_msgconv"
echo "     gst-inspect-1.0 nvurisrcbin 2>/dev/null | head -5"
