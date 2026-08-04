#!/usr/bin/env bash
# Builds arena if needed, then forwards all arguments to the binary.
#
# Lets the compose CLI service behave like the binary itself:
#   docker compose run --rm cli --list
#   docker compose run --rm cli --run reduce -p n=4000000 --profile
#
# Build output goes to stderr so stdout stays clean for JSON.
set -euo pipefail

BUILD_DIR=${ARENA_BUILD_DIR:-/workspace/gpgpu-arena/build}
BUILD_GUI=${ARENA_BUILD_GUI:-ON}

# Always build. The source tree is bind-mounted, so it can change between
# runs; skipping the build when ./arena merely exists silently runs stale code
# against edited sources. cmake --build is incremental, so this is cheap when
# nothing changed. Set ARENA_SKIP_BUILD=1 to bypass.
if [ "${ARENA_SKIP_BUILD:-0}" != "1" ]; then
    echo "[entrypoint] building arena (BUILD_GUI=${BUILD_GUI}) ..." >&2
    cmake -S /workspace/gpgpu-arena -B "${BUILD_DIR}" \
          -DBUILD_GUI="${BUILD_GUI}" >&2
    cmake --build "${BUILD_DIR}" -j"$(nproc)" >&2
fi

cd "${BUILD_DIR}"
exec ./arena "$@"
