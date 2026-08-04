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

if [ ! -x "${BUILD_DIR}/arena" ] || [ "${ARENA_FORCE_BUILD:-0}" = "1" ]; then
    echo "[entrypoint] building arena (BUILD_GUI=${BUILD_GUI}) ..." >&2
    cmake -S /workspace/gpgpu-arena -B "${BUILD_DIR}" \
          -DBUILD_GUI="${BUILD_GUI}" >&2
    cmake --build "${BUILD_DIR}" -j"$(nproc)" >&2
fi

cd "${BUILD_DIR}"
exec ./arena "$@"
