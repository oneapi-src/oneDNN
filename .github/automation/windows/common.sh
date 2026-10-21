#!/usr/bin/env bash
# *******************************************************************************
# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for Windows MSVC builds on self-hosted Git Bash runners.
#
# Runner layout (Python 3.12 system install):
#   C:\Program Files\Python312
#   C:\Program Files\Python312\Lib\site-packages\cmake\data\bin\cmake.exe
#   C:\Program Files\Python312\Lib\site-packages\ninja\data\bin\ninja.exe
# *******************************************************************************

set -euo pipefail

PYTHON312_ROOT="/c/Program Files/Python312"
SITE_PACKAGES="${PYTHON312_ROOT}/Lib/site-packages"

windows_parallel_jobs() {
  nproc 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-4}"
}

windows_resolve_cmake() {
  if [ -n "${CMAKE:-}" ] && [ -x "${CMAKE}" ]; then
    echo "Using CMAKE=${CMAKE}"
    return 0
  fi
  local candidate
  for candidate in \
    "${SITE_PACKAGES}/cmake/data/bin/cmake.exe" \
    "${PYTHON312_ROOT}/Scripts/cmake.exe"; do
    if [ -x "${candidate}" ]; then
      export CMAKE="${candidate}"
      echo "Using CMAKE=${CMAKE}"
      return 0
    fi
  done
  echo "error: cmake.exe not found under ${PYTHON312_ROOT}" >&2
  return 1
}

windows_resolve_ninja() {
  if [ -n "${NINJA:-}" ] && [ -x "${NINJA}" ]; then
    echo "Using NINJA=${NINJA}"
    return 0
  fi
  local candidate
  for candidate in \
    "${SITE_PACKAGES}/ninja/data/bin/ninja.exe" \
    "${PYTHON312_ROOT}/Scripts/ninja.exe"; do
    if [ -x "${candidate}" ]; then
      export NINJA="${candidate}"
      echo "Using NINJA=${NINJA}"
      return 0
    fi
  done
  if command -v ninja >/dev/null 2>&1; then
    export NINJA="$(command -v ninja)"
    echo "Using NINJA=${NINJA}"
    return 0
  fi
  echo "error: ninja.exe not found under ${PYTHON312_ROOT}" >&2
  return 1
}

windows_load_msvc() {
  local zendnn_root="${1:?ZenDNN root path required}"
  # shellcheck source=/dev/null
  source "${zendnn_root}/scripts/load_msvc.sh"
}
