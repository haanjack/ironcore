# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Shared setup for the scripts the CI test jobs run inside the container.
# Source it, do not execute it.

# Refresh the editable install so the container picks up the checked-out tree.
#
# launch.sh runs the container as the invoking user (-u $(id -u):$(id -g)), and
# on the ROCm images site-packages under /opt/venv is root-owned, so this write
# fails with EACCES. That is not fatal: the image already has ironcore and the
# dev extras installed at build time, and the source tree is bind-mounted, so an
# editable install has nothing new to contribute. Abort only if the package is
# genuinely missing, rather than letting `set -e` kill the job over a no-op.
ci_install_package() {
    if pip install -e ".[dev]" -q; then
        return 0
    fi
    echo "warning: 'pip install -e .[dev]' failed — falling back to the preinstalled package" >&2
    if ! python -c "import ironcore" 2>/dev/null; then
        echo "error: ironcore is not importable and cannot be installed; aborting" >&2
        return 1
    fi
    python - <<'PY' >&2
import importlib.metadata as md

for dist in ("pytest", "pytest-timeout"):
    try:
        md.version(dist)
    except md.PackageNotFoundError:
        print(f"error: required dev dependency '{dist}' is missing")
        raise SystemExit(1)
PY
}
