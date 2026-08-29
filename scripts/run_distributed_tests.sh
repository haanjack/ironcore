#!/bin/bash
set -e
source "$(dirname "$0")/_ci_setup.sh"
ci_install_package

failed=0
get_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

# Single source of truth for the mp-marked file list — see that file's header.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./distributed_test_files.sh
source "$SCRIPT_DIR/distributed_test_files.sh"

for f in "${DIST_TEST_FILES_NP2[@]}"; do
    if [ -f "$f" ]; then
        echo "=== Running: $f ==="
        timeout 300 torchrun --nproc_per_node=2 --master_port=$(get_free_port) \
            -m pytest "$f" -m "mp" --timeout=120 -v --tb=short -q || {
            echo "FAILED or TIMED OUT: $f (exit=$?)"
            failed=1
        }
    fi
done

exit $failed
