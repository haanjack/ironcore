#!/bin/bash
set -e
source "$(dirname "$0")/_ci_setup.sh"
ci_install_package

failed=0
get_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

for f in \
    tests/integration/attention/test_attention_multi_gpu.py \
    tests/integration/kvcache/test_kv_cache.py \
    tests/integration/lora/test_lora_async.py \
    tests/integration/lora/test_lora_checkpoint.py \
    tests/integration/lora/test_lora_correctness.py \
    tests/integration/optimizer/test_optimizer.py \
    tests/multi_gpu/test_expert_parallel.py \
    tests/multi_gpu/test_grad_norm.py \
    tests/multi_gpu/test_all_to_all_ep.py \
    tests/multi_gpu/test_distributed_optimizer.py \
    tests/multi_gpu/test_distributed_optimizer_checkpoint.py \
    tests/integration/offload/test_weight_streaming_mp.py \
    tests/integration/offload/test_weight_streaming_dp.py \
    tests/multi_gpu/offload/test_tp_offload.py \
    tests/multi_gpu/offload/test_ddp_offload.py \
    tests/multi_gpu/offload/test_fsdp_full_shard_activation_spill.py \
    tests/multi_gpu/offload/test_fsdp_shard_grad_op_offload.py \
    tests/multi_gpu/offload/test_distopt_offload.py; do
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
