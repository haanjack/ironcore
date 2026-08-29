# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Single source of truth for which test files need a real `torchrun` launch,
# and at what --nproc_per_node. Sourced by both scripts/run_tests.sh (full
# local suite, also runs the NP1 list for general single-process-under-torchrun
# sanity) and scripts/run_distributed_tests.sh (CI job for `@pytest.mark.mp`
# tests, NP2 only). Keeping one copy avoids the two scripts silently
# diverging on which mp-marked files actually get exercised under torchrun —
# see docs/experiments/test_suite_isolation_issues.md.
#
# Regenerate/audit the NP2 list against actual @pytest.mark.mp usage with:
#   grep -rl 'pytest.mark.mp' tests/ | sort

# torchrun --nproc_per_node=1: files that need a torchrun-style launch
# (env vars, single-process distributed init) but not real multi-GPU.
DIST_TEST_FILES_NP1=(
    "tests/integration/alignment/test_dpo_integration.py"
    "tests/integration/attention/test_chunked_parallel.py"
    "tests/integration/attention/test_flash_attention_cache.py"
    "tests/integration/eval/test_eval_integration.py"
    "tests/integration/kvcache/test_kv_cache.py"
    "tests/integration/kvcache/test_kv_cache_stateful.py"
    "tests/integration/lora/test_lora_async.py"
    "tests/integration/lora/test_lora_checkpoint.py"
    "tests/integration/moe/test_moe_correctness.py"
    "tests/integration/moe/test_moe_functional.py"
    "tests/integration/moe/test_moe_layer.py"
    "tests/integration/optimizer/test_optimizer.py"
    "tests/integration/test_integration.py"
)

# torchrun --nproc_per_node=2: every file containing @pytest.mark.mp tests.
DIST_TEST_FILES_NP2=(
    "tests/integration/attention/test_attention_multi_gpu.py"
    "tests/integration/kvcache/test_kv_cache.py"
    "tests/integration/lora/test_lora_async.py"
    "tests/integration/lora/test_lora_checkpoint.py"
    "tests/integration/lora/test_lora_correctness.py"
    "tests/integration/offload/test_checkpoint_offload.py"
    "tests/integration/offload/test_combinational_training.py"
    "tests/integration/offload/test_gradient_accumulation_offload.py"
    "tests/integration/offload/test_moe_offload_smoke.py"
    "tests/integration/offload/test_training_loop_offload.py"
    "tests/integration/offload/test_weight_streaming_dp.py"
    "tests/integration/offload/test_weight_streaming_e2e.py"
    "tests/integration/offload/test_weight_streaming_mp.py"
    "tests/integration/optimizer/test_optimizer.py"
    "tests/integration/test_grpo_smoke.py"
    "tests/multi_gpu/offload/test_ddp_offload.py"
    "tests/multi_gpu/offload/test_distopt_offload.py"
    "tests/multi_gpu/offload/test_fsdp_full_shard_activation_spill.py"
    "tests/multi_gpu/offload/test_fsdp_shard_grad_op_offload.py"
    "tests/multi_gpu/offload/test_tp_offload.py"
    "tests/multi_gpu/test_all_to_all_ep.py"
    "tests/multi_gpu/test_distributed_optimizer.py"
    "tests/multi_gpu/test_distributed_optimizer_checkpoint.py"
    "tests/multi_gpu/test_expert_parallel.py"
    "tests/multi_gpu/test_grad_norm.py"
    "tests/multi_gpu/test_tp_equivalence.py"
    "tests/unit/offload/test_pairwise_optimizer_activation_spill.py"
    "tests/unit/offload/test_pairwise_optimizer_weight_offload.py"
    "tests/unit/reward/test_reward_manager.py"
)
