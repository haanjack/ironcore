# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Compare actual EP layers against an unsharded, differentiable MoE oracle.

Run on two GPUs. Each rank checks its own samples and gradients; a finite
output or an existing gradient alone is insufficient. The report retains
unsupported/failing paths instead of silently treating them as passes.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    import torch.distributed as dist

    from ironcore.config import (
        DataConfig,
        InitConfig,
        MainConfig,
        ModelConfig,
        OperationConfig,
        OptimConfig,
        ParallelConfig,
        PEFTConfig,
        ProfilerConfig,
        TrainerConfig,
        UtilsConfig,
    )
    from ironcore.config.config_model import BiasConfig
    from ironcore.config.config_moe import MoEConfig
    from ironcore.global_vars import set_global_states
    from ironcore.layers.moe.moe_layer import CommunicationMode, MoEMLP
    from ironcore.parallel.expert_parallel import initialize_expert_parallel
    from ironcore.parallel.parallel import initialize_parallelism
    from ironcore.parallel.parallel_states import initialize_model_parallel

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    initialize_model_parallel(1, timeout_in_minutes=2)
    initialize_expert_parallel(2, 1)
    cfg = MainConfig(
        model=ModelConfig(
            d_model=32,
            d_ffn=32,
            num_layers=1,
            num_attention_heads=4,
            num_attention_groups=4,
            head_dim=8,
            precision="float32",
            activation_type="swiglu",
            bias=BiasConfig.none(),
            dropout_mlp=0.0,
            moe=MoEConfig(
                use_moe=True,
                num_routed_experts=4,
                num_shared_experts=1,
                num_experts_per_token=2,
                aux_loss_alpha=0.0,
            ),
        ),
        trainer=TrainerConfig(),
        init=InitConfig(seed=42),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(rank=rank, local_rank=rank, world_size=2),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )
    cfg.model.tokenizer_type = "sentencepiece"
    cfg.model.vocab_name_or_path = str(args.output / "tokenizer")
    set_global_states(cfg)
    torch.manual_seed(42)
    reference = MoEMLP(cfg).cuda()
    reference.init_weights()
    checks = []
    try:
        for mode in (CommunicationMode.ALL_REDUCE, CommunicationMode.ALL_TO_ALL):
            for identical in (True, False):
                ep_cfg = copy.deepcopy(cfg)
                ep_cfg.model.moe.expert_model_parallel_size = 2
                parallel = MoEMLP(ep_cfg, communication_mode=mode).cuda()
                copied = {}
                source = reference.state_dict()
                for name in parallel.state_dict():
                    if name.startswith("routed_experts."):
                        parts = name.split(".")
                        parts[1] = str(rank * 2 + int(parts[1]))
                        copied[name] = source[".".join(parts)]
                    else:
                        copied[name] = source[name]
                parallel.load_state_dict(copied)
                torch.manual_seed(123 + (0 if identical else rank))
                x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
                ep_x = x.detach().clone().requires_grad_(True)
                reference.zero_grad(set_to_none=True)
                expected = reference(x)
                actual = parallel(ep_x)
                probe = torch.linspace(0.1, 1.0, expected.numel(), device="cuda").reshape_as(
                    expected
                )
                (expected * probe).sum().backward()
                (actual * probe).sum().backward()
                # Owned experts process tokens from every EP source. Compare
                # against the sum of rank-local reference expert derivatives;
                # input/router/shared derivatives remain rank-local here.
                reference_expert_grads = {}
                for name, parameter in reference.named_parameters():
                    if name.startswith("routed_experts."):
                        gradient = (
                            parameter.grad.clone()
                            if parameter.grad is not None
                            else torch.zeros_like(parameter)
                        )
                        dist.all_reduce(gradient)
                        reference_expert_grads[name] = gradient
                errors = {
                    "output": (expected - actual).abs().max().item(),
                    "input_gradient": (x.grad - ep_x.grad).abs().max().item()
                    if ep_x.grad is not None
                    else None,
                }
                missing = []
                gradient_errors = {}
                reference_params = dict(reference.named_parameters())
                for name, p in parallel.named_parameters():
                    full_name = name
                    if name.startswith("routed_experts."):
                        parts = name.split(".")
                        parts[1] = str(rank * 2 + int(parts[1]))
                        full_name = ".".join(parts)
                    expected_grad = reference_params[full_name].grad
                    if full_name in reference_expert_grads:
                        expected_grad = reference_expert_grads[full_name]
                    if expected_grad is None:
                        expected_grad = torch.zeros_like(reference_params[full_name])
                    if p.grad is None:
                        missing.append(name)
                    gradient_errors[name] = (
                        (expected_grad - (p.grad if p.grad is not None else torch.zeros_like(p)))
                        .abs()
                        .max()
                        .item()
                    )
                errors["parameter_gradient"] = max(gradient_errors.values())
                passed = all(v is not None and v <= 2e-5 for v in errors.values())
                checks.append(
                    dict(
                        mode=mode.value,
                        identical_inputs=identical,
                        status="passed" if passed else "failed",
                        max_abs_errors=errors,
                        missing_gradients=missing,
                        gradient_errors=gradient_errors,
                    )
                )
        # Use the real wrapping function: different expert IDs must retain their weights.
        before = {n: p.detach().clone() for n, p in parallel.named_parameters()}
        try:
            wrapped = initialize_parallelism(ep_cfg, parallel)
        except (ValueError, NotImplementedError) as error:
            checks.append(dict(mode="trainer_ep_wrapping", status="unsupported", reason=str(error)))
        else:
            errors = {
                n: (p - before[n]).abs().max().item()
                for n, p in wrapped.module.named_parameters()
                if n.startswith("routed_experts.")
            }
            checks.append(
                dict(
                    mode="trainer_ep_wrapping",
                    status="passed" if max(errors.values()) == 0 else "failed",
                    max_abs_errors=errors,
                )
            )
        (args.output / f"rank{rank}.json").write_text(
            json.dumps({"rank": rank, "checks": checks}, indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        p.error("Use a new output directory")
    args.output.mkdir(parents=True)
    from validate_trainers import create_tokenizer

    create_tokenizer(args.output / "tokenizer")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(args.output),
    ]
    with (args.output / "run.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log, timeout=180, check=False)
    report = {"command": cmd, "returncode": result.returncode, "status": "failed"}
    if result.returncode == 0:
        report["ranks"] = [
            json.loads((args.output / f"rank{rank}.json").read_text()) for rank in (0, 1)
        ]
        if all(c["status"] == "passed" for r in report["ranks"] for c in r["checks"]):
            report["status"] = "passed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("EP oracle:", report["status"], args.output / "report.json", flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
