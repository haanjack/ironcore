# IronCore Parallelism Architecture Design

## Executive Summary

This document outlines the parallelism architecture for IronCore's distributed optimizer implementation. The design focuses on **Optimizer State Partitioning** (partitioning optimizer states across data-parallel ranks while keeping parameters and gradients replicated), integrated with existing Tensor Parallelism and Expert Parallelism.

**Key Design Decision**: We implement optimizer state partitioning only. For gradient sharding, users should use FSDP with `shard_grad_op` strategy.

**Why not implement gradient partitioning in DistributedOptimizer?**
- FSDP's `shard_grad_op` uses reduce-scatter (more efficient than all-reduce + zero)
- FSDP has better ecosystem support (CPU offload, torch.compile)
- Clear separation of concerns avoids duplicate functionality

---

## 1. Current State Analysis

### 1.1 Existing Components

| Component | Status | Location |
|-----------|--------|----------|
| Tensor Parallelism (TP) | ✅ Complete | `ironcore/parallel/tensor_parallel/` |
| Data Parallelism (DP) | ✅ Complete | `ironcore/parallel/parallel.py` (DDP/FSDP) |
| Expert Parallelism (EP) | ✅ Complete | `ironcore/parallel/expert_parallel/` |
| Pipeline Parallelism (PP) | ❌ Not Implemented | Referenced in config only |
| DistributedOptimizer | ⚠️ Exists but not integrated | `ironcore/optimizer/distributed_optimizer.py` |

### 1.2 Process Group Layout (Current)

```
World Layout: [DP][TP] or [DP][EP][TP]

Example (World=8, TP=2, EP=2):
├── DP Group 0: [0, 1, 2, 3]
│   ├── EP Group 0: [0, 1] (TP-sharded experts 0-31)
│   │   ├── TP Rank 0: [0]
│   │   └── TP Rank 1: [1]
│   └── EP Group 1: [2, 3] (TP-sharded experts 32-63)
│       ├── TP Rank 0: [2]
│       └── TP Rank 1: [3]
└── DP Group 1: [4, 5, 6, 7]
    └── ... (same structure)
```

---

## 2. Parallelism Architecture

### 2.1 Multi-Dimensional Parallelism

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Parallelism Dimensions                           │
├──────────────┬──────────────────────────────────────────────────────┤
│ Dimension    │ Description                                          │
├──────────────┼──────────────────────────────────────────────────────┤
│ TP (Tensor)  │ Shards model weights within a layer                  │
│ DP (Data)    │ Replicates model, shards data batches                │
│ EP (Expert)  │ Distributes MoE experts across ranks                 │
│ PP (Pipeline)│ Splits model layers across pipeline stages           │
├──────────────┼──────────────────────────────────────────────────────┤
│ Optimizer    │ PARTITIONS optimizer states across DP ranks          │
│ Partitioning │ - Parameters: REPLICATED (full copy on each rank)    │
│              │ - Gradients: ALL-REDUCED (via DDP)                   │
│              │ - Optimizer States: PARTITIONED (1/N on each rank)   │
└──────────────┴──────────────────────────────────────────────────────┘
```

### 2.2 When to Use What

| Scenario | Recommended Setup | Why |
|----------|-------------------|-----|
| Small model, many GPUs | DDP only | Simplicity |
| Large model, fast interconnect | TP + DDP | TP reduces memory, DDP handles sync |
| Memory-constrained training | TP + DistributedOptimizer | Optimizer states partitioned |
| Need gradient sharding too | FSDP `shard_grad_op` | Built-in support |
| MoE models | TP + EP + DistributedOptimizer | Expert-aware partitioning |
| Very large models (future) | TP + PP + DistributedOptimizer | Layer distribution |

### 2.3 Comparison with FSDP

| Feature | DistributedOptimizer + DDP | FSDP |
|---------|----------------------------|------|
| Parameter sharding | ❌ Replicated | ✅ Configurable |
| Gradient sharding | ❌ All-reduced | ✅ Configurable |
| Optimizer state sharding | ✅ Partitioned | ✅ Configurable |
| Forward pass overhead | None | All-gather on layer access |
| Best for | Memory-bound optimizer states | Memory-bound everything |
| Integration with TP | Simple | Requires careful config |

---

## 3. Detailed Design

### 3.1 Configuration Schema

```python
# File: ironcore/config/config_parallel.py

@dataclass
class ParallelConfig(BaseConfig):
    # Existing fields
    rank: int = -1
    local_rank: int = 0
    world_size: int = 1
    dist_backend: str = "nccl"
    timeout_minute: float = 10.0

    # FSDP options (mutually exclusive with distributed optimizer)
    use_fsdp: bool = False
    fsdp_sharding_strategy: Literal["full", "hybrid", "no_shard", "shard_grad_op"] = "full"
    fsdp_offload_params: bool = False
    fsdp_mixed_precision: Literal["fp16", "bf16", "fp32", "mixed"] = "mixed"
    fsdp_use_orig_params: bool = False
    fsdp_state_dict_type: Literal["full", "local", "sharded"] = "full"

    # DistributedOptimizer options
    # Partitions optimizer states across data-parallel ranks
    use_distributed_optimizer: bool = False

    # Pipeline parallelism (future)
    use_pipeline: bool = False
    pipeline_num_stages: int = 1
```

### 3.2 DistributedOptimizer Architecture

```python
# File: ironcore/optimizer/distributed_optimizer.py

class DistributedOptimizer:
    """
    Distributed optimizer that partitions optimizer states across DP ranks.

    Memory savings (N = DP world size):
    - Parameters: P bytes (replicated on each rank)
    - Gradients: P bytes (all-reduced via DDP)
    - Optimizer states: 2P/N bytes (partitioned, fp32 moments)

    Total per rank: 2P + 2P/N (vs 4P without partitioning)

    Communication pattern:
    1. Forward pass: Normal (no communication)
    2. Backward pass: DDP all-reduces gradients
    3. Optimizer step:
       a. Each rank updates only its partition of parameters
       b. All-gather updated parameters from all ranks

    Compatible with:
    - Tensor Parallelism (TP)
    - Expert Parallelism (EP)
    - DDP (required, not FSDP)

    Incompatible with:
    - FSDP (use FSDP's built-in sharding instead)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        process_group=None,  # Defaults to DP group
    ):
        self.optimizer = optimizer

        if process_group is None:
            from ironcore.parallel.parallel_states import get_data_parallel_group
            process_group = get_data_parallel_group()

        self.process_group = process_group

        if dist.is_initialized():
            self.dp_size = dist.get_world_size(group=process_group)
            self.dp_rank = dist.get_rank(group=process_group)
        else:
            self.dp_size = 1
            self.dp_rank = 0

        # Collect all parameters
        self.all_params: list[torch.nn.Parameter] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                self.all_params.append(p)

        # Round-robin assignment: param i owned by rank (i % dp_size)
        self.local_param_indices: set[int] = {
            i for i in range(len(self.all_params))
            if i % self.dp_size == self.dp_rank
        }

    def step(self, closure=None):
        """
        Optimizer step with parameter broadcasting.

        1. Temporarily null non-local gradients (so inner optimizer skips them)
        2. Run inner optimizer (updates only local partition)
        3. Broadcast updated parameters from owner ranks
        """
        # Save and null non-local grads
        saved_grads = {}
        if self.dp_size > 1:
            for i, p in enumerate(self.all_params):
                if i not in self.local_param_indices and p.grad is not None:
                    saved_grads[i] = p.grad
                    p.grad = None

        # Run inner optimizer
        result = self.optimizer.step(closure)

        # Restore saved grads (for grad_norm logging)
        for i, grad in saved_grads.items():
            self.all_params[i].grad = grad

        # Broadcast updated params
        self._all_gather_params()

        return result

    def _all_gather_params(self):
        """
        Broadcast updated parameters using zero-out + all_reduce.

        Each owner rank keeps its value, others zero out.
        All_reduce SUM gives everyone the owner's value.
        """
        if self.dp_size <= 1:
            return

        # Non-owners zero their copy
        for i, p in enumerate(self.all_params):
            if i % self.dp_size != self.dp_rank:
                p.data.zero_()

        # All_reduce SUM: owner contributes value, others contribute 0
        handles = []
        for p in self.all_params:
            handles.append(
                dist.all_reduce(p.data, op=dist.ReduceOp.SUM,
                               group=self.process_group, async_op=True)
            )
        for h in handles:
            h.wait()
```

### 3.3 Trainer Integration

```python
# File: ironcore/trainers/base_trainer.py

class BaseTrainer:
    def _build_model_and_optimizer(self):
        # ... existing model creation ...

        optimizer = get_optimizer(self.config, model, device_type=device)
        self.logger.info("Created Optimizer")

        # Wrap with DistributedOptimizer if requested
        if self.config.parallel.use_distributed_optimizer:
            from ironcore.optimizer import DistributedOptimizer
            optimizer = DistributedOptimizer(
                optimizer,
                process_group=get_data_parallel_group(),
            )
            self.logger.info("Wrapped optimizer with DistributedOptimizer")

        # ... rest of initialization ...
```

### 3.4 Process Group Initialization (No Changes Needed)

The existing `parallel_states.py` already provides:
- `get_data_parallel_group()` - Used by DistributedOptimizer
- `get_tensor_model_parallel_world_size()` - For TP-aware logic
- `get_data_parallel_world_size()` - For DP size

### 3.5 Checkpointing Strategy

```python
# File: ironcore/checkpointing/native.py

def save_checkpoint(config, model, optimizer, lr_scheduler, step):
    """
    Save checkpoint with DistributedOptimizer state handling.

    For DistributedOptimizer, optimizer states are partitioned:
    - Universal checkpoint: Gather all partitions to rank 0
    - Distributed checkpoint: Each rank saves its local partition
    """
    # Check if optimizer is wrapped
    is_distributed = hasattr(optimizer, 'optimizer')

    if is_distributed and not config.operation.save_dist_ckpt:
        # Universal checkpoint: gather all partitions
        optimizer_state_dict = _gather_distributed_optimizer_state(optimizer)
    else:
        # Standard save (local partition for distributed optimizer)
        optimizer_state_dict = _build_optimizer_state_dict(optimizer)
    ...

def load_checkpoint(config, model, optimizer, lr_scheduler, step):
    """
    Load checkpoint with DistributedOptimizer state handling.
    """
    is_distributed = hasattr(optimizer, 'optimizer')

    if is_distributed and load_dist_ckpt:
        # Load local partition directly
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif is_distributed:
        # Load full state and partition locally
        _partition_and_load_optimizer_state(optimizer, checkpoint)
    ...
```

---

## 4. Implementation Phases

### ✅ Phase 1: Configuration

**Completed:**
- [x] Added `use_distributed_optimizer: bool = False` to `ParallelConfig`
- [x] Validation: `use_distributed_optimizer` incompatible with `use_fsdp`
- [x] Validation: Warning when DP world size is 1

### ✅ Phase 2: Trainer Integration

**Completed:**
- [x] Optimizer wrapping in `_build_model_and_optimizer()`
- [x] Logging for distributed optimizer initialization

### ✅ Phase 3: Checkpointing Support

**Completed:**
- [x] `_gather_distributed_optimizer_states()` for universal checkpoints
- [x] `_partition_optimizer_states_for_load()` for loading
- [x] Detection of DistributedOptimizer via `_is_distributed_optimizer()`

### ✅ Phase 4: Testing

**Completed:**
- [x] Unit tests for single GPU (initialization, step, state_dict)
- [x] Multi-GPU tests (parameter partitioning, DDP integration, consistency)

### Phase 5: Documentation (TODO)

**Pending:**
- [ ] Update README with DistributedOptimizer usage
- [ ] Example YAML configs

---

## 5. Testing Strategy

### 5.1 2-GPU Test Configurations

```yaml
# Test 1: Basic DistributedOptimizer (DP=2)
parallel:
  use_distributed_optimizer: true
  world_size: 2
trainer:
  tensor_model_parallel_size: 1

# Test 2: TP + DistributedOptimizer (TP=2, DP=1)
# Note: No actual partitioning since DP=1, but tests compatibility
parallel:
  use_distributed_optimizer: true
  world_size: 2
trainer:
  tensor_model_parallel_size: 2

# Test 3: Compare with FSDP shard_grad_op (for reference)
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "shard_grad_op"
  world_size: 2
trainer:
  tensor_model_parallel_size: 1
```

### 5.2 Test Cases

| Test | Description |
|------|-------------|
| Partition correctness | Verify parameters correctly assigned to ranks |
| Gradient handling | Ensure DDP gradients work with partitioned optimizer |
| Parameter broadcast | Verify all-gather produces identical parameters |
| Checkpoint save | Universal and distributed formats |
| Checkpoint load | Resume training with same convergence |
| TP compatibility | No conflicts with tensor parallelism |
| EP compatibility | Expert parameters partitioned correctly |
| Convergence | Training converges same as baseline |

---

## 6. Memory Analysis

### 6.1 Memory Breakdown (7B Model, bf16)

| Component | DDP Only | DDP + DistributedOptimizer (DP=2) |
|-----------|----------|-----------------------------------|
| Parameters | 14 GB | 14 GB |
| Gradients | 14 GB | 14 GB |
| Optimizer States (fp32) | 56 GB | **28 GB** |
| Activations | ~10 GB | ~10 GB |
| **Total** | ~94 GB | **~66 GB** |

### 6.2 Savings Formula

```
Memory = P + G + O/N + A

Where:
- P: Parameter memory (14 GB for 7B bf16)
- G: Gradient memory (14 GB for 7B bf16)
- O: Optimizer state memory (56 GB for 7B, fp32 moments)
- N: DP world size
- A: Activation memory

Savings = O × (1 - 1/N)

For N=2: Saves 28 GB (50% of optimizer states)
For N=4: Saves 42 GB (75% of optimizer states)
For N=8: Saves 49 GB (87.5% of optimizer states)
```

---

## 7. Comparison Summary

| Approach | Optimizer State Memory | Gradient Memory | Param Memory | Comm Overhead |
|----------|------------------------|-----------------|--------------|---------------|
| DDP | 100% | 100% | 100% | All-reduce grads |
| DDP + DistributedOptimizer | 100%/N | 100% | 100% | + All-gather params |
| FSDP (shard_grad_op) | 100%/N | 100%/N | 100% | Reduce-scatter + all-gather |
| FSDP (full) | 100%/N | 100%/N | 100%/N | All-gather per layer |

**Recommendation**:
- Use `DistributedOptimizer` when optimizer states are the memory bottleneck
- Use `FSDP shard_grad_op` when both optimizer states and gradients are bottlenecks
- Use `FSDP full` when even parameters don't fit

---

## 8. Future Extensions

### 8.1 Pipeline Parallelism Integration

When PP is implemented:
- DistributedOptimizer operates within each PP stage
- Each stage has independent optimizer
- DP dimension is per-stage

### 8.2 CPU Offload (Optional Future)

Could add CPU offload for optimizer states:
```python
# Future configuration
parallel:
  use_distributed_optimizer: true
  distributed_optimizer_offload: true  # CPU offload
```

---

## 9. Summary

This design provides:

1. **Focused scope**: Optimizer state partitioning only
2. **Clear positioning**: Complements existing FSDP options
3. **Compatibility**: Works with TP and EP
4. **2-GPU testable**: DP=2 provides meaningful memory savings
5. **Minimal changes**: Primarily configuration and checkpointing

The existing `DistributedOptimizer` class already implements the core logic. Main work:
1. Add configuration option
2. Trainer integration (wrap optimizer)
3. Checkpointing for partitioned states
4. Testing with TP/EP
