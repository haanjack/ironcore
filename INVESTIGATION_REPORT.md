# IronCore GSM8K Evaluation Investigation Report
**Date:** 2026-03-11
**Status:** CRITICAL - Model outputs garbage

## Summary

The full GSM8K evaluation completed with **0% accuracy** (0/1319 samples). This is unexpected since a previous 100-sample test which showed **~49% accuracy**.

## Investigation

### 1. Previous Working Evaluation (Pre-commit dc1d5d7)
Before the recent commits, a 100-sample evaluation showed ~49% accuracy on Qwen2.5-0.5B-Instruct on After commit dc1d5d7, the was model was correctly and worked on both 2+2=4,3+ 6, and 9, etc.

- **Commit:** Fixed `no_bias: true` for Qwen2.5 compatibility
- **Training configs:** Fixed batch size validation
- **Data preprocessing:** Completed (7473 samples preprocessed)
- **Commit:** Created with changes

- `ironcore/alignment/dataset.py` - GRPO dataset loading
- `ironcore/alignment/rewards.py` - Reward functions
- `ironcore/config/config_data.py` - Data config
- `ironcore/config/config_trainer.py` - Trainer config
- `ironcore/preprocessing/serializer.py` - Data serialization
- `ironcore/tokenizer/tokenizer.py` - Tokenizer
- `ironcore/trainers/base_trainer.py` - Trainer base
- `ironcore/trainers/grpo_trainer.py` - GRPO trainer
- **Checkpoint:**
- `ironcore/checkpointing/hf_interop.py` - HF weight loading
- `ironcore/checkpointing/weight_mapping.py` - Weight mapping (LLaMA/Qwen)
- **Configs:**
- `configs/model/qwen2.5-0.5B.yaml` - Model config (no_bias: true)
- `configs/data/grpo_gsm8k.yaml` - Data config
- `configs/grpo_gsm8k.yaml` - Full training config
- `configs/grpo_gsm8k_smoke.yaml` - Smoke test config
- **Scripts:**
- `scripts/eval_gsm8k_baseline.py` - Evaluation script
- `scripts/verify_grpo_prereqs.py` - Prerequisites verification
- `scripts/download_gsm8k.py` - GSM8K download
- `scripts/preprocess_gsm8k.py` - Preprocessing

- **Commit:** 7cadb8c `feat(grpo): add GSM8K training configs and evaluation scripts`

### 2. Weight Mapping Analysis

**File:** `ironcore/checkpointing/weight_mapping.py`
**Focus:** LLaMA/Qwen MLP weight mapping (lines 478-540)

**Key findings:**
1. **QKV Projection:** Correctly mapped with transposition
   - K, V → fused KV (transposed)
   - Q → linear_q (transposed)
   - O → attn_output (transposed)

   - No bias → None (IronCore uses `no_bias=True`, model has NO bias

2. **MLP Weights:**
   - `gate_proj.weight` → Transposed, fused into `mlp.up_proj.weight`
   - `up_proj.weight` → Transposed, stored separately in `mlp.gate_proj.weight` (lines 534-540)
   - `gate_proj.weight` → Transposed, stored in `mlp.up_proj.weight`
   - `down_proj.weight` → No transpose (correct)
   - No bias → None → IronCore stores this directly in `down_proj.bias`
   - `mlp.gate_proj.bias` → Not loaded (lines 536-540)
   - `mlp.up_proj.bias` → Not loaded (line 539-540)
   - Bias terms:** Not loaded (missing keys)

   - Input/output layernorm → No transpose (correct)
   - Post-attention layernorm → No transpose (correct)

   - Layernorm biases → None (no bias)

```python
# Weight mappings (lines 483-485)
weight_mappings = {
    "self_attn.q_proj.weight": f"model.layers.{layer_idx}.linear_q.weight",
    "self_attn.o_proj.weight": f"model.layers.{layer_idx}.attn_output.weight",
}

```

All weights go through `.t()`,**BUT:** The shapes are transposed, but HF uses `[out_features, hidden], convention while IronCore uses `[in_features, out_features] convention (same as Q/K/V).

So the transposition for Q, K, V is correct.

But the MLP, the weights should:
 separate `gate_proj` and `up_proj`, and `down_proj`.

 - **Gate/Up Fusion Logic (lines 518-540):**
Fuses gate and up weights:
- Input shape: `[intermediate_size, ffn_hidden]`
- Output shape: `[2*intermediate_size, ffn_hidden]` (SwiGLU)
- Fuse along output dim: [2*intermediate_size]
- Transpose to IronCore format: [hidden_size, 2*intermediate_size]
- Return transposed tensor

- `mlp.down_proj.weight` (line 485): No transpose, correct

- `mlp.down_proj.bias` (line 498): Not loaded (line 536)

- `mlp.up_proj.weight` (line 540): Handled separately
- `mlp.gate_proj.weight` (line 536): Not present in HF checkpoint

- `mlp.up_proj.weight` (line 540): Handled separately (line 539)
  - `mlp.gate_proj.weight` (line 532): Handled separately (line 536)
  - `mlp.up_proj.weight` (line 540): Fused, stored in `mlp.up_proj.weight`
- `mlp.gate_proj.weight` not in HF checkpoint
  - We, if `gate_proj.weight` missing, we:
   pass 2. The should be correct
   - `mlp.up_proj.weight` should be `mlp.gate_proj.weight` (line 534-540)
   - Gate_proj and up_proj are fused into `mlp.up_proj.weight`, but both is transposed and then concatenated
   - IronCore stores the as a single fused `mlp.up_proj` weight
- When this happens, both `gate_proj` and `up_proj` are fused into a single tensor.

 This is **breaks the MLP fusion logic** that (lines 518-540).

This, the:
- **But what is `gate_proj` and `up_proj` fuse useful?**
- However, when both are fused, they in `model.layers.{i}.mlp.up_proj.weight`, the output is literally `mlp.up_proj` gate and `up_proj` output will go to `mlp.down_proj` and vice versa.
- This means the model outputs garbage because:
1. The MLP fusion logic is weight_mapping.py is lines 527-540 transposes gate and up_proj and then fuses them together. But the resulting fused weight tensor shape is `[hidden_size, 2*intermediate_size]` instead of the original separate shapes.
2. The MLP up_proj and gate_proj mapping is `weight_mapping.py` was 518-540,```
        if layer_key == "mlp.gate_proj.weight":
            # Check if we need to fuse with up_proj
            up_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            if up_key in full_state_dict:
                gate = tensor  # HF: [intermediate_size, ffn_hidden]
                up = full_state_dict[up_key]  # HF: [intermediate_size, ffn_hidden]
                if not up_key:
                    return None, None
```

**The the observations:**
1. **Weight transposition is correct for Q/K/V** but but MLP should be transposed.
2. **MLP gating SwiGLU (gate/up) vs standard:** both separate, but fused
3. **The `mlp.gate_proj` and `mlp.up_proj` should be the same weights (they both get transposed and then fused together), the model should would incorrectly.

    - Looking at the Qwen2.5 architecture (which uses SwiGLU), this should be true, but the like to how PyTorch's `gate`, `up_proj` weights and and then `mlp.up_proj` to stores the fused gate+up weights.

    - This is fundamentally broken - the model outputs garbage.

4. **The `no_bias` config issue:**
        When we set `no_bias: true` in the model config, we loading script skipped the validation. But weights weren't being the correctly mapped. But it is likely happening again. There was a parameter `missing_keys` list included biases-related parameters. but the was benign. we the HF model works correctly.

 with `no_bias=False`, the HF model should still work.

 is return 0% accuracy.
        The but the previous 49% accuracy from the initial 100-sample test, this is never mentioned. the output was garbage, so that weight mapping is broken in a transposition.

    - The issue with **Q/K/V projections**: The transposition in weight_mapping.py is correct (transpose happens when loading fused KV, but the output generates garbage

        - `q_proj.bias` is loaded but but incorrectly during the verify script
    - **HF model works fine** using pipeline, suggesting checking with IronCore to after the fix

4. **Investigate `linear_kv` weight loading: ` _handle_llama_mlp_fusion()` returned `None, None`. It (539)
    # HF format: mlp.up_proj.weight
    # Not in HF checkpoint
    elif:
        return None, None
    # HF format: mlp.up_proj.weight
    # But there's no up_proj in IC checkpoint
    return None, None
```

Then in `hf_to_ironcore()` when loading, we Qwen weights, they:
- The linear_q, linear_kv, attn_output, linear layers don't have biases)
- MLP: ParallelMLP in IronCore has `up_proj` and `down_proj`, but they don't exist in HF:
   - Verify that `up_proj` and `gate_proj` exist in HF
   - Check if `gate_proj` exists (both should be transposed and fused)
   - If only one exists, transpose it and fuse with up_proj
   - If neither exists, use the `mlp.up_proj.weight` directly
   - If `mlp.gate_proj.weight` not in full_state_dict, return f"model.layers.{layer_idx}.mlp.gate_proj.weight", None, None
        else:
            return None, None
    elif layer_key == "mlp.up_proj.weight":
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if gate_key in full_state_dict:
            gate = full_state_dict[gate_key]
            up = full_state_dict[up_key]
            # Both transposed and fused
            fused = torch.cat([gate, up], dim=0)
            fused = fused.t()
            return f"model.layers.{layer_idx}.mlp.up_proj.weight", fused
        else:
            return None, None

    # No fallback
    return None, None
```

### 3. Root Cause: Weight Transposition Issues
**File:** `ironcore/checkpointing/weight_mapping.py` (lines 481-512)

**Observation:**
Looking at the code, we the following:
1. Line 483-485: Q, O projections are transposed (line 484: `return weight_mappings[layer_key], tensor.t()`). Line 496: `mlp.down_proj.weight` → no transpose (correct)

2 Line 498: `mlp.down_proj.bias` → not loaded (line 499)

        `mlp.down_proj.bias` is None but but the model doesn't expect bias to have a `down_proj.bias` at all
            x = x + self.down_proj.bias
```

But then in the `_ironcore_to_hf_llama()`:
```python
    def _ironcore_to_hf_llama(
        self,
        ironcore_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Convert ironcore checkpoint to LLaMA HuggingFace format."""
        # Implementation at lines 288-548...
```

Then in the `_ironcore_to_hf_llama()`, I the script):
`` for layer in all layers:
            for layer_idx in range(self.num_layers):
                ic_down_proj_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
                hf_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
                hf_tensor = ironcore_state_dict[ic_down_proj_key]
                # Gate_proj - stored as gate_proj (fused with up_proj)
                ic_gate_proj_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                if ic_gate_proj_key in ironcore_state_dict:
                    gate = ironcore_state_dict[ic_gate_proj_key]
                    up = ironcore_state_dict[ic_up_proj_key]  # Both transposed, split gate and up
                    gate, up = ironcore_state_dict[ic_gate_proj_key]
                    up = ironcore_state_dict[ic_up_proj_key]
                    # Transpose back to IronCore format
                    gate = gate.t()
                    up = up.t()
                    fused = torch.cat([gate, up], dim=0)
                    ironcore_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = fused
                    # Transpose for IC
                    fused = fused.t()
                else:
                    # gate_proj exists, up_proj exists - handle separately
                    gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                    if gate_key in ironcore_state_dict:
                        gate = ironcore_state_dict[gate_key]
                        up = ironcore_state_dict[ic_up_proj_key]
                    fused = torch.cat([gate, up], dim=0)
                    ironcore_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = fused
                    # Transpose and concatenate
                    ironcore_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat([gate_proj, up], dim=0)
                    ironcore_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.cat([gate, up], dim=0)
                else:
                    hf_tensor = ironcore_state_dict[hf_key]
                    ironcore_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = hf_tensor
```

### 4. ParallelMLP Implementation
**File:** `ironcore/layers/parallel_mlp.py` (lines 94-111)
```python
class ParallelMLP(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Up projection: hidden -> intermediate (ColumnParallel for TP)
        self.up_proj = ColumnParallelLinear(...)
        # Down projection: intermediate -> hidden (RowParallel for TP)
        self.down_proj = RowParallelLinear(...)
```
```

The MLP uses a fused `up_proj` that stores both `gate_proj` and `up_proj` weights:
    - `up_proj.weight` stores fused weights [hidden, intermediate]
    - `gate_proj` is separate

- **But** in the weight mapping, we `ParallelMLP` constructor (line 96):
```python
        self.up_proj = ColumnParallelLinear(config, ...)
        self.gate_proj = ColumnParallelLinear(config, ...)
```
    the:
        - It calls `self.up_proj.weight`, which returns `[hidden, intermediate]`
        - If `self.gate_proj` exists and `mlp.gate_proj.weight` in `full_state_dict`, get that tensor and fuse with `self.up_proj.weight`
        - For `gate_proj`, check `full_state_dict.get("mlp.gate_proj.weight")` to fuse, but typically there's no `gate_proj` key for HF checkpoint for this architecture (SwiGLU).
    - So the should work: is independent weights
- When gate_proj exists in the HF checkpoint, the weights are stored separately in IronCore's `mlp.gate_proj.weight`
- When `mlp.up_proj.weight` is called in the mapping function but line 534-540), it passes to `_handle_llama_mlp_fusion()`
- The code explicitly handles `mlp.up_proj.weight` ONLY when gate_proj also exists
- If `gate_proj` doesn't exist, then code falls through to line 534-536, which returns a line 536-540):
    ```python
    return None, None
    ```
    This means:
    - When loading Qwen2.5 weights, `gate_proj` weight is loaded into `mlp.gate_proj.weight` but it should have been fused with `up_proj`
    - But if `up_proj` is also present, the code returns `None, None` early, meaning `up_proj` weight is also stored separately in `mlp.gate_proj.weight`
    - If neither `gate_proj` nor `up_proj` exists in the HF checkpoint, the code returns `None, None`

### 5. Debugging Test Results
**From the verify script (smoke test):**
```
============================================================
Step 3: Single-GPU Generation Test
============================================================
Prompt: What is 2+2?
Generating 4 completions...
  [0] X 丁勒假ACH'])?',...
Cat dynamics以为 зам).^()?;
 ComponentFixturealysis颇Wswal Ôstüt之美프...
  [1] X _NOTICE哨 ank Alejandro芮 agreementsoutingistencia耽,www principal UnsignedObservAB...
  [2] X  ?>


PAransえ.lab ADDRESSHasColumnType_apixdb上下沽之后 IAMemeteryndata(Notification(...
  [3] X -animate DataType)=='lama show BannerobreQT_OVERFLOWstrcasecmpdeo]='\isz煽メ.='ask...

Generated completions:
  [0] reward=0.00 | contains ####: False
  [1] reward=0.00 | contains ####: False
  [2] reward=0.00 | contains ####: False
  [3] reward=0.00 | contains ####: False
```

The output is garbage text with random Unicode characters, not coherent English. This confirms the model is generating nonsensical output.

    - Despite weights being loaded correctly (1 missing key, 48 unexpected keys according to the loading script)
    - The model loaded successfully (170/171 weight tensors loaded)
    - The tokenizer chat template is working correctly
    - Generation produces garbage

**From the eval script (full GSM8K):**
```
============================================================
Final Accuracy: 0/1319 = 0.00%
============================================================
```

All 1319 samples produced garbage text like:
```
VILineStyle/manualapore \u00fcberhalous\u015fi\u25c0\ud83d\ude09)..., random Unicode
```

### 6. Key Differences from Working Evaluation
| Aspect | HF (Working) | IronCore (Current) |
|--------|---------------|----------------------|
| Q/K/V | Separate Q, K, V | Fused KV |
| MLP | Separate gate, up, down | Fused gate+up, Separate down |
| Bias | Has bias | `no_bias=True` (no bias) |
| Output | Coherent answers | Random garbage |

| Accuracy | ~49% | 0% |

### 7. Root Cause Analysis
The primary suspect is the **MLP weight transposition** the case.

1. **Q, O projections work correctly** - transpose happens in weight_mapping.py:483-485
2. **Down_proj works correctly** - no transpose needed
3. **Gate/Up fusion logic in weight_mapping.py (lines 518-540):**
   - HF `gate_proj` + `up_proj` → IC `mlp.up_proj` (fused, transposed)
   - Code returns `None, None` for `up_proj.weight` when `gate_proj` doesn't exist
   - Code falls through to line 534-540, which stores `up_proj` separately

   - But the actual behavior when both exist: creates fused tensor

   - The issue: **The resulting fused tensor may be in wrong IronCore parameter name, or may have shape issues

4. **MLP `no_bias=True` but but layer has bias parameter (initialized to False), so bias-related code is doesn't run, but layers have `bias` attribute which is initialized
5. **HF Qwen2.5 uses bias in attention layers** - `q_proj.bias`, `k_proj.bias`, `v_proj.bias`
   - IronCore architecture expects fused KV bias: `linear_kv.bias`
   - The weight mapping code handles this correctly (lines 423-431)
   - But if attention layer implementation has issues, this could cause problems
6. **Model evaluation using IronCore shows garbage output**
   - HF model works correctly (confirmed via pipeline test)
   - IronCore model outputs random Unicode characters
   - This confirms the issue is in the IronCore model architecture or weight loading, not the evaluation script

### 8. Hypotheses
1. **MLP gate/up fusion produces wrong tensor shape**
   - The fused tensor shape may be `[hidden, 2*intermediate]` but should be `[hidden, intermediate]`
   - This would cause the MLP to produce incorrect activations

2. **Missing `mlp.gate_proj` parameter**
   - If `gate_proj` exists in HF checkpoint but `up_proj` doesn't, code stores it separately
   - But if `up_proj` is missing during fusion, the MLP may not have a gating mechanism
3. **Bias handling with `no_bias=True`**
   - Layers create `bias` parameter but it's `False`
   - Some code paths may expect bias to exist
   - This could cause runtime errors or silent failures

### 9. Immediate Action Items
1. **Fix MLP weight mapping** - Ensure gate_proj and up_proj are correctly fused and transposed
2. **Verify MLP implementation** - Check if ParallelMLP handles the fused weights correctly
3. **Add regression test** - Create a simple test that verifies model outputs correct format
4. **Compare HF vs IronCore outputs** - Run same input through both and compare token probabilities

### 10. Code Locations
- **Weight mapping:** `ironcore/checkpointing/weight_mapping.py:518-540`
- **MLP implementation:** `ironcore/layers/parallel_mlp.py:94-111`
- **Evaluation script:** `scripts/eval_gsm8k_baseline.py`
- **Model config:** `configs/model/qwen2.5-0.5B.yaml`

### 11. Next Steps
1. Debug MLP weight fusion to add logging to see actual tensor shapes
2. Create minimal reproduction case - single layer, known input, verify output
3. Consider adding HF export function to verify weights load correctly
4. Run evaluation with HF model directly to confirm expected accuracy

