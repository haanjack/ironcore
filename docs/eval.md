# Evaluation

## Built-in evaluators

Two evaluation mechanisms run during training:

1. **Perplexity / eval loss:** computed from the training data distribution by running `_eval_step()` on the held-out eval split. Logged as `eval_loss` and `eval_accuracy` (per-token accuracy).

2. **Task evaluators** (e.g. HellaSwag): loaded from `data.eval_datasets` in config. Each evaluator is a `Task` subclass that runs independently of the training data.

## `Task` interface

`Task` in `ironcore/eval/tasks/base_task.py` is the abstract base class for all evaluators.

```
process(model) → metrics dict
```

Internal steps:
1. `_preprocess()` — load and transform the dataset (using HuggingFace `datasets.map()`)
2. `_get_dataloader()` — create a `DataLoader` from the preprocessed dataset
3. For each batch: `_get_batch(batch)` → `_do_predict(model, inputs)` → accumulate
4. `_get_score(...)` → return metrics dict

## Dynamic loading

`get_evaluators()` in `ironcore/eval/evaluator.py` loads evaluators dynamically:

```python
module = importlib.import_module(f"ironcore.eval.tasks.{task_name}")
evaluator_class = next(cls for cls in vars(module).values()
                       if isinstance(cls, type) and issubclass(cls, Task) and cls != Task)
```

The task name must match the module filename (lowercase). For example, `{name: hellaswag}` loads `ironcore/eval/tasks/hellaswag.py` and finds the `HellaSwag` class.

## HellaSwag

`HellaSwag` in `ironcore/eval/tasks/hellaswag.py` measures commonsense NLI accuracy.

**Format:** prompt = `ctx_a + " " + ctx_b` + four candidate continuations. Each (prompt, continuation) pair is scored independently.

**Scoring:** per-token average cross-entropy loss. The candidate with the **lowest** loss is the model's prediction. Accuracy = fraction of questions where the predicted continuation matches the label.

**`_do_predict()`** calls `model(input_ids, labels=None)` to get logits, then computes `F.cross_entropy(..., reduction="mean")` per sample.

## Adding a new evaluator

1. Create `ironcore/eval/tasks/<name>.py` (lowercase, matches config `name` field).
2. Define a class that subclasses `Task`.
3. Implement:
   - `_preprocess(examples)` — HF datasets `map()` function; returns expanded dict.
   - `_get_batch(batch)` — extracts inputs and labels from a dataloader batch.
   - `_do_predict(model, tokenized_inputs)` — runs the model and returns per-sample scores.
   - `_get_score(...)` — aggregates scores and returns a metrics dict with at least `"score"`.
4. Add the task to `data.eval_datasets` in config.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.do_eval` | `false` | Enable evaluation during training |
| `trainer.eval_batch_size` | `null` | Batch size for evaluators (falls back to `micro_batch_size`) |
| `operation.eval_interval` | `100` | Evaluate every N steps (gated by `trainer.do_eval`) |
| `operation.eval_samples` | `100` | Number of samples for eval-loss evaluation |
| `data.eval_datasets` | `[]` | List of task evaluator configs |

### `data.eval_datasets` YAML example

```yaml
data:
  eval_datasets:
    - name: hellaswag
      source: Rowan/hellaswag
      max_samples: 1000
```

Each entry is a `DatasetConfig`; `name` selects the `Task` module under `ironcore/eval/tasks/`.
