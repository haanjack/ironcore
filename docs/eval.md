# Evaluation

## Built-in evaluators

Two evaluation mechanisms run during training when `trainer.do_eval: true`:

1. **Eval loss / perplexity** — runs on the held-out eval split of the training data. Logged as `eval_loss` and `eval_accuracy` (per-token accuracy).
2. **Task evaluators** (e.g., HellaSwag) — loaded from `data.eval_datasets`. Each runs independently of the training data distribution.

## HellaSwag

Measures commonsense NLI accuracy. Each question provides a context and four candidate continuations; the model picks the continuation with the lowest per-token cross-entropy loss.

Enable via:

```yaml
trainer:
  do_eval: true

data:
  eval_datasets:
    - name: hellaswag
      source: Rowan/hellaswag
      max_samples: 1000
```

## Adding a custom evaluator

1. Create `ironcore/eval/tasks/<name>.py` (lowercase filename; must match the `name` field in config).
2. Subclass `Task` from `ironcore/eval/tasks/base_task.py`.
3. Implement:
   - `_preprocess(examples)` — HF `datasets.map()` function; returns expanded dict.
   - `_get_batch(batch)` — extracts inputs and labels from a dataloader batch.
   - `_do_predict(model, inputs)` — runs the model and returns per-sample scores.
   - `_get_score(...)` — aggregates scores and returns a metrics dict with at least `"score"`.
4. Add the task to `data.eval_datasets` in config.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.do_eval` | `false` | Enable evaluation during training |
| `trainer.eval_batch_size` | `null` | Batch size for evaluators (falls back to `micro_batch_size`) |
| `operation.eval_interval` | `100` | Evaluate every N steps |
| `operation.eval_samples` | `100` | Number of samples for eval-loss evaluation |
| `data.eval_datasets` | `[]` | List of task evaluator configs (`name`, `source`, `max_samples`) |
