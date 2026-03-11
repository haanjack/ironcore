# IronCore Experiment Reporting Protocol

When instructed to "run an experiment" or "verify performance", you MUST follow this reporting standard.

## Report Generation Rules
1. Always save the report in `reports/YYYY-MM-DD_[ExperimentName].md`.
2. Do not summarize loosely; use the specific headers below.

## Report Template
# Experiment Report: [Task Name]
**Date:** ...
**Commit Hash:** ...

## 1. Objective
Subject: Generate Structured Experiment Reports for Documentation

For EACH of the tasks (Task 1, 2, and 3) above, after executing the code and obtaining the results, you must generate a Markdown Report in the reports/ directory.

This report will serve as the raw material for my technical blog. Do not just dump logs; analyze the results.

Report Format (Template):
Please follow this structure strictly:
```markdown
# Experiment Report: [Task Name]
**Date:** [YYYY-MM-DD]
**Commit Hash:** [Current Git Hash or Version]
```
1. Objective
Briefly explain what we are testing and why. (e.g., "Verifying gradient detachment for unselected experts to ensure correct routing.")

2. Experimental Setup
Hardware: 2x RTX 3090

Model Config: [Hyperparameters used: layers, hidden_size, num_experts, etc.]

Key Libraries: [Pytorch version, CUDA version if applicable]

3. Results & Metrics (The Evidence)
(Use Tables for quantitative data)
| Metric | Value | Baseline | Improvement/Status |
| :--- | :--- | :--- | :--- |
| Latency (ms) | ... | ... | ... |
| VRAM Usage (MB)| ... | ... | ... |

(Insert any generated plot images here, e.g., ![Heatmap](./plots/heatmap_step_100.png))

4. Analysis & Key Findings
(Critical Section: Interpret the data)

Observation: [e.g., "Throughput dropped by 15% compared to Dense model, but parameter count increased by 800%."]

Interpretation: [e.g., "This indicates that the communication overhead is well-hidden by the compute intensity..."]

Issues Found: [Any anomalies, OOMs, or unexpected behaviors]

5. Conclusion
[Pass / Fail]

[Next steps or required optimizations]


**Action Item:**
Create the reports/ folder if it doesn't exist.

Save the report as reports/Exp_[TaskNumber]_[TaskName].md.