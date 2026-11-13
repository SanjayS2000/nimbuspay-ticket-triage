# NimbusPay triage fine-tuning report

## Setup

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Final load mode: 4-bit NF4 QLoRA, rank 8
- Trainable parameters: paste the `model.print_trainable_parameters()` output.
- Chat template and loss masking: Qwen native template; loss only on assistant
  JSON tokens.

## Data audit

Summarize `AUDIT_FINDINGS.md` and include `cleaning_log.csv`. State that exact
train/dev input overlaps were removed using inputs only. Explain why all 25
truncated labels were reconstructed from their own intact tickets and schema.

## Baselines and controlled experiments

| Run | Single change | Exact match | Mean field accuracy | Peak GPU memory | Minutes |
|---|---|---:|---:|---:|---:|
| Base model | Fixed prompt, no fine-tune | | | | |
| Base + rules | Rules in prompt; baseline reference only | | | | |
| 1 | Raw parseable data vs cleaned | | | | |
| 2 | 16-bit LoRA vs 4-bit QLoRA | | | | |
| 3 | Rank 4 vs rank 8 | | | | |
| 4 | One versus two epochs | | | | |

## Error analysis

Report the weakest fields and category slices from `score.py --json`. Include
ten concrete dev examples, the incorrect output, correct output, and cause.
Discuss likely next work: targeted augmentation for date arithmetic/ID
normalization, longer-context strategy, and format-reliability fine-tuning.

## AI assistance disclosure

Describe any use of AI tools accurately. Example: “An AI coding assistant was
used to inspect data corruption patterns and draft the cleaning/training code;
all rules, changes, metrics, and conclusions were reviewed by the candidate.”
