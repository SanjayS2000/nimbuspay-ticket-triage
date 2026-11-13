# NimbusPay ticket-triage fine-tuning report

## Setup

- Base model: `Qwen/Qwen2.5-3B-Instruct` (3.1B parameters).
- Final method: NF4 4-bit QLoRA, rank 8, targeting Q/K/V/O and MLP projection
  modules. The run reported 14,966,784 trainable parameters (0.4827%).
- Training: 2 epochs, learning rate `2e-4`, micro-batch size 1, gradient
  accumulation 16, maximum sequence length 2,048.
- Loss masking: system and user tokens have label `-100`; loss is calculated
  only on the assistant JSON response tokens.
- Inference: Qwen native chat template, greedy `model.generate`,
  `max_new_tokens=200`, and whitespace stripping only.

## Data audit and cleaning

`audit_data.py` applied `json.loads` to all 1,445 supplied training assistant
completions. It found 25 invalid/truncated JSON labels and verified an exact
one-to-one match with the explicit, ticket-derived repair table in
`clean_data.py`. No dev or test labels were read during repair.

The cleaner records every change in `cleaning_log.csv`. It repaired malformed
JSON targets, the `prio` key typo, non-ISO dates, unnormalised NimbusPay IDs,
and priority values contradicting `SCHEMA.md`. It dropped 30 exact train/dev
input overlaps (comparison used inputs only) and 60 duplicate training inputs.
The final cleaned set contains 1,355 unique rows.

## Results

| Run | Exact match | Mean field accuracy | JSON valid |
|---|---:|---:|---:|
| Base model, fixed prompt | 0.0% | 0.1% | 74.5% |
| Base model, schema prompt reference | 11.0% | 77.5% | 100.0% |
| Final cleaned-data QLoRA model | **83.5%** | **85.2%** | 85.5% |

The supplied saved artifacts contain the baselines and final QLoRA run above.
Additional controlled-experiment results were not available in the preserved
artifacts and are therefore not reported as completed.

## Error analysis

The strongest slice was fraud (100% exact match, 15 examples). The weakest
slice was account access (52.6% exact match, 19 examples), followed by other
(68.8%) and KYC (70.8%). The principal failure mode was invalid output format:
29 of 33 dev errors were not valid single JSON objects.

| Dev ID | Observed failure | Likely cause |
|---|---|---|
| `dv-00002` | Generated `Pizza time!` instead of JSON | Format/control failure on a KYC question |
| `dv-00011` | Generated unrelated pizza-delivery prose | Severe off-task generation |
| `dv-00018` | Amount was 10× too large | Paise conversion error for ₹4,200 |
| `dv-00030` | Prepended `P3` before otherwise correct JSON | Strict JSON-format failure |
| `dv-00033` | Generated unrelated pizza prose | Severe off-task generation |
| `dv-00037` | Generated only `P1` | Incomplete structured response |
| `dv-00038` | Generated a JWT-like string | Severe off-task generation |
| `dv-00043` | Added explanatory text before/truncated JSON | Strict JSON-format failure |
| `dv-00047` | Repeated text and JSON fragments | Strict JSON-format failure |
| `dv-00052` | Amount was 10× too large | `lakh` conversion error |

With another week, I would prioritize format reliability through more clean
JSON-only examples and targeted augmentation for lakh/k amount conversion,
relative-date resolution, and account-access tickets. I would also run the
planned raw-vs-cleaned, LoRA-vs-QLoRA, rank, and epoch experiments with timing
and peak-memory logs.

## AI assistance disclosure

An AI coding assistant was used to inspect corruption patterns and draft the
audit, cleaning, training, and inference code. All cleaning decisions, metrics,
and submission contents were reviewed by the candidate.
