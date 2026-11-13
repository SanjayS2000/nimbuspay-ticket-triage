# Data audit findings

This document records the reproducible findings from the supplied pack before
any training. Run `python audit_data.py` to reproduce the invalid-JSON evidence,
then run `python clean_data.py` to regenerate the cleaning log.

## Input and split integrity

- `train.jsonl`: 1,445 rows.
- `dev.jsonl`: 200 labelled rows.
- `test_inputs.jsonl`: 400 unlabelled rows.
- The training file contains 60 exact duplicate-input pairs. Labels agree
  inside each pair.
- Thirty training inputs are exact matches of dev inputs and their labels also
  agree. They are excluded from cleaned training by comparing **only input
  text**; dev assistant labels are not read.
- There is no exact input overlap between test and either train or dev.

## Label integrity

The raw training targets contain several synthetic corruption patterns:

| Finding | Count | Cleaning action |
|---|---:|---|
| Physically truncated / invalid JSON target | 25 | `audit_data.py` detects them; reconstruct from intact ticket and `SCHEMA.md` |
| `prio` instead of `priority` | 70 | Rename key |
| Non-ISO transaction date | 30 | Convert `DD/MM/YYYY` to `YYYY-MM-DD` |
| Non-normalized NimbusPay ID | 20 | Remove separators and uppercase |
| Priority violation | 135 after parsing/repair | Recompute from the explicit priority rules |
| Exact dev-input overlap | 30 | Drop from training |
| Repeated retained training input | 60 | Drop duplicate excess row |

Some rows carry multiple fixes, so the final CSV has 280 `fix` entries and 90
`drop` entries. The cleaned dataset has **1,355** unique input rows.

## Long-context behaviour

Forty train, three dev, and eleven test tickets exceed 1,000 characters; the
largest is about 8,800 characters. These are prior-email chains containing
quoted boilerplate. The schema says quoted lines beginning `>` cannot supply a
transaction date. The training code right-truncates only the few overly long
user prompts while preserving the initial ticket, system prompt, and complete
assistant JSON target; it prints a count during tokenization.

## Validation after cleaning

`clean_data.py` asserts that every retained label has exactly the required keys.
It then serializes compact valid JSON. The independent scorer remains the
authority for dev exact match and field accuracy.

`artifacts/audit_invalid_json.csv` records the parse error, error position,
started keys, and a short raw-label prefix for each malformed completion.
`artifacts/audit_summary.json` must report `"result": "pass"` before training.
