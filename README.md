# NimbusPay Ticket Triage — Colab QLoRA

Fine-tuning submission for the NimbusPay ticket-triage assessment.

## Final model

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning method: NF4 4-bit QLoRA
- LoRA rank: 8
- Loss: assistant-response tokens only
- Inference: Qwen native chat template, greedy decoding,
  `max_new_tokens=200`, whitespace stripping only

## Best development result

- JSON validity: 85.5%
- Exact match: 83.5%
- Mean field accuracy: 85.2%

## Repository contents

- `adapter/` — final standard PEFT LoRA adapter
- `predictions_test.jsonl` — 400 hidden-test predictions
- `cleaning_log.csv` — all training-data fixes and dropped rows
- `notebook.ipynb` — Colab training and evaluation notebook
- `report.md` — audit, experiments, and error analysis
- Python scripts — audit, cleaning, training, inference, baseline, and scoring code

## Reproducibility

The final adapter was trained on a free Google Colab T4 GPU. The project uses
only the supplied training data and `SCHEMA.md` for data cleaning. No supplied
training, development, or test dataset is included in this repository.