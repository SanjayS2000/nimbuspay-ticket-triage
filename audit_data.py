"""Reproducible, read-only audit of malformed assistant labels in train.jsonl.

This script does not repair data and never reads dev/test labels. Its purpose is
to document why the explicit TRUNCATED_REPAIRS table in clean_data.py exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from clean_data import TRUNCATED_REPAIRS


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--csv", default="artifacts/audit_invalid_json.csv")
    ap.add_argument("--summary", default="artifacts/audit_summary.json")
    args = ap.parse_args()

    findings = []
    for row in read_jsonl(Path(args.train)):
        output = row["messages"][-1]["content"]
        try:
            json.loads(output)
        except json.JSONDecodeError as exc:
            findings.append({
                "id": row["id"],
                "error": exc.msg,
                "error_position": exc.pos,
                "output_characters": len(output),
                "keys_started": "|".join(re.findall(r'"([a-z_]+)"\s*:', output)),
                "label_prefix": output[:160],
                "repair_present": row["id"] in TRUNCATED_REPAIRS,
            })

    found_ids = {row["id"] for row in findings}
    repair_ids = set(TRUNCATED_REPAIRS)
    summary = {
        "rows_scanned": len(read_jsonl(Path(args.train))),
        "invalid_assistant_json_count": len(findings),
        "invalid_assistant_json_ids": sorted(found_ids),
        "repair_table_count": len(repair_ids),
        "unmapped_invalid_ids": sorted(found_ids - repair_ids),
        "unused_repair_ids": sorted(repair_ids - found_ids),
        "result": "pass" if found_ids == repair_ids else "fail",
        "method": "json.loads each train assistant completion; no dev/test labels read",
    }

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(findings[0]) if findings else ["id"])
        writer.writeheader()
        writer.writerows(findings)
    with Path(args.summary).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    if summary["result"] != "pass":
        raise SystemExit("Audit/repair-table mismatch: review before training.")


if __name__ == "__main__":
    main()
