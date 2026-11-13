"""Audit and clean NimbusPay training labels using only train.jsonl and SCHEMA.md.

The script deliberately reads dev INPUTS only to remove exact input leakage.  It
never reads dev assistant labels and does not use dev data to create a label.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
KEY_SET = set(KEYS)
LEGAL = re.compile(r"\b(RBI|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.I)

# These are the 25 labels that were physically truncated in the supplied
# training file. Every value below is recovered from that row's intact ticket
# using SCHEMA.md; do not infer values from dev/test labels.
TRUNCATED_REPAIRS = {
    "tr-00060": ["kyc", "P3", None, None, None, None, "en", False],
    "tr-00095": ["other", "P3", None, None, None, None, "en", False],
    "tr-00125": ["payment_failed", "P3", 392500, "2026-08-12", "NP6948115654", "card", "en", False],
    "tr-00172": ["kyc", "P3", None, None, None, None, "en", False],
    "tr-00267": ["kyc", "P3", None, None, None, None, "en", False],
    "tr-00357": ["account_access", "P2", None, None, None, None, "en", False],
    "tr-00478": ["refund", "P1", 16041600, None, "NP1893758181", None, "en", True],
    "tr-00497": ["payment_failed", "P3", 242300, "2026-06-25", None, None, "en", False],
    "tr-00558": ["payment_failed", "P1", 15137000, "2026-07-30", "NP8621659965", "netbanking", "en", True],
    "tr-00605": ["payment_failed", "P3", 37700, "2026-07-25", None, None, "en", False],
    "tr-00674": ["payment_failed", "P3", 338500, "2026-09-16", None, "upi", "en", False],
    "tr-00735": ["payment_failed", "P1", 29687900, None, "NP7764977551", "netbanking", "en", True],
    "tr-00765": ["refund", "P3", 13600, "2026-06-27", "NP4193555354", "card", "hi-en", False],
    "tr-00809": ["payment_failed", "P3", 97000, "2026-07-05", "NP8738912683", "card", "hi-en", False],
    "tr-00824": ["offers", "P3", 27100, None, None, None, "hi-en", False],
    "tr-00846": ["payment_failed", "P3", 23000, None, "NP4657109851", "wallet", "en", False],
    "tr-00958": ["refund", "P3", 59700, None, None, "wallet", "en", True],
    "tr-00960": ["account_access", "P1", None, None, None, None, "hi-en", True],
    "tr-00975": ["fraud", "P1", 94400, "2026-08-29", None, "card", "en", True],
    "tr-00978": ["account_access", "P2", None, None, None, None, "en", False],
    "tr-00979": ["payment_failed", "P3", 83325, "2026-08-07", "NP7335161087", "card", "en", True],
    "tr-00980": ["other", "P3", None, None, None, None, "en", False],
    "tr-01070": ["payment_failed", "P2", 1787400, "2026-08-11", "NP2339235751", "card", "en", False],
    "tr-01087": ["offers", "P3", None, None, None, "card", "en", False],
    "tr-01199": ["payment_failed", "P3", 141099, None, None, "upi", "en", False],
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expected_priority(label, ticket: str) -> str:
    amount, category = label["amount_paise"], label["category"]
    if category == "fraud" or (isinstance(amount, int) and amount >= 5_000_000) or LEGAL.search(ticket):
        return "P1"
    if category == "account_access" or (category in {"payment_failed", "refund"} and isinstance(amount, int) and amount >= 500_000):
        return "P2"
    return "P3"


def normalize_id(value):
    if value is None:
        return None
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return compact


def normalize_date(value):
    if value is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return datetime.strptime(value, "%d/%m/%Y").strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out", default="artifacts/train_clean.jsonl")
    ap.add_argument("--log", default="artifacts/cleaning_log.csv")
    args = ap.parse_args()

    train, dev = read_jsonl(Path(args.train)), read_jsonl(Path(args.dev))
    # Input-only comparison: protects the validity of dev without consuming its labels.
    dev_inputs = {row["messages"][1]["content"] for row in dev}
    seen_inputs, cleaned, log = set(), [], []

    for row in train:
        row = json.loads(json.dumps(row))  # do not mutate the raw input object
        row_id, ticket = row["id"], row["messages"][1]["content"]
        if ticket in dev_inputs:
            log.append([row_id, "drop", "exact train/dev input overlap; dev labels were not read"])
            continue
        if ticket in seen_inputs:
            log.append([row_id, "drop", "exact duplicate of an earlier retained training input"])
            continue
        seen_inputs.add(ticket)

        raw = row["messages"][-1]["content"]
        if row_id in TRUNCATED_REPAIRS:
            label = dict(zip(KEYS, TRUNCATED_REPAIRS[row_id]))
            log.append([row_id, "fix", "truncated assistant JSON reconstructed from intact ticket under SCHEMA.md"])
        else:
            label = json.loads(raw)
            if "prio" in label:
                label["priority"] = label.pop("prio")
                log.append([row_id, "fix", "renamed invalid key prio to priority"])
            old_date = label.get("txn_date")
            new_date = normalize_date(old_date)
            if new_date != old_date:
                label["txn_date"] = new_date
                log.append([row_id, "fix", "normalized txn_date to required YYYY-MM-DD"])
            old_id = label.get("txn_id")
            new_id = normalize_id(old_id)
            if new_id != old_id:
                label["txn_id"] = new_id
                log.append([row_id, "fix", "normalized NimbusPay transaction ID"])

        corrected_priority = expected_priority(label, ticket)
        if label["priority"] != corrected_priority:
            label["priority"] = corrected_priority
            log.append([row_id, "fix", "recomputed priority from explicit SCHEMA.md rules"])
        if set(label) != KEY_SET:
            raise ValueError(f"{row_id}: unexpected keys after cleaning: {sorted(label)}")
        row["messages"][-1]["content"] = json.dumps(label, ensure_ascii=False, separators=(",", ":"))
        cleaned.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(args.out), cleaned)
    with Path(args.log).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action", "reason"])
        writer.writerows(log)
    print(f"Wrote {len(cleaned)} cleaned rows and {len(log)} log entries.")
    print(f"Dropped {sum(x[1] == 'drop' for x in log)} rows; fixed {sum(x[1] == 'fix' for x in log)} rows.")


if __name__ == "__main__":
    main()
