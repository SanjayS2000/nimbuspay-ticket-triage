"""Create the deliberately unclean baseline dataset; only unparsable labels are excluded."""
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/train.jsonl"); ap.add_argument("--output",default="artifacts/train_raw_parseable.jsonl")
args=ap.parse_args(); Path(args.output).parent.mkdir(parents=True,exist_ok=True)
kept=0
with open(args.input,encoding="utf-8") as src, open(args.output,"w",encoding="utf-8") as dst:
    for line in src:
        row=json.loads(line)
        try: json.loads(row["messages"][-1]["content"])
        except json.JSONDecodeError: continue
        dst.write(json.dumps(row,ensure_ascii=False)+"\n"); kept+=1
print(f"Wrote {kept} parseable but otherwise raw rows to {args.output}.")
