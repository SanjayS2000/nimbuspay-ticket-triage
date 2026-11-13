"""Generate the two required no-fine-tuning dev baselines."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def rows(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/dev.jsonl"); ap.add_argument("--output",required=True)
    ap.add_argument("--model",default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--with-rules",action="store_true"); ap.add_argument("--schema",default="SCHEMA.md")
    a=ap.parse_args(); tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.pad_token or tok.eos_token
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=q,torch_dtype=torch.float16,device_map="auto").eval()
    rules=Path(a.schema).read_text(encoding="utf-8") if a.with_rules else ""
    with open(a.output,"w",encoding="utf-8") as f:
        for r in rows(a.input):
            messages=[dict(m) for m in r["messages"][:2]]
            if a.with_rules: messages[0]["content"] += "\n\nUse this labelling specification:\n"+rules
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            x=tok(prompt,return_tensors="pt").to(model.device)
            with torch.inference_mode(): y=model.generate(**x,max_new_tokens=200,do_sample=False,pad_token_id=tok.eos_token_id)
            text=tok.decode(y[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            f.write(json.dumps({"id":r["id"],"output":text},ensure_ascii=False)+"\n")
if __name__=="__main__":main()
