"""Greedy, no-postprocessing inference required by the assessment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def read_jsonl(path):
    with open(path, encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--output", required=True)
    args=ap.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=q,torch_dtype=torch.float16,device_map="auto")
    model=PeftModel.from_pretrained(base,args.adapter); model.eval()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        for row in read_jsonl(args.input):
            prompt=tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=200,do_sample=False,temperature=None,top_p=None,
                    pad_token_id=tokenizer.eos_token_id)
            output=tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            f.write(json.dumps({"id":row["id"],"output":output},ensure_ascii=False)+"\n")

if __name__=="__main__": main()
