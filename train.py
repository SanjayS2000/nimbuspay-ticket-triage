"""T4-safe QLoRA fine-tuning for NimbusPay ticket triage."""
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, set_seed)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(rows, tokenizer, max_length):
    examples, truncated = [], 0
    for row in rows:
        messages = row["messages"]
        # Transformers versions differ here: older releases return a list,
        # whereas newer ones can return BatchEncoding.
        prompt_encoded = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
        prompt_ids = prompt_encoded["input_ids"] if isinstance(prompt_encoded, dict) else prompt_encoded
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        target_ids = tokenizer(messages[2]["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        # Preserve the system/user prefix and always retain the complete assistant target.
        available_prompt = max_length - len(target_ids)
        if available_prompt <= 0:
            raise ValueError("max_length is too short for a target JSON")
        if len(prompt_ids) > available_prompt:
            prompt_ids = prompt_ids[:available_prompt]
            truncated += 1
        ids = prompt_ids + target_ids
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + target_ids})
    print(f"Tokenized {len(examples)} examples; truncated long user prompts: {truncated}")
    return Dataset.from_list(examples)


@dataclass
class CompletionCollator:
    tokenizer: object
    def __call__(self, features):
        longest = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for x in features:
            pad = longest - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="artifacts/train_clean.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--output", default="artifacts/adapter")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.rank > 8:
        raise ValueError("Use rank <= 8: Qwen 3B with all target modules stays safely below the 50M trainable-parameter cap.")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    compute_dtype = torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=compute_dtype) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, torch_dtype=compute_dtype,
                                                 device_map="auto", trust_remote_code=False)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    rows = read_jsonl(args.train)
    dataset = build_dataset(rows, tokenizer, args.max_length)
    train_kwargs = dict(output_dir=args.output + "_checkpoints", num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_steps=9, logging_steps=10, save_strategy="epoch", report_to="none",
        fp16=True, optim="paged_adamw_8bit", gradient_checkpointing=True, group_by_length=True,
        save_total_limit=1, seed=args.seed)
    supported = set(inspect.signature(TrainingArguments).parameters)
    skipped = sorted(set(train_kwargs) - supported)
    if skipped:
        print(f"Installed Transformers does not support these optional settings; skipping: {skipped}")
    train_args = TrainingArguments(**{key: value for key, value in train_kwargs.items() if key in supported})
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=CompletionCollator(tokenizer))
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    with open(Path(args.output) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved standard PEFT adapter to {args.output}")


if __name__ == "__main__":
    main()
