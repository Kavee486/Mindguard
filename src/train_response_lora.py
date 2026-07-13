"""
Stage 9 - LoRA fine-tune an instruct LLM to generate empathetic, non-
diagnostic mental-health support responses.

This is the heaviest stage. Check your GPU's VRAM first (`nvidia-smi`):
  - 16GB+ VRAM: use --base_model mistralai/Mistral-7B-Instruct-v0.2
  - 8-12GB VRAM: use --base_model microsoft/Phi-3-mini-4k-instruct (default)
  - <8GB VRAM or no GPU: run this stage on Google Colab / Kaggle instead.

Run:
    python src/train_response_lora.py --epochs 3

After training, merge + export to GGUF and serve through Ollama (see
README.md "Serving the response model" section) so the FastAPI backend can
call it like any other local model.

Requires:
    pip install torch transformers datasets peft bitsandbytes accelerate
"""
import os
import sys
import argparse

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, MODELS_DIR, RESPONSE_BASE_MODEL_SMALL, RANDOM_SEED

SYSTEM_PROMPT = (
    "You are a warm, supportive listener. Respond to the user's message with empathy "
    "and validation. Offer gentle, practical coping suggestions where relevant. Never "
    "diagnose a condition. Always encourage seeking help from a qualified professional "
    "when appropriate."
)


def format_example(context: str, response: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=RESPONSE_BASE_MODEL_SMALL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8, help="Effective batch size = batch_size * grad_accum")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. 4-bit QLoRA fine-tuning requires a GPU - running this "
            "on CPU would silently attempt full-precision training of a multi-billion "
            "parameter model and hang or OOM. Use Google Colab / Kaggle instead (see the "
            "module docstring at the top of this file)."
        )

    torch.manual_seed(RANDOM_SEED)

    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "counseling.csv")).fillna("")
    train_df = df.sample(frac=0.9, random_state=RANDOM_SEED)
    eval_df = df.drop(train_df.index)
    print(f"Train: {len(train_df)}  Eval: {len(eval_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def build_dataset(split_df):
        texts = [
            format_example(row["context"], row["response"], tokenizer)
            for _, row in split_df.iterrows()
        ]
        ds = Dataset.from_dict({"text": texts})
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = build_dataset(train_df)
    eval_ds = build_dataset(eval_df)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = os.path.join(MODELS_DIR, "response_lora")
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")
    print("Next: merge the adapter and convert to GGUF - see README.md 'Serving the response model'.")


if __name__ == "__main__":
    main()
