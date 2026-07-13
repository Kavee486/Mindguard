"""
Helper: merge a trained LoRA adapter into its base model and save the
result as a standalone HF model, ready for GGUF conversion (see README.md).

Run:
    python src/merge_lora.py --base_model microsoft/Phi-3-mini-4k-instruct \
        --adapter_dir models/response_lora/adapter \
        --out_dir models/response_lora/merged

Requires: pip install torch transformers peft
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    print(f"Loading base model {args.base_model}...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    merged.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Merged model saved to {args.out_dir}")
    print("Next: convert to GGUF using llama.cpp's convert_hf_to_gguf.py, then "
          "`ollama create mindguard-responder -f Modelfile` (see README.md).")


if __name__ == "__main__":
    main()
