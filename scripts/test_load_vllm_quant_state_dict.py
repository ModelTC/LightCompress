#!/usr/bin/env python3
"""
Load the vLLM quant model from save_for_vllm/industrialcoder_rtn_fp8_wikitext/vllm_quant_model
and print state_dict keys (and optionally full state_dict).
"""
import argparse
import os
import sys

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Load vLLM quant model and print state_dict")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="save_for_vllm/industrialcoder_rtn_int_awq_wikitext/",
        help="Path to vllm_quant_model directory",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print all state_dict keys (default: only summary and weight_scale keys)",
    )
    parser.add_argument(
        "--no-load-weights",
        action="store_true",
        help="Only load config and print expected keys from index (no full model load)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Load model on CPU (default: load on GPU)",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Error: not a directory: {model_dir}")
        sys.exit(1)

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"Error: config.json not found in {model_dir}")
        sys.exit(1)

    print(f"Loading from: {model_dir}\n")

    if args.no_load_weights:
        # Only inspect index / config without loading full model
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        print("Config model_type:", getattr(config, "model_type", "?"))
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            import json
            with open(index_path) as f:
                index = json.load(f)
            meta = index.get("metadata", {})
            weight_map = index.get("weight_map", {})
            print(f"Total tensors in index: {len(weight_map)}")
            print("\nFirst 20 keys in weight_map:")
            for i, k in enumerate(sorted(weight_map.keys())):
                if i >= 20:
                    print("  ...")
                    break
                print(f"  {k}")
            weight_scale_keys = [k for k in weight_map if "weight_scale" in k]
            print(f"\nKeys containing 'weight_scale': {len(weight_scale_keys)}")
            for k in sorted(weight_scale_keys)[:30]:
                print(f"  {k}")
            if len(weight_scale_keys) > 30:
                print(f"  ... and {len(weight_scale_keys) - 30} more")
        return

    device_map = "cpu" if args.cpu else "cuda:0"
    print(f"Loading full model on {device_map} (may take a while and use significant memory)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    print(f"Total keys in state_dict: {len(keys)}\n")

    if args.list_keys:
        print("All state_dict keys:")
        for k in sorted(keys):
            t = state_dict[k]
            print(f"  {k}  shape={tuple(t.shape)}  dtype={t.dtype}")
    else:
        print("Sample keys (first 30):")
        for k in sorted(keys)[:30]:
            t = state_dict[k]
            print(f"  {k}  shape={tuple(t.shape)}  dtype={t.dtype}")
        if len(keys) > 30:
            print("  ...")

    weight_scale_keys = [k for k in keys if "weight_scale" in k]
    print(f"\nKeys containing 'weight_scale': {len(weight_scale_keys)}")
    for k in sorted(weight_scale_keys):
        t = state_dict[k]
        print(f"  {k}  shape={tuple(t.shape)}  dtype={t.dtype}")

    # Check for the key that was missing in the error
    target = "layers.0.mlp.down_proj.weight_scale"
    if target in keys:
        print(f"\nKey '{target}' present in state_dict.")
    else:
        print(f"\nKey '{target}' NOT in state_dict.")
        # Show similar keys
        similar = [k for k in keys if "down_proj" in k and "weight_scale" in k]
        if similar:
            print("Similar keys (down_proj + weight_scale):")
            for k in sorted(similar)[:10]:
                print(f"  {k}")


if __name__ == "__main__":
    main()
