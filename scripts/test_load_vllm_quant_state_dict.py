#!/usr/bin/env python3
"""
Load the vLLM quant model from save_for_vllm/.../vllm_quant_model and print state_dict keys.
Supports:
  - Full model via from_pretrained (HF)
  - State_dict only via torch.load / safetensors
  - Mimic vLLM: load with vLLM's LLM() like vLLM does for FP8 (--vllm)
"""
import argparse
import json
import os
import sys

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_state_dict_only(model_dir, device="cpu"):
    """Load state_dict from disk using safetensors or torch.load (no model instantiation)."""
    model_dir = os.path.abspath(model_dir)
    state_dict = {}

    # 1) Safetensors: sharded (model.safetensors.index.json) or single (model.safetensors)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_paths = sorted(set(weight_map.values()))
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise RuntimeError("Safetensors format detected but 'safetensors' not installed. pip install safetensors")
        for shard_name in shard_paths:
            shard_path = os.path.join(model_dir, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            tensors = load_file(shard_path, device=device)
            state_dict.update(tensors)
        return state_dict

    single_safetensors = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single_safetensors):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise RuntimeError("safetensors not installed. pip install safetensors")
        return dict(load_file(single_safetensors, device=device))

    # 2) PyTorch .bin / .pt: torch.load
    def _torch_load(path):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    for name in ("pytorch_model.bin", "model.pt", "pytorch_model.pt"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return _torch_load(path)
    # Sometimes sharded as model-00001-of-00003.bin
    import glob
    bin_files = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
    if bin_files:
        for path in bin_files:
            state_dict.update(_torch_load(path))
        return state_dict

    raise FileNotFoundError(
        f"No state dict found in {model_dir}. "
        "Expected: model.safetensors.index.json + .safetensors, model.safetensors, or pytorch_model.bin / .pt"
    )


def main():
    parser = argparse.ArgumentParser(description="Load vLLM quant model and print state_dict")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="save_for_vllm/industrialcoder_rtn_fp8_wikitext/",
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
    parser.add_argument(
        "--state-dict-only",
        action="store_true",
        help="Load only state_dict via torch.load / safetensors (no full model). Lighter and faster for key inspection.",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Load model with vLLM's LLM() (same as vLLM does for FP8). Requires vLLM installed. Use to verify vLLM compatibility.",
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

    if args.vllm:
        # Mimic vLLM loading FP8 model (same code path vLLM uses)
        try:
            from vllm import LLM
        except ImportError as e:
            print("Error: vLLM is not installed. Install with: pip install vllm")
            sys.exit(1)
        print("Loading with vLLM LLM() (same as vLLM for FP8 / compressed-tensors)...")
        try:
            llm = LLM(
                model=model_dir,
                trust_remote_code=True,
                tensor_parallel_size=1,
            )
            print("OK: vLLM loaded the model successfully.")
            # Optional: print one sample to confirm inference
            out = llm.generate(["Hello"], max_tokens=4)
            print("Sample generate:", out)
        except Exception as e:
            print(f"vLLM load failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return

    if args.no_load_weights:
        # Only inspect index / config without loading full model
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        print("Config model_type:", getattr(config, "model_type", "?"))
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
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

    if args.state_dict_only:
        device = "cpu" if args.cpu else "cuda:0"
        print(f"Loading state_dict only (torch.load / safetensors) on {device}...")
        state_dict = load_state_dict_only(model_dir, device=device)
    else:
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
    target = "model.layers.0.mlp.down_proj.weight_scale"
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
