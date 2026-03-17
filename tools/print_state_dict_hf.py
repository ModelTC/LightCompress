import argparse
import json
import os
from collections import defaultdict
from importlib.metadata import version

from huggingface_hub import snapshot_download
from safetensors import safe_open


def _find_index_file(model_dir: str) -> str:
    candidates = [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    for name in candidates:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find an index json in {model_dir}. Tried: {', '.join(candidates)}"
    )


def _iter_safetensors_index(index_path: str):
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    if "weight_map" not in index:
        raise ValueError(f"Index file missing 'weight_map': {index_path}")

    weight_map = index["weight_map"]
    shard_to_keys = defaultdict(list)
    for k, shard_rel in weight_map.items():
        shard_to_keys[shard_rel].append(k)

    for shard_rel, keys in shard_to_keys.items():
        yield shard_rel, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=str,
        default="charles2530/Wan2.2-T2V-A14B-Diffusion-AWQ-INT4",
        help="Hugging Face repo id, e.g. charles2530/Wan2.2-T2V-A14B-Diffusion-AWQ-INT4",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="If provided, read model files from this local directory instead of downloading.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HF revision (branch/tag/commit). Default: main",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force download snapshot (ignored if --local_dir is set).",
    )
    parser.add_argument(
        "--max_keys",
        type=int,
        default=200,
        help="Max number of parameter keys to print (across all shards). Default: 200",
    )
    parser.add_argument(
        "--print_values",
        action="store_true",
        help="Also print tensor repr (VERY large output). Default: off",
    )
    args = parser.parse_args()

    print(f"huggingface-hub : {version('huggingface-hub')}")
    print(f"safetensors : {version('safetensors')}")

    if args.local_dir is not None:
        model_dir = args.local_dir
    else:
        model_dir = snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_files_only=not args.download,
        )

    index_path = _find_index_file(model_dir)
    print(f"model_dir : {model_dir}")
    print(f"index : {index_path}")

    printed = 0
    for shard_rel, keys in _iter_safetensors_index(index_path):
        shard_path = os.path.join(model_dir, shard_rel)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"Shard not found: {shard_path}\n"
                "Tip: re-run with --download to fetch all shards."
            )

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                print(f"{k}  shape={tuple(t.shape)} dtype={t.dtype}")
                if args.print_values:
                    print(t)
                printed += 1
                if args.max_keys is not None and printed >= args.max_keys:
                    print(f"Reached --max_keys={args.max_keys}, stopping.")
                    return


if __name__ == "__main__":
    main()

