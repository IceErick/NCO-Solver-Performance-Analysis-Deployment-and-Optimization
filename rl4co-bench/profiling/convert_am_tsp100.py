"""Convert Kool et al.'s published TSP-100 model to the pinned RL4CO AM layout."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from run_profile import BENCH, SOURCE

UPSTREAM_COMMIT = "c9abf41ac2f878a55b20dc7e829bc942bb999631"
UPSTREAM_SHA256 = "eefdf5496cc1a15e1b95b0a3025e1bcf703279d22aebc19b5b57c80ca8e3200a"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/wouterkool/attention-learn-to-route/"
    f"{UPSTREAM_COMMIT}/pretrained/tsp_100/epoch-99.pt"
)


def convert_weights(torch, source, target):
    source = dict(source)
    converted = {}

    for name in ("weight", "bias"):
        converted[f"encoder.init_embedding.init_embed.{name}"] = source.pop(f"init_embed.{name}")

    for layer in range(3):
        old = f"embedder.layers.{layer}"
        new = f"encoder.net.layers.{layer}"
        # Original layout is [head, input, head_feature]. Fused Linear expects
        # [qkv * head * head_feature, input] in Q, K, V order.
        projected = [
            source.pop(f"{old}.0.module.W_{name}").transpose(1, 2).reshape(128, 128)
            for name in ("query", "key", "val")
        ]
        converted[f"{new}.0.module.Wqkv.weight"] = torch.cat(projected)
        converted[f"{new}.0.module.out_proj.weight"] = (
            source.pop(f"{old}.0.module.W_out").reshape(128, 128).t().contiguous()
        )
        for name in ("Wqkv", "out_proj"):
            key = f"{new}.0.module.{name}.bias"
            converted[key] = torch.zeros_like(target[key])

        for old_index, new_index in ((0, 0), (2, 1)):
            for name in ("weight", "bias"):
                converted[f"{new}.2.module.lins.{new_index}.{name}"] = source.pop(
                    f"{old}.2.module.{old_index}.{name}"
                )
        for part in (1, 3):
            for name in ("weight", "bias", "running_mean", "running_var"):
                key = f"{part}.normalizer.{name}"
                converted[f"{new}.{key}"] = source.pop(f"{old}.{key}")
            key = f"{new}.{part}.normalizer.num_batches_tracked"
            converted[key] = torch.zeros_like(target[key])  # absent in the original checkpoint, unused in eval

    decoder_keys = {
        "W_placeholder": "context_embedding.W_placeholder",
        "project_step_context.weight": "context_embedding.project_context.weight",
        "project_out.weight": "pointer.project_out.weight",
        "project_node_embeddings.weight": "project_node_embeddings.weight",
        "project_fixed_context.weight": "project_fixed_context.weight",
    }
    for old_key, new_key in decoder_keys.items():
        converted[f"decoder.{new_key}"] = source.pop(old_key)

    if source or set(converted) != set(target):
        raise ValueError("Unexpected source or destination state dict layout")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Published checkpoint: {UPSTREAM_URL}")
    parser.add_argument("input", type=Path, help="published epoch-99.pt")
    parser.add_argument("output", type=Path, help="new tensor checkpoint for RL4CO")
    args = parser.parse_args()

    try:
        payload = args.input.read_bytes()
        if hashlib.sha256(payload).hexdigest() != UPSTREAM_SHA256:
            raise ValueError(f"Input SHA-256 does not match the published checkpoint ({UPSTREAM_SHA256})")
        if args.output.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError("The converted checkpoint must be outside the vendored RL4CO tree")

        sys.path.insert(0, str(BENCH / "instances"))
        from fixed_inputs import verify_rl4co

        pin = verify_rl4co()
        sys.path.insert(0, str(SOURCE))
        import torch
        from rl4co.models.zoo.am import AttentionModelPolicy

        raw = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        policy = AttentionModelPolicy(env_name="tsp")
        converted = convert_weights(torch, raw["model"], policy.state_dict())
        policy.load_state_dict(converted, strict=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as f:
            torch.save(
                {
                    "state_dict": converted,
                    "provenance": {
                        "upstream_url": UPSTREAM_URL,
                        "upstream_sha256": UPSTREAM_SHA256,
                        "trained_on": "TSP-100",
                        "rl4co_git_tree": pin["git_tree"],
                    },
                },
                f,
            )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(args.output)


if __name__ == "__main__":
    main()
