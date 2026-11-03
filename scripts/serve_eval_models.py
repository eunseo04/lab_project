"""Serve the base model and an existing GRPO LoRA under separate model IDs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ad_eval.models import checkpoint_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--grpo-checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    info = checkpoint_info(args.grpo_checkpoint)
    if args.base_model == args.trained_model:
        parser.error("base and trained model IDs must differ")
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.base_model, "--served-model-name", args.base_model,
            "--enable-lora", "--lora-modules", f"{args.trained_model}={info['path']}",
            "--max-loras", "2", "--max-cpu-loras", "4",
            "--host", args.host, "--port", str(args.port),
            "--max-model-len", str(args.max_model_len), "--max-lora-rank", str(args.max_lora_rank),
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    if args.dry_run:
        import shlex
        print(shlex.join(argv))
        return
    # Keep vLLM in this foreground process; no unrelated running server is stopped.
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
