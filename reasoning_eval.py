"""
reasoning_eval.py -- per-benchmark general-reasoning scores for the policy model.

Reports the four reasoning benchmarks SEPARATELY (each with its own task metric), so a
GRPO run that sharpens the red/blue behaviour can be checked for reasoning regressions:

    IFEval             prompt-level strict accuracy   (instruction following)
    GPQA               accuracy (diamond split)       (graduate-level QA)
    LiveCodeBench-v5   pass@1                          (code generation)
    AIME 2025          accuracy                        (competition math)

This module does NOT re-implement the benchmarks. It reads a results JSON (produced by an
eval harness such as lm-evaluation-harness, written to output/reasoning/reasoning_results.json)
and prints a per-benchmark table to the console. If no results file exists it prints the
table with placeholders plus a one-line hint on how to produce it, so `main.py` always shows
the four benchmarks separately.

Results JSON schema (flexible; unknown benchmarks are shown too):
    {
      "model": "Qwen/Qwen3-8B",
      "benchmarks": {
        "IFEval":           {"score": 0.78, "metric": "prompt-level strict acc", "n": 541},
        "GPQA":             {"score": 0.41, "metric": "accuracy", "n": 198},
        "LiveCodeBench-v5": {"score": 0.33, "metric": "pass@1", "n": 268},
        "AIME 2025":        {"score": 0.26, "metric": "accuracy", "n": 30}
      }
    }

Usage:
    python reasoning_eval.py                         # read + print default results file
    python reasoning_eval.py --results path.json     # read a specific file
    python reasoning_eval.py --run --endpoint http://localhost:8000 --model Qwen/Qwen3-8B
                                                     # best-effort: run via lm-evaluation-harness
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

# The four benchmarks, each reported with its own primary metric (NOT merged into one score).
BENCHMARKS: list[tuple[str, str]] = [
    ("IFEval", "prompt-level strict acc"),
    ("GPQA", "accuracy"),
    ("LiveCodeBench-v5", "pass@1"),
    ("AIME 2025", "accuracy"),
]

DEFAULT_RESULTS = os.environ.get(
    "REASONING_RESULTS", os.path.join("output", "reasoning", "reasoning_results.json"))

# lm-evaluation-harness task ids (best-effort; override via env if your install differs).
_LM_EVAL_TASKS = {
    "IFEval": os.environ.get("LMEVAL_IFEVAL_TASK", "ifeval"),
    "GPQA": os.environ.get("LMEVAL_GPQA_TASK", "gpqa_diamond_zeroshot"),
    "LiveCodeBench-v5": os.environ.get("LMEVAL_LCB_TASK", "livecodebench"),
    "AIME 2025": os.environ.get("LMEVAL_AIME_TASK", "aime2025"),
}


def load_results(path: str = DEFAULT_RESULTS) -> Optional[dict]:
    """Read a results JSON, or None if it is missing/unreadable."""
    try:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # pragma: no cover
        print(f"[reasoning] could not read {path}: {e}")
    return None


def _fmt(score) -> str:
    if score is None:
        return "   –  "
    try:
        return f"{float(score) * 100:6.2f}%" if float(score) <= 1 else f"{float(score):6.2f}"
    except Exception:
        return str(score)


def print_reasoning_scores(results: Optional[dict] = None, path: str = DEFAULT_RESULTS) -> dict:
    """Print the four reasoning benchmarks SEPARATELY to the console. Returns the scores
    dict actually shown (empty values when a benchmark has no result yet)."""
    if results is None:
        results = load_results(path)
    bench = (results or {}).get("benchmarks", {}) if isinstance(results, dict) else {}
    model = (results or {}).get("model", os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B"))

    print("\n" + "=" * 70)
    print("GENERAL REASONING BENCHMARKS (per-benchmark, model = "
          f"{model})")
    print("=" * 70)
    print(f"  {'benchmark':<18} {'metric':<26} {'score':>8}  {'n':>5}")
    print("  " + "-" * 60)
    shown = {}
    # show the four canonical benchmarks in order, then any extras present in the file
    names = [b for b, _ in BENCHMARKS] + [b for b in bench if b not in dict(BENCHMARKS)]
    for name in names:
        rec = bench.get(name, {}) if isinstance(bench, dict) else {}
        default_metric = dict(BENCHMARKS).get(name, rec.get("metric", "score"))
        metric = rec.get("metric", default_metric)
        score = rec.get("score")
        n = rec.get("n", "")
        shown[name] = score
        print(f"  {name:<18} {str(metric):<26} {_fmt(score):>8}  {str(n):>5}")
    if not any(v is not None for v in shown.values()):
        print("  " + "-" * 60)
        print(f"  (no results yet — produce {path} with an eval harness, or run:")
        print("     python reasoning_eval.py --run --endpoint http://localhost:8000 "
              "--model Qwen/Qwen3-8B)")
    print("=" * 70)
    return shown


def run_lm_eval(endpoint: str, model: str, out_path: str = DEFAULT_RESULTS,
                limit: Optional[int] = None) -> Optional[dict]:
    """Best-effort: run the four benchmarks via lm-evaluation-harness against an
    OpenAI-compatible endpoint (e.g. the vLLM server at :8000), parse its output into our
    schema and write `out_path`. Requires `lm_eval` installed and the tasks available;
    on any failure it returns None and the caller falls back to reading a results file."""
    tasks = ",".join(_LM_EVAL_TASKS.values())
    base = endpoint.rstrip("/")
    model_args = f"model={model},base_url={base}/v1/completions,num_concurrent=4,max_retries=2"
    tmp = Path(out_path).with_suffix(".lmeval.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["lm_eval", "--model", "local-completions", "--model_args", model_args,
           "--tasks", tasks, "--output_path", str(tmp), "--log_samples"]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[reasoning] running lm-eval: tasks={tasks} endpoint={base}")
    try:
        subprocess.run(cmd, check=True, timeout=float(os.environ.get("LMEVAL_TIMEOUT", "7200")))
    except FileNotFoundError:
        print("[reasoning] lm_eval not installed (pip install lm-eval); skipping --run.")
        return None
    except Exception as e:  # pragma: no cover
        print(f"[reasoning] lm-eval run failed: {e}")
        return None
    # lm-eval writes <tmp>/results*.json; collect the 'results' metrics into our schema.
    try:
        res_files = sorted(Path(tmp).rglob("results*.json")) if Path(tmp).is_dir() else [tmp]
        raw = {}
        for rf in res_files:
            raw.update(json.loads(rf.read_text(encoding="utf-8")).get("results", {}))
        inv = {v: k for k, v in _LM_EVAL_TASKS.items()}
        benchmarks = {}
        for task_id, metrics in raw.items():
            name = inv.get(task_id, task_id)
            # pick the first accuracy-like metric lm-eval reports
            score = next((v for k, v in metrics.items()
                          if isinstance(v, (int, float)) and any(
                              m in k for m in ("acc", "pass", "exact_match", "score"))), None)
            benchmarks[name] = {"score": score, "metric": dict(BENCHMARKS).get(name, "score")}
        results = {"model": model, "benchmarks": benchmarks}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[reasoning] wrote {out_path}")
        return results
    except Exception as e:  # pragma: no cover
        print(f"[reasoning] could not parse lm-eval output: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Per-benchmark general-reasoning scores")
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="results JSON to read")
    ap.add_argument("--run", action="store_true",
                    help="run the benchmarks via lm-evaluation-harness before printing")
    ap.add_argument("--endpoint", default=os.environ.get("QWEN_ENDPOINT", "http://localhost:8000"))
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--limit", type=int, default=None, help="limit examples per task (debug)")
    args = ap.parse_args()
    results = None
    if args.run:
        results = run_lm_eval(args.endpoint, args.model, args.results, limit=args.limit)
    print_reasoning_scores(results, path=args.results)


if __name__ == "__main__":
    main()
