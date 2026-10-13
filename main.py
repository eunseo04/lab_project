"""
main.py -- single entry point for the financial-security closed-loop workflow.

Runs the full pipeline end to end:

  A. Wazuh coverage check  -> can Wazuh detect every attack_dataset.json technique?
                              (coverage.py; writes output/coverage/)
  B. Red/Blue closed loop  -> MCTS search + OpenAI idea generation + Target exec +
                              Wazuh detection + GRPO export + LLM improve
                              (pipeline.py; writes output/pipeline/)
  C. GRPO training hint     -> exact command to train the LOCAL policy using OpenAI
                              as the reward/eval model.

LLM: set OPENAI_API_KEY in .env (copy from .env.example). Without a key the loop
still runs with rule-based fallbacks. See llm.py.

Examples:
  python main.py                        # dry-run, OpenAI if key present
  python main.py --iterations 5
  python main.py --live --ctf-name cowsay   # real Docker target + Wazuh
  python main.py --skip-coverage
"""
from __future__ import annotations

import argparse
import os

from llm import get_llm
import coverage as cov
import pipeline


def step_coverage(profile: str, out_dir: str = "output/coverage") -> dict:
    print("\n" + "=" * 70)
    print("A. ATT&CK x Wazuh COVERAGE CHECK")
    print("=" * 70)
    try:
        dataset = cov.load_dataset()
        wazuh_ids = cov.load_wazuh_ids()
    except FileNotFoundError as e:
        print(f"[coverage] skipped: {e}")
        return {}
    report = cov.coverage_report(dataset, wazuh_ids, profile=profile)
    g, t = report["global"], report["target"]
    print(f"  Global : {g['wazuh_mapped']}/{g['total_techniques']} techniques "
          f"have a Wazuh rule ID ({g['coverage_pct']}%).")
    print(f"  Target ({profile}): {t['covered']}/{t['selected_techniques']} "
          f"({t['coverage_pct']}%); {t['uncovered']} need custom rules.")
    verdict = "YES" if t["uncovered"] == 0 else "NO"
    print(f"\n  >> Can Wazuh detect ALL Target techniques by rule-ID mapping? {verdict}")
    if verdict == "NO":
        print("     (ID mapping != detection. Add custom rules for the uncovered set and")
        print("      verify the required logs actually reach Wazuh before claiming detection.)")
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, f"coverage_{profile}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"needs_custom_rules_{profile}.json"), "w", encoding="utf-8") as f:
        json.dump(report["needs_custom_rules"], f, ensure_ascii=False, indent=2)
    print(f"  artifacts -> {out_dir}/")
    return report


def step_loop(args) -> None:
    print("\n" + "=" * 70)
    print("B. RED/BLUE CLOSED LOOP (MCTS + Qwen2.5-0.5B + Target + Wazuh + GRPO)")
    print("=" * 70)
    pipeline.run_closed_loop(
        iterations=args.iterations,
        ctf_name=args.ctf_name,
        dry_run=args.dry_run,
        out_dir=args.out_dir,
        seed=args.seed,
        profile=args.profile,
        use_llm=args.use_llm,
        attacker_container=args.attacker_container,
        fresh=args.fresh,
    )


def step_train_hint(args) -> None:
    model = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    reward_url = os.environ.get("REWARD_MODEL_URL", "http://localhost:8000")
    reward_name = os.environ.get("REWARD_MODEL_NAME", model)
    print("\n" + "=" * 70)
    print("C. GRPO TRAINING (local Qwen2.5-0.5B-Instruct policy)")
    print("=" * 70)
    print("  Policy, reward, and eval all use Qwen2.5-0.5B-Instruct locally.")
    print("  Start a vLLM server of the model for the reward/eval endpoint, e.g.:")
    print(f"    python -m vllm.entrypoints.openai.api_server --model {model} --port 8000\n")
    print("  Then run GRPO:\n")
    print(f"    python train.py --mode red \\")
    print(f"      --model_path {model} \\")
    print(f"      --traindataset_path {os.path.join(args.out_dir, 'grpo_red_iter1.jsonl')} \\")
    print(f"      --qa_model_url {reward_url} --qa_model_name {reward_name} \\")
    print(f"      --eval_model_url {reward_url} --eval_model_name {reward_name} \\")
    print(f"      --qa_model_weights 1.0 --num_train_epochs 1")
    print("\n  (A GPU is recommended; 0.5B is small enough to fit modest hardware.)")


def parse_args():
    ap = argparse.ArgumentParser(description="Financial-security closed-loop workflow")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--profile", default=os.environ.get("TARGET_PROFILE", "financial_linux_web"))
    ap.add_argument("--ctf-name", default=os.environ.get("CTF_NAME", "cowsay"))
    ap.add_argument("--attacker-container", default=os.environ.get("ATTACKER_CONTAINER"),
                    help="Container to launch real attack tools from (--live)")
    ap.add_argument("--out-dir", default="output/pipeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live", dest="dry_run", action="store_false",
                    help="Use the real Docker target + Wazuh (requires daemon/SIEM)")
    ap.add_argument("--no-llm", dest="use_llm", action="store_false",
                    help="Disable OpenAI even if a key is set")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore saved state.json and start learning from scratch")
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--skip-train-hint", action="store_true")
    ap.set_defaults(dry_run=True, use_llm=True)
    return ap.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("FINANCIAL-SECURITY RED/BLUE SELF-PLAY WORKFLOW")
    print("=" * 70)
    print(f"  profile={args.profile}  mode={'dry-run' if args.dry_run else 'LIVE'}  "
          f"iterations={args.iterations}")
    llm = get_llm() if args.use_llm else None
    if llm and not llm.config.disabled:
        print(f"  LLM=local Qwen:{llm.config.model} (loads on first use)")
    else:
        print("  LLM=disabled (rule-based fallback)")

    if not args.skip_coverage:
        step_coverage(args.profile)
    step_loop(args)

    # cumulative attack/detection metrics table
    traj_path = os.path.join(args.out_dir, "trajectories.json")
    if os.path.exists(traj_path):
        import metrics as _metrics
        _metrics.load_and_report(traj_path, args.out_dir)

    if not args.skip_train_hint:
        step_train_hint(args)

    print("\n" + "=" * 70)
    print("DONE. Artifacts in output/pipeline/ :")
    print("  trajectories.json  -> every red/blue action, detection, mitigation")
    print("  metrics.json/.csv  -> cumulative attack & detection rates")
    print("  llm_improve_*.md   -> per-role strategy adaptation notes")
    print("=" * 70)


if __name__ == "__main__":
    main()
