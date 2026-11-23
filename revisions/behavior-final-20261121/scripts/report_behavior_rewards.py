"""Summarize exact original-reward replays; these are not new evaluations."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research_outputs/behavior-v2"
KINDS=("action_cost","RIA","LWF","ASF")


def main(require_complete=False):
    queue=json.loads((OUT/"queue.json").read_text());groups=defaultdict(list);missing=[]
    for task in queue:
        path=OUT/"reward-decomposition"/(task["id"]+".json")
        if task["status"]!="completed" or not path.exists():missing.append(task["id"]);continue
        audit=json.loads(path.read_text())
        assert audit["passed"] and audit["all_blue_observations_actions_rewards_matched"]
        assert not audit["new_performance_sample"] and not audit["decomposition_supplied_to_actor_or_optimizer"]
        assert (ROOT/audit["source"]).resolve()==Path(task["summary"]).resolve()
        episode=json.loads(Path(task["summary"]).read_text())["episodes"][0]
        assert audit["seed"]==episode["seed"] and audit["joint_return"]==episode["joint_return"]
        assert sum(audit["components"].values())==audit["joint_return"]
        groups[task["stage"]].append(audit)
    metrics=[]
    for stage,items in groups.items():
        values={kind:statistics.mean(x["components"].get(kind,0) for x in items) for kind in KINDS}
        metrics.append({"stage":stage,"n":len(items),**values,
            "joint_return":statistics.mean(x["joint_return"] for x in items),
            "seeds":[x["seed"] for x in items]})
    pairs=[]
    for target,base in (("baseline_context","baseline_raw"),("static","baseline_context"),
                        ("loop1","static"),("loop2","static"),("loop2","baseline_context"),
                        ("transfer_loop2","transfer_baseline")):
        a={x["seed"]:x for x in groups[target]};b={x["seed"]:x for x in groups[base]}
        seeds=sorted(a.keys()&b.keys())
        if seeds:
            deltas={kind:statistics.mean(a[s]["components"].get(kind,0)-b[s]["components"].get(kind,0) for s in seeds) for kind in KINDS}
            pairs.append({"comparison":target+" - "+base,"n":len(seeds),**deltas,"joint_return":sum(deltas.values())})
    result={"status":"completed" if not missing else "partial","original_model_episodes":sum(len(v) for v in groups.values()),
        "new_performance_samples":0,"components":metrics,"paired_component_differences":pairs,"missing":missing,
        "limitation":"Exact reward attribution on recorded trajectories; not a causal ablation of each behavior or of the Skill text."}
    target=OUT/"report";target.mkdir(exist_ok=True)
    (target/"reward_components.json").write_text(json.dumps(result,indent=2)+"\n")
    with (target/"reward_components.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["stage","n",*KINDS,"joint_return","seeds"])
        writer.writeheader();writer.writerows(metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages=("baseline_raw","baseline_context","static","loop1","loop2")
    shown=sorted((m for m in metrics if m["stage"] in stages),key=lambda x:stages.index(x["stage"]))
    if shown:
        fig,ax=plt.subplots(figsize=(9,4.8));bottom=[0.]*len(shown)
        labels={"action_cost":"Team action cost","RIA":"Red Impact","LWF":"Green local work failure","ASF":"Green service access failure"}
        for kind in KINDS:
            values=[-x[kind] for x in shown]
            ax.bar(range(len(shown)),values,bottom=bottom,label=labels[kind]);bottom=[x+y for x,y in zip(bottom,values)]
        ax.set_xticks(range(len(shown)),[f"{x['stage']}\nN={x['n']}" for x in shown])
        ax.set_ylabel("Mean original penalty magnitude (lower is better)")
        ax.set_title("Exact replay of recorded Qwen / Hybrid / FiniteState episodes")
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(target/"reward_components.png",dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ("components","paired_component_differences")},indent=2))
    if require_complete and missing:raise SystemExit("Exact reward replay is incomplete")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--require-complete",action="store_true")
    main(parser.parse_args().require_complete)
