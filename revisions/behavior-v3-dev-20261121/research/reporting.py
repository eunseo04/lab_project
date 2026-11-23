"""Tables and figures from real completed records only; unavailable cells stay N/A."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from .config import ARMS, REDS
from .adapter import ACTIONS
from .metrics import aggregate_episodes
from .experiment import atomic_json
from . import ROOT


def render_table(path, headers, rows):
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(headers); writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)]
    lines += ["| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |" for row in rows]
    path.with_suffix(".md").write_text("\n".join(lines)+"\n")


def number(value):
    return "N/A" if value is None else f"{value:.3f}"


def report(directory, reasoning=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    destination = directory/"report"; destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    rejected_mock = []
    for path in sorted(directory.rglob("summary.json")):
        if "report" in path.relative_to(directory).parts: continue
        value = json.loads(path.read_text())
        if "config" not in value or "stage" not in value: continue
        if value.get("result_kind") != "real_cage4":
            rejected_mock.append(str(path)); continue
        if value["stage"] != "eval": continue
        value["episodes"] = [e for e in value.get("episodes", [])
                             if e.get("status") == "completed" and e.get("kind") == "real_cage4"]
        if not value["episodes"]: continue
        value["path"] = str(path)
        summaries.append(value)
    groups = defaultdict(list)
    controls = []
    for s in summaries:
        c = s["config"]
        if c["composition"] in ("no_blue", "all_rl"):
            m=aggregate_episodes(s["episodes"])
            controls.append([c["composition"],c["red"],c["steps"],c["scoring"],c["communication"],m["n"],
                number(m["mean_joint_reward"]),number(m["reward_std"]),number(m["execution_time_s"])])
        if c["composition"] not in ("hybrid", "all_llm"): continue
        if c["arm"] in ("E2","E3") and not c["evolution"].get("enabled",True): continue
        if not all(c["evolution"].get(k,True) for k in ("semantic_gradient","ppo_gate","score_maintenance")): continue
        model = c["model"]
        key = (c["composition"], c["profile"], c["scoring"], c["communication"], c["steps"],
               model["model"], model["revision"], model.get("temperature", 1), model.get("max_tokens", 192), c["evolution"]["gate_mode"],
               c["evolution"].get("maintenance_mode", "official_code"), c["skip_busy_policy"],
               c["selector"]["epsilon"], c["selector"]["max_skill_age"])
        groups[key].append(s)
    diagnostics = {"excluded_mock_summaries": rejected_mock, "groups": [], "unexecuted_cells": "N/A"}
    render_table(destination/"control_baselines", ["Composition","Red","Nominal steps","Scoring","Communication","N","Mean Reward","Sample Std","Seconds"], controls)
    # Architecture is a description of actual implemented modules, not performance data.
    fig, ax = plt.subplots(figsize=(13, 5)); ax.axis("off")
    boxes = [("Blue observation\nCAGE4 observable state",.12,.73),
             ("Skill Pool → selector\nE3: observable verification",.37,.73),
             ("Frozen policy LLM\nQwen3-8B by default",.62,.73),
             ("Joint Blue actions\nOriginal CAGE4 reward",.87,.73),
             ("Per-Blue experience\nbefore/action/status/after",.87,.27),
             ("Semantic gradients\nBatch aggregation",.62,.27),
             ("Exact token PPO Gate\nBest-of-N, fail closed",.37,.27),
             ("Skill Pool update\nScores / dedup / capacity",.12,.27)]
    for label,x,y in boxes:
        ax.text(x,y,label,ha="center",va="center",fontsize=10,
            bbox={"boxstyle":"round,pad=.6","fc":"#e9f1fb"},transform=ax.transAxes)
    for i in (0,1,2):
        ax.annotate("",xy=(boxes[i+1][1]-.105,.73),xytext=(boxes[i][1]+.105,.73),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    ax.annotate("",xy=(.87,.36),xytext=(.87,.64),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    for i in (4,5,6):
        ax.annotate("",xy=(boxes[i+1][1]+.105,.27),xytext=(boxes[i][1]-.105,.27),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    ax.plot([.12,.12,.37],[.36,.51,.51],color="#336699",transform=ax.transAxes)
    ax.annotate("",xy=(.37,.64),xytext=(.37,.51),xycoords="axes fraction",arrowprops={"arrowstyle":"->","color":"#336699"})
    ax.text(.25,.54,"training only",color="#336699",ha="center",transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(destination/"figure1_architecture.png", dpi=180); plt.close(fig)
    if not groups:
        (destination/"status.md").write_text("No completed real Qwen comparison episodes. All measured performance cells are N/A. Mock results are excluded.\n")
        reference = json.loads((ROOT/"research_configs/paper_reference.json").read_text())
        render_table(destination/"table1_overall", ["Metric", "KEEP (Paper Reference)", "GPT-4o-mini (Paper Reference)", *ARMS],
            [[label, number(reference["all_rl"][field]), number(reference["all_llm"][field]), *["N/A"]*4]
             for label, field in (("Mean Joint Reward","mean_joint_reward"),("Reward Std.","reward_std"),("Execution Time (s)","execution_time_s"))])
        render_table(destination/"table2_red_strategy", ["Red Strategy", *ARMS], [[red,*["N/A"]*4] for red in REDS])
        render_table(destination/"table3_executed_actions", ["Action", *ARMS], [[action,*["N/A"]*4] for action in ACTIONS])
        for figure, label in ((2,"reward"),(3,"red_strategy"),(4,"actions")):
            fig, ax = plt.subplots(); ax.axis("off"); ax.text(.5,.5,"N/A: no real paired evaluation",ha="center")
            fig.savefig(destination/f"figure{figure}_{label}.png",dpi=150); plt.close(fig)
    elif (destination/"status.md").exists():
        (destination/"status.md").unlink()
    for index, (key, items) in enumerate(groups.items()):
        folder = destination/f"group-{index:02d}-{key[0]}"; folder.mkdir(exist_ok=True)
        by_arm_red = {}
        for item in items:
            case=(item["config"]["arm"],item["config"]["red"])
            if case in by_arm_red: raise ValueError(f"Duplicate run for {case}; report one experiment root at a time")
            by_arm_red[case]=item
        arm_metrics, arm_episodes, available_metrics = {}, {}, {}
        automatic_counts = {a: Counter() for a in ARMS}
        red_rows, counts = [], {a: Counter() for a in ARMS}
        paired_seeds = {}
        available_seed_pairs = {}
        pairwise_rows = []
        for red in REDS:
            sets = [{e["seed"] for e in by_arm_red.get((arm, red), {}).get("episodes", [])} for arm in ARMS]
            paired = set.intersection(*sets) if all(sets) else set()
            paired_seeds[red] = sorted(paired)
            nonempty = [s for s in sets if s]
            displayed = set.intersection(*nonempty) if nonempty else set()
            available_seed_pairs[red] = sorted(displayed)
            row = [red]
            for arm in ARMS:
                episodes = [e for e in by_arm_red.get((arm, red), {}).get("episodes", []) if e["seed"] in displayed]
                row.append(number(aggregate_episodes(episodes)["mean_joint_reward"]))
            red_rows.append(row)
            for left,right in (("E0","E1"),("E1","E2"),("E2","E3"),("E0","E3")):
                a={e["seed"]:e["joint_return"] for e in by_arm_red.get((left,red),{}).get("episodes",[])}
                b={e["seed"]:e["joint_return"] for e in by_arm_red.get((right,red),{}).get("episodes",[])}
                common=sorted(a.keys() & b.keys())
                differences=[b[s]-a[s] for s in common]
                pairwise_rows.append([red,f"{right} - {left}",len(common),
                    number(mean([a[s] for s in common]) if common else None),
                    number(mean([b[s] for s in common]) if common else None),
                    number(mean(differences) if differences else None),
                    number(stdev(differences) if len(differences)>1 else None)])
        complete_red_coverage = all(paired_seeds[r] for r in REDS)
        for arm in ARMS:
            episodes = []
            for red in REDS:
                for episode in by_arm_red.get((arm, red), {}).get("episodes", []):
                    if episode["seed"] in paired_seeds[red]: episodes.append(episode)
            arm_episodes[arm] = episodes
            available_metrics[arm] = aggregate_episodes(episodes)
            arm_metrics[arm] = aggregate_episodes(episodes if complete_red_coverage else [])
            for episode in episodes:
                for line in Path(episode["trace"]).read_text().splitlines():
                    row = json.loads(line)
                    if row.get("policy") != "llm": continue
                    if row.get("automatic_monitor"): automatic_counts[arm]["Monitor"] += 1
                    for action in row.get("executed_actions", []): counts[arm][action["name"]] += 1
        reference = json.loads((ROOT/"research_configs/paper_reference.json").read_text())
        rows = []
        for label, field, old_field in [("Mean Joint Reward", "mean_joint_reward", "mean_joint_reward"),
            ("Reward Std. (sample)", "reward_std", "reward_std"), ("Execution Time (s)", "execution_time_s", "execution_time_s")]:
            rows.append([label, number(reference["all_rl"][old_field]), number(reference["all_llm"][old_field])]
                        + [number(arm_metrics[arm][field]) for arm in ARMS])
        render_table(folder/"table1_overall", ["Metric", "ACD KEEP (Paper Reference)", "ACD GPT-4o-mini (Paper Reference)", *ARMS], rows)
        render_table(folder/"table2_red_strategy", ["Red Strategy", *ARMS], red_rows)
        render_table(folder/"pairwise_available", ["Red","Comparison","Paired N","Left Mean","Right Mean","Mean difference","Sample Std of differences"],pairwise_rows)
        render_table(folder/"table3_executed_actions", ["Action (actual LLM-Blue execution totals)", *ARMS],
            [[action, *[counts[arm][action] if arm_episodes[arm] else "N/A" for arm in ARMS]] for action in ACTIONS])
        render_table(folder/"table3_automatic_actions", ["Automatic action (LLM Blue, separate)", *ARMS],
            [["Monitor", *[automatic_counts[arm]["Monitor"] if arm_episodes[arm] else "N/A" for arm in ARMS]]])
        render_table(folder/"paired_available_policies", ["Arm", "Paired N", "Mean Reward", "Sample Std", "Seconds"],
            [[a, available_metrics[a]["n"], *[number(available_metrics[a][k]) for k in ("mean_joint_reward","reward_std","execution_time_s")]] for a in ARMS])
        completed_rows = []
        for item in items:
            m = aggregate_episodes(item.get("episodes", []))
            completed_rows.append([item["config"]["arm"], item["config"]["red"], m["n"],
                number(m["mean_joint_reward"]), number(m["reward_std"]), number(m["execution_time_s"]), item["status"]])
        render_table(folder/"completed_cases", ["Arm", "Red", "N", "Mean Reward", "Sample Std", "Seconds", "Status"], completed_rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        available = [a for a in ARMS if available_metrics[a]["n"]]
        if available:
            ax.bar(available, [available_metrics[a]["mean_joint_reward"] for a in available],
                yerr=[available_metrics[a]["reward_std"] or 0 for a in available], capsize=4)
        else: ax.text(.5,.5,"N/A: five-Red paired real evaluation is incomplete", ha="center", transform=ax.transAxes)
        scope = ", ".join(red for red in REDS if paired_seeds[red]) or "N/A"
        measured_n=available_metrics[available[0]]["n"] if available else 0
        ax.set_title(f"Mean Joint Reward — paired policies: {scope}\nSample SD error bars; N={measured_n} per arm",fontsize=11); ax.set_ylabel("Original CAGE4 return")
        fig.tight_layout(); fig.savefig(folder/"figure2_reward.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9,4))
        for arm in ARMS:
            x, y = [], []
            for j, red in enumerate(REDS):
                episodes = [e for e in by_arm_red.get((arm,red),{}).get("episodes",[]) if e["seed"] in available_seed_pairs[red]]
                if episodes: x.append(j); y.append(aggregate_episodes(episodes)["mean_joint_reward"])
            if x: ax.plot(x,y,"o-",label=arm)
        ax.set_xticks(range(5), REDS); ax.set_title("Real reward by Red policy — available arms, paired seeds")
        if ax.lines: ax.legend()
        else: ax.text(.5,.5,"N/A: no matched E0–E3 seeds",ha="center",transform=ax.transAxes)
        fig.tight_layout();fig.savefig(folder/"figure3_red_strategy.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4))
        for j,arm in enumerate(ARMS):
            if arm_episodes[arm]: ax.bar(np.arange(len(ACTIONS))+j*.18,[counts[arm][a]/len(arm_episodes[arm]) for a in ACTIONS],.18,label=arm)
        ax.set_xticks(np.arange(len(ACTIONS))+.27,ACTIONS,rotation=35,ha="right")
        ax.set_title("Actual LLM-Blue actions per episode; automatic Monitor separate")
        if ax.patches: ax.legend()
        fig.tight_layout();fig.savefig(folder/"figure4_actions.png",dpi=180);plt.close(fig)
        info = {"comparison_key": list(key), "paired_seeds": paired_seeds,
            "table2_available_arm_paired_seeds":available_seed_pairs,
            "action_red_scope":[red for red in REDS if paired_seeds[red]],
            "overall_requires_all_five_reds": True, "complete_red_coverage": complete_red_coverage,
            "metrics": arm_metrics, "paired_available_policy_metrics": available_metrics,
            "aggregation": "Sample statistics over pooled paired episode returns; overall requires all five policies.",
            "training_budgets": {s["config"]["arm"]+":"+s["config"]["red"]: s.get("training_provenance") for s in items},
            "paper_reference_aggregation": reference["aggregation_uncertainty"]}
        atomic_json(folder/"comparison.json",info);diagnostics["groups"].append(info)
    # Training/skill plots are distinct from held-out evaluation metrics.
    histories = []
    for path in sorted(directory.rglob("summary.json")):
        s=json.loads(path.read_text())
        if s.get("stage")=="train" and s.get("result_kind")=="real_cage4":
            s["source_path"]=str(path);histories.append(s)
    diagnostics["training_groups"]=training_figures(histories,destination)
    ablations=[]
    for path in directory.rglob("summary.json"):
        s=json.loads(path.read_text())
        if s.get("stage")=="eval" and s.get("result_kind")=="real_cage4":
            e=s["config"]["evolution"]
            labels=[name for flag,name in [("semantic_gradient","Without Semantic Gradient"),("ppo_gate","Without PPO Gate"),("score_maintenance","Without Score Maintenance")] if not e.get(flag,True)]
            if not e.get("enabled",True): labels.append("Without Skill Evolution")
            if len(labels)>1: labels=["Combined: "+" + ".join(labels)]
            if not labels: labels=["Proposed" if s["config"]["arm"]=="E3" else "Full Skill-Pro"] if s["config"]["arm"] in ("E2","E3") else []
            for label in labels:
                m=aggregate_episodes(s.get("episodes",[]));ablations.append([label,s["config"]["composition"],s["config"]["red"],s["config"]["steps"],e["gate_mode"],e.get("maintenance_mode","official_code"),m["n"],number(m["mean_joint_reward"]),number(m["reward_std"]),number(m["execution_time_s"])])
    known={r[0] for r in ablations}
    for label in ["Without Semantic Gradient","Without PPO Gate","Without Score Maintenance","Full Skill-Pro","Proposed"]:
        if label not in known: ablations.append([label,"N/A","N/A","N/A","N/A","N/A",0,"N/A","N/A","N/A"])
    render_table(destination/"table4_ablation",["Configuration","Composition","Red","Nominal steps","Gate mode","Maintenance mode","N","Mean Reward","Sample Std","Time"],ablations)
    reasoning_status=destination/"figure7_reasoning_status.json"
    previous=json.loads(reasoning_status.read_text()) if reasoning_status.exists() else {}
    if reasoning or previous.get("status")!="completed":
        atomic_json(reasoning_status,{"status":"not_run","reason":"Requires actual reason records and explicit embedding analysis; paper embedding model is not silently replaced."})
        for name in ("figure7_reasoning.png","reasoning_clusters.csv","reasoning_clusters.md"):
            if (destination/name).exists(): (destination/name).unlink()
    if reasoning: reasoning_analysis(summaries,destination)
    atomic_json(destination/"report_manifest.json",diagnostics)
    print("Reports saved:",destination)
    return diagnostics


def training_figures(histories,destination):
    """Never overlay short/full horizons or different reward/model protocols."""
    import hashlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    groups=defaultdict(list)
    for s in histories:
        c=s["config"];e=c["evolution"]
        scope={k:c[k] for k in ("composition","red","steps","profile","scoring","communication","skip_busy_policy","selector")}
        scope["model"]={k:c["model"].get(k) for k in ("model","revision","temperature","max_tokens")}
        scope["algorithm"]={k:e.get(k,True) for k in ("enabled","semantic_gradient","ppo_gate","score_maintenance")}
        scope["algorithm"].update(gate_mode=e["gate_mode"],maintenance_mode=e.get("maintenance_mode","official_code"))
        scope["validation_seeds"]=list(range(c["validation_seed"],c["validation_seed"]+c["validation_episodes"]))
        groups[json.dumps(scope,sort_keys=True)].append(s)
    if not groups: groups["null"]=[]
    if len(groups)>1:
        for name in ("figure5_evolution_validation.png","figure6_skill_pool.png"):
            if (destination/name).exists(): (destination/name).unlink()
    index=[]
    for key,items in groups.items():
        scope=json.loads(key)
        folder=destination
        if len(groups)>1:
            tag=f"training-{scope['steps']}-{scope['composition']}-{scope['red']}-"+hashlib.sha256(key.encode()).hexdigest()[:8]
            folder=destination/tag;folder.mkdir(exist_ok=True)
        title="N/A: no real training" if scope is None else f"{scope['composition']} / {scope['red']} / {scope['steps']} steps / {scope['algorithm']['gate_mode']}"
        arm_counts=Counter(s["config"]["arm"] for s in items)
        def label(s):
            arm=s["config"]["arm"]
            return arm if arm_counts[arm]==1 else arm+" / "+Path(s["source_path"]).parent.parent.parent.name
        fig,ax=plt.subplots(figsize=(9,4))
        for s in items:
            h=s.get("history",[])
            if h: ax.plot([r["iteration"] for r in h],[r.get("validation_reward_mean",np.nan) for r in h],"o-",label=label(s))
        ax.set_title("Held-out validation — "+title,fontsize=10)
        ax.set_xlabel("Skill evolution iteration");ax.set_ylabel("Original CAGE4 joint return")
        iterations=sorted({r["iteration"] for s in items for r in s.get("history",[])})
        if iterations:ax.set_xticks(iterations)
        if ax.lines: ax.legend(fontsize=8)
        else: ax.text(.5,.5,"N/A: no real training/validation records",ha="center",transform=ax.transAxes)
        fig.tight_layout();fig.savefig(folder/"figure5_evolution_validation.png",dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for item_index,s in enumerate(items):
            h=s.get("history",[])
            if not h: continue
            name=label(s);color=f"C{item_index%10}"
            axes[0].plot([r["iteration"] for r in h],[len((r.get("pool_snapshot") or {}).get("skills",[])) for r in h],"o-",label=name,color=color)
            first=True
            for r in h:
                for skill in (r.get("pool_snapshot") or {}).get("skills",[]):
                    axes[1].scatter(r["iteration"],skill["avg_gain"],s=10,color=color,label=name if first else None);first=False
            axes[2].plot([r["iteration"] for r in h],[sum(c.get("accepted",False) for e in r.get("evolution",[]) for c in e.get("candidates",[])) for r in h],"o-",label=name+" accepted",color=color)
            axes[2].plot([r["iteration"] for r in h],[len(r.get("maintenance",[])) for r in h],"x--",label=name+" removed",color=color)
        axes[0].set_title("Skill-pool size");axes[1].set_title("Actual skill scores");axes[2].set_title("Accepted / removed Skills")
        if axes[0].lines:axes[0].legend(fontsize=8)
        if axes[1].collections:axes[1].legend(fontsize=8)
        if axes[2].lines:axes[2].legend(fontsize=8)
        for axis in axes:
            axis.set_xlabel("Iteration")
            if iterations:axis.set_xticks(iterations)
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True));axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle(title,fontsize=10)
        fig.tight_layout();fig.savefig(folder/"figure6_skill_pool.png",dpi=180);plt.close(fig)
        info={"scope":scope,"summaries":[s["source_path"] for s in items],"directory":str(folder)}
        atomic_json(folder/"training_scope.json",info);index.append(info)
    return index


def reasoning_analysis(summaries,destination):
    from .client import ModelClient
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    candidates=[s for s in summaries if s.get("episodes") and s["config"]["composition"] in ("hybrid","all_llm")]
    if not candidates: return
    source=candidates[0]; episode=source["episodes"][0]
    rows=[json.loads(line) for line in Path(episode["trace"]).read_text().splitlines()]
    rows=[r for r in rows if r["agent"]=="blue_agent_4" and r["step"]>0 and r.get("reason") and not r.get("parse_error") and r.get("policy_invoked")]
    if len(rows)<5: return
    client=ModelClient(source["config"]["model"])
    vectors=np.asarray(client.embed_many([r["reason"] for r in rows]))
    if not np.isfinite(vectors).all(): raise ValueError("Nonfinite reasoning embeddings")
    reduced=PCA(n_components=3,random_state=42).fit_transform(vectors)
    clustering=KMeans(n_clusters=4,random_state=42,n_init=10)
    labels=clustering.fit_predict(reduced)
    result={"status":"completed","embedding_model":"sentence-transformers/all-MiniLM-L6-v2",
        "difference_from_paper":"Local 384-dimensional embeddings; NOT text-embedding-3-large/3072. No model-generated cluster summaries.",
        "samples":len(rows),"source":episode["trace"],"clusters":4,
        "rows":[{"step":r["step"],"reason":r["reason"],"requested_action":r["requested_action"],"cluster":int(k)} for r,k in zip(rows,labels)]}
    atomic_json(destination/"figure7_reasoning_status.json",result)
    cluster_rows=[]
    for k in range(4):
        selected=np.flatnonzero(labels==k)
        if not len(selected): continue
        closest=min(selected,key=lambda i:np.linalg.norm(reduced[i]-clustering.cluster_centers_[k]))
        actions=Counter(rows[i]["requested_action"].split()[0] for i in selected)
        cluster_rows.append([k,len(selected),json.dumps(dict(actions)),rows[closest]["reason"]])
    render_table(destination/"reasoning_clusters",["Cluster","Count","Observed action counts","Representative actual reason"],cluster_rows)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection="3d")
    scatter=ax.scatter(*reduced.T,c=labels,s=15,cmap="tab10")
    ax.set_xlabel("PC1");ax.set_ylabel("PC2");ax.set_zlabel("PC3")
    ax.legend(*scatter.legend_elements(),title="Reason cluster",loc="upper right")
    ax.set_title("Actual "+source["config"]["model"]["model"].split("/")[-1]+" reasons\nLocal MiniLM embedding (changed from paper)",fontsize=11)
    fig.tight_layout();fig.savefig(destination/"figure7_reasoning.png",dpi=180);plt.close(fig)
