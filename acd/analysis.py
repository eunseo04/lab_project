"""Local plots and the paper's PCA/K-means analysis, using supplied embeddings."""
import csv
import json
from pathlib import Path

import numpy as np
from .config import LEGACY_ENCODER
from .runner import write_json


def plot_results(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    rows = []
    for path in sorted(directory.rglob("summary.json")):
        summary = json.loads(path.read_text())
        if "reward" not in summary:
            continue
        c = summary["config"]
        short_training = any((weight.get("training") or {}).get("validation_only", False)
                             for weight in summary.get("weights", {}).values())
        rows.append({"path": str(path), "team": c["team"], "red": c["red"],
                     "communication": c["communication"],
                     "communication_encoder": c.get("communication_encoder", LEGACY_ENCODER),
                     "kind": summary["result_kind"],
                     "steps": c["steps"], "episodes": c["episodes"],
                     "short_training_checkpoint": short_training,
                     "reward_mean": summary["reward"]["mean"],
                     "reward_stdev": summary["reward"]["stdev"], "seconds": summary["elapsed_seconds"]})
    if not rows:
        raise ValueError("No summary.json experiment results found")
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for kind, encoder in sorted({(r["kind"], r["communication_encoder"]) for r in rows}):
        selected = [r for r in rows if (r["kind"], r["communication_encoder"]) == (kind, encoder)]
        fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.65), 5))
        ax.bar(range(len(selected)), [r["reward_mean"] for r in selected],
               yerr=[r["reward_stdev"] or 0 for r in selected], capsize=3)
        ax.set_xticks(range(len(selected)), [f"{r['team']}\n{r['red']}\n{r['communication']}" for r in selected], rotation=45, ha="right")
        ax.set_ylabel("Cumulative shared team reward")
        titles = {"offline_mock_pipeline_only": "Offline mock pipeline — no LLM results",
                  "replay_pipeline_only": "Replay pipeline — verify original model provenance"}
        title = titles.get(kind, "CC4 simulation evaluation")
        if any(r["short_training_checkpoint"] for r in selected):
            title += " — short-training verification"
        ax.set_title(f"{title} / {encoder}")
        fig.tight_layout()
        suffix = "" if encoder == LEGACY_ENCODER else f"-{encoder}"
        fig.savefig(directory / f"rewards-{kind}{suffix}.png", dpi=160)
        plt.close(fig)
    return rows


def export_reasons(trace, output, agent="blue_agent_4", episode=0):
    rows = []
    for line in Path(trace).read_text().splitlines():
        row = json.loads(line)
        if (row["agent"] == agent and row["episode"] == episode and row["step"] > 0
                and row.get("reason") and not row.get("parse_error")):
            rows.append({k: row[k] for k in ("episode", "step", "agent", "reason", "response")})
    if not rows:
        raise ValueError("No matching reasons after excluding initialization step")
    with Path(output).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def cluster_reasons(input_path, embeddings_path, output, clusters=4):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    vectors = np.load(embeddings_path, allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows) or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite NxD and in the exact exported reason row order")
    if min(vectors.shape) < 3 or not 2 <= clusters < len(rows):
        raise ValueError("Need >=3 dimensions, >=3 rows and 2 <= clusters < rows")
    reduced = PCA(n_components=3, random_state=42).fit_transform(vectors)
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit(reduced)
    labels = model.labels_
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    diagnostics = []
    for k in range(2, min(10, len(rows) - 1) + 1):
        fit = KMeans(n_clusters=k, random_state=42, n_init=10).fit(reduced)
        diagnostics.append({"k": k, "inertia": fit.inertia_,
                            "silhouette": silhouette_score(reduced, fit.labels_) if len(set(fit.labels_)) > 1 else None})
    import hashlib
    result = {"samples": len(rows), "embedding_dimensions": vectors.shape[1], "k": clusters,
              "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
              "embeddings_sha256": hashlib.sha256(Path(embeddings_path).read_bytes()).hexdigest(),
              "paper_embedding_dimensions": 3072, "embedding_model_verified": False,
              "silhouette": silhouette_score(reduced, labels) if len(set(labels)) > 1 else None,
              "diagnostics": diagnostics,
              "rows": [{**row, "cluster": int(label)} for row, label in zip(rows, labels)]}
    write_json(destination / "clusters.json", result)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*reduced.T, c=labels, s=12, cmap="tab10")
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title="Reason embeddings: PCA + K-means")
    fig.tight_layout()
    fig.savefig(destination / "clusters.png", dpi=180)
    plt.close(fig)
    return result


# Embedding API integration is intentionally commented out as requested.
# Replace with your chosen embedding model and preserve exported JSONL row order.
# from openai import OpenAI
# result = OpenAI().embeddings.create(model="text-embedding-3-large",
#     input=[row["reason"] for row in rows])
# np.save("embeddings.npy", np.asarray([r.embedding for r in result.data]))
# Cluster text summarization via a model is also left for a future backend.
