"""Download public, revision-pinned model weights, with Hugging Face resume support."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"policy": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
          "encoder": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")}
lock = {}
for label, (name, revision) in MODELS.items():
    info = HfApi(token=False).model_info(name, revision=revision)
    path = snapshot_download(name, revision=info.sha, token=False,
        local_dir=str(ROOT / "models" / label),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "1_Pooling/*", "2_Normalize/*"],
        ignore_patterns=["onnx/*", "openvino/*"], max_workers=3)
    files = {}
    for p in sorted(Path(path).rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            digest = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files[str(p.relative_to(path))] = {"bytes": p.stat().st_size, "sha256": digest.hexdigest()}
    lock[label] = {"model": name, "revision": info.sha, "path": str(Path(path).resolve().relative_to(ROOT)), "files": files}
    (ROOT / "research_models.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(label, name, info.sha, "downloaded and hashed", flush=True)
