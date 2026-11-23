"""Verify pinned local policy/encoder files without any network access."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
lock=json.loads((ROOT/"research_models.lock.json").read_text())
checked=[]
errors=[]
for label,model in lock.items():
    for name,expected in model["files"].items():
        path=ROOT/"models"/label/name
        if not path.is_file():errors.append({"file":str(path),"error":"missing"});continue
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda:stream.read(8*1024*1024),b""):digest.update(data)
        actual=digest.hexdigest()
        if actual!=expected["sha256"] or path.stat().st_size!=expected["bytes"]:
            errors.append({"file":str(path),"error":"content differs from pinned download"})
        checked.append({"file":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":actual})
result={"passed":not errors,"files_checked":len(checked),"errors":errors,"files":checked}
destination=ROOT/"research_outputs/assets_verification.json"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="files"}))
if errors:raise SystemExit(1)
