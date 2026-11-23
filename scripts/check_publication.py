"""Audit the staged Git blobs; print locations, never secret values."""
import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
LIMIT=10*1024*1024
PRIVATE={'.bash_history','.lesshst','.env','.aws','.ssh','.codex','.idea','.vscode','.DS_Store'}
RUNTIME={'.venv','.model-venv','models','third_party','runtime','checkpoints','__pycache__','.pytest_cache'}
BINARY={'.safetensors','.pt','.pth','.ckpt','.bin','.npy','.npz','.whl','.sqlite','.db'}
ALLOWED={'acd','research','revisions','research_configs','configs','scripts','tests','research_tests',
    'docs','reports','references','artifacts','research_outputs','analysis_outputs','verification',
    '.gitignore','.gitattributes','README.md','README.workspace.md','LICENSE','NOTICE.md','pytest.ini',
    'requirements.txt','requirements.lock.txt','requirements-model.txt','requirements-model.lock.txt',
    'sources.lock.json','research_sources.lock.json','research_models.lock.json','setup.sh','run.sh',
    'research.sh','research_v3.sh','research_v4.sh','research_v5.sh','research_v5_reusable.sh'}
SECRETS={
    'private_key':re.compile(rb'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----'),
    'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{50,255})\b'),
    'model_api_token':re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b'),
    'huggingface_token':re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b'),
    'aws_access_key':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
}


def blob_issues(name,mode,data,root=ROOT):
    path=Path(name);issues=[]
    if path.parts[0] not in ALLOWED:issues.append('outside_research_scope')
    if path.is_absolute() or '..' in path.parts:issues.append('path_outside_repository')
    if PRIVATE.intersection(path.parts):issues.append('private_file_or_directory')
    if mode=='120000':
        target=data.decode('utf-8')
        resolved=Path(os.path.abspath(root/path.parent/target))
        if Path(target).is_absolute() or not resolved.is_relative_to(root):issues.append('external_symlink')
    else:
        if RUNTIME.intersection(path.parts):issues.append('runtime_or_downloaded_asset')
        if path.suffix.lower() in BINARY:issues.append('model_or_binary_artifact')
        if path.suffix in {'.jsonl','.log','.pid'} or path.name=='.lock':issues.append('raw_or_transient_output')
        if len(data)>LIMIT:issues.append('file_over_10MiB')
    for rule,pattern in SECRETS.items():
        if pattern.search(data):issues.append(rule)
    return issues


def audit():
    top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip()
    if Path(top)!=ROOT:raise RuntimeError('Run inside the isolated lab_project checkout')
    entries=[]
    raw=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT)
    for item in raw.split(b'\0'):
        if not item:continue
        metadata,name=item.split(b'\t',1);mode,oid,stage=metadata.decode().split()
        if stage!='0':raise RuntimeError('Unmerged index entry')
        entries.append((name.decode(),mode,oid))
    if not entries:raise RuntimeError('No staged research files')
    request=''.join(oid+'\n' for _,_,oid in entries).encode()
    stream=io.BytesIO(subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=request,capture_output=True,check=True).stdout)
    failures=[];sizes=[];categories=Counter();symlinks=0
    for name,mode,oid in entries:
        actual,kind,length=stream.readline().decode().split();assert actual==oid and kind=='blob'
        data=stream.read(int(length));assert stream.read(1)==b'\n'
        issues=blob_issues(name,mode,data)
        if issues:failures.append({'path':name,'issues':issues})
        categories[Path(name).parts[0]]+=1;sizes.append((len(data),name));symlinks+=mode=='120000'
    return {'passed':not failures,'files':len(entries),'bytes':sum(n for n,_ in sizes),
        'MiB':round(sum(n for n,_ in sizes)/2**20,2),'internal_relative_symlinks':symlinks,
        'largest_files':[{'path':name,'bytes':size} for size,name in sorted(sizes,reverse=True)[:5]],
        'by_top_level':dict(categories),'issues':failures,
        'scope':'Checks staged blobs, not only working files; remote commit/push is not performed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=audit()
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
