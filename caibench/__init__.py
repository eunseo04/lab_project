import json
from pathlib import Path
import time
from environment.ctf import CTF

# NOTE: Despite the `.jsonl` extension this file is a single JSON array that may
# contain full-line `#` comments (e.g. Korean notes). `json.load` cannot parse
# either, and on Windows the default cp949 codec chokes on non-ASCII bytes, so we
# load it explicitly as UTF-8 and strip comment-only lines first.
_PACKAGE_DIR = Path(__file__).parent


def _config_path():
    """Locate ctf_configs.jsonl, tolerating both the flat and ctf-jsons/ layouts."""
    candidates = [
        _PACKAGE_DIR / 'ctf_configs.jsonl',
        _PACKAGE_DIR / 'ctf-jsons' / 'ctf_configs.jsonl',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"ctf_configs.jsonl not found. Looked in: {', '.join(str(c) for c in candidates)}"
    )


def load_configs():
    """Load and parse the CTF config list, stripping comment-only lines."""
    path = _config_path()
    raw_lines = path.read_text(encoding='utf-8').splitlines()
    # Drop lines whose first non-whitespace char is '#'. '#' inside JSON string
    # values never starts a line, so this is safe.
    cleaned = "\n".join(
        line for line in raw_lines if not line.lstrip().startswith('#')
    )
    return json.loads(cleaned)


# 원하는 CTF 이름을 찾아서 CTF 객체로 만들어 반환
def ctf(name=None, list=None, **kwargs):
    """
    Initialize a CTF object and report execution time.

    Takes about 0.0057 seconds to find a CTF within the existing 1500 CTFs.
    """
    start_time = time.time()  # Start timing

    if list or name:
        configs = load_configs()
        # name이 같은 환경 찾기
        if name:
            for config in configs:
                if config['name'].lower() == name.lower():
                    if 'container_name' in config:
                        config['container_name'] = config['container_name'].lower()
                    # 기존 json 설정에 추가 설정 더하기 가능
                    config = {**config, **kwargs}
                    end_time = time.time()  # End timing
                    print(f"CAIBench, time it took to find CTF: {end_time - start_time:.4f} seconds")
                    # CTF 객체 생성
                    return CTF(config)
            raise ValueError(f"No CTF configuration found for '{name}'")

    if not list and not name:
        raise ValueError("Either 'name' or 'list' must be provided")

    end_time = time.time()  # End timing
    print(f"CAIBench, time it took to find CTF: {end_time - start_time:.4f} seconds")


def list():
    """
    List all CTFs.
    """
    configs = load_configs()
    ctf_list = []
    for config in configs:
        ctf_info = {
            'name': config.get('name'),
            'difficulty': config.get('difficulty'),
            'type': config.get('type'),
            'challenge': config.get('challenge')
        }
        ctf_list.append(ctf_info)

    return ctf_list
