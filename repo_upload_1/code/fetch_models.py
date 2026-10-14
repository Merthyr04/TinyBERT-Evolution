"""Prefetch the three pretrained checkpoints into a plain local folder.

huggingface.co is unreachable from this machine and hf-mirror.com throttles a
single connection to ~365 kB/s, so the large weight files are fetched with the
same multi-connection downloader used for the torch wheel.  Small side files
(config / vocab / tokenizer) go through a plain GET.

Everything lands in <project>/hf_models/<name>/ so `from_pretrained` can be
pointed at a local directory and no further network access is needed.
"""
from __future__ import annotations

import os
import sys
import urllib.request

from fetch_wheel import download

MIRROR = "https://hf-mirror.com"
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("HF_MODELS_DIR", os.path.join(os.path.dirname(_HERE), "hf_models"))

REPOS = [
    ("bert-base-uncased", [
        "config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json",
        "pytorch_model.bin"]),
    ("bert-base-multilingual-cased", [
        "config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json",
        "pytorch_model.bin"]),
    ("huawei-noah/TinyBERT_General_4L_312D", [
        "config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json",
        "pytorch_model.bin"]),
]

BIG = {"pytorch_model.bin", "model.safetensors", "tokenizer.json"}


def get(url, path, proxy=None, retries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    opener = urllib.request.build_opener()
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    for a in range(retries):
        try:
            with opener.open(url, timeout=90) as r, open(path, "wb") as f:
                while True:
                    b = r.read(1 << 18)
                    if not b:
                        break
                    f.write(b)
            return True
        except Exception as e:                      # noqa: BLE001
            print(f"  retry {a} {url}: {e}", flush=True)
    return False


def main():
    proxy = os.environ.get("HTTPS_PROXY") or None
    threads = int(os.environ.get("FETCH_THREADS", "16"))
    for repo, files in REPOS:
        name = repo.split("/")[-1]
        out = os.path.join(ROOT, name).replace("\\", "/")
        os.makedirs(out, exist_ok=True)
        print(f"=== {repo} -> {out}", flush=True)
        for fn in files:
            url = f"{MIRROR}/{repo}/resolve/main/{fn}"
            path = os.path.join(out, fn)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"  have {fn} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
                continue
            if fn in BIG:
                try:
                    rc = download(url, path, threads=threads, proxy=proxy,
                                  chunk_mb=16)
                    if rc != 0:
                        print(f"  FAILED {fn} rc={rc}", flush=True)
                except Exception as e:              # noqa: BLE001
                    print(f"  FAILED {fn}: {e}", flush=True)
            else:
                ok = get(url, path, proxy)
                print(f"  {'ok ' if ok else 'FAIL'} {fn}"
                      f" ({os.path.getsize(path) if os.path.exists(path) else 0} B)",
                      flush=True)
    print("model prefetch finished", flush=True)
    for repo, _ in REPOS:
        name = repo.split("/")[-1]
        d = os.path.join(ROOT, name)
        tot = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        print(f"{name}: {tot / 1e6:.1f} MB, files={sorted(os.listdir(d))}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
