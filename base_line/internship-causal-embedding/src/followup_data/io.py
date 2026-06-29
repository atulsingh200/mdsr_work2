"""HTTP / git / huggingface download helpers shared by loaders."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk: int = 1 << 14) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for block in r.iter_content(chunk):
                if block:
                    f.write(block)
                    bar.update(len(block))
    tmp.rename(dest)
    return dest


def extract(archive: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest_dir)
    elif name.endswith((".tar.bz2", ".tbz2")):
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(dest_dir)
    elif name.endswith(".tar"):
        with tarfile.open(archive, "r:") as t:
            t.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive}")
    return dest_dir


def git_clone(url: str, dest: Path, depth: int = 1) -> Path:
    dest = Path(dest)
    if dest.exists() and (dest / ".git").exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--depth", str(depth), url, str(dest)],
        check=True,
    )
    return dest


def hf_snapshot(repo_id: str, dest: Path, repo_type: str = "dataset") -> Path:
    """Download a HuggingFace dataset/model snapshot."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(dest),
    )
    return dest
