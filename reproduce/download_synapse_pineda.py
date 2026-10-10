"""Download the Pineda et al. 2024 (Cell) processed snRNA-seq matrices from Synapse.

Project syn51105515: per-library counts_fil.mtx + row/col metadata (folder syn53421586,
"Processed Data") and the supplemental tables (syn53416892). Synapse refuses anonymous
downloads, so a Personal Access Token is read from ~/.synapseConfig ([authentication]
authtoken = ...) or $SYNAPSE_AUTH_TOKEN. The token is never printed.

Usage: python download_synapse_pineda.py [--dry-run]   (from anywhere; files go to
resources/Pineda2024/<folder>/<file>; existing files are skipped)
"""

from __future__ import annotations

import configparser
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://repo-prod.prod.sagebase.org/repo/v1"
ROOTS = {"syn53421586": "processed", "syn53416892": "supplemental"}
OUT = Path(__file__).resolve().parents[1] / "resources" / "Pineda2024"


def _token() -> str:
    tok = os.environ.get("SYNAPSE_AUTH_TOKEN", "")
    cfg = Path.home() / ".synapseConfig"
    if not tok and cfg.exists():
        cp = configparser.ConfigParser()
        cp.read(cfg)
        tok = cp.get("authentication", "authtoken", fallback="").strip()
    if not tok or tok.startswith("PASTE_"):
        sys.exit("No Synapse token: put it in ~/.synapseConfig (authtoken = ...) or $SYNAPSE_AUTH_TOKEN.")
    return tok


def _open(req: urllib.request.Request | str, timeout: int = 120, tries: int = 6):
    """urlopen with backoff: rate limits (429), S3's sporadic 403 and 5xx are retried.

    Waits the server's Retry-After when it gives seconds, else 10, 20, 40, 80, 160 s.
    """
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503, 504) or i == tries - 1:
                raise
            ra = e.headers.get("Retry-After") or ""
            wait = int(ra) if ra.isdigit() else 10 * 2 ** i
            print(f"  HTTP {e.code}, retrying in {wait} s", flush=True)
        except OSError as e:
            if i == tries - 1:
                raise
            wait = 10 * 2 ** i
            print(f"  {e}, retrying in {wait} s", flush=True)
        time.sleep(wait)


def _call(path: str, token: str | None = None, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(_open(req).read())


def _presigned(syn: str, token: str) -> str:
    """Signed download URL of a file entity (the endpoint answers with the URL as text)."""
    req = urllib.request.Request(f"{API}/entity/{syn}/file?redirect=false",
                                 headers={"Authorization": f"Bearer {token}"})
    return _open(req).read().decode().strip().strip('"')


def _walk(parent: str, rel: Path):
    """Yield (synapse id, relative path) for every file under *parent*."""
    token = None
    while True:
        page = _call("/entity/children", body={"parentId": parent, "includeTypes": ["folder", "file"],
                                                **({"nextPageToken": token} if token else {})})
        for c in page["page"]:
            if c["type"].endswith("Folder"):
                yield from _walk(c["id"], rel / c["name"])
            else:
                yield c["id"], rel / c["name"]
        token = page.get("nextPageToken")
        if not token:
            return


def _fetch(syn: str, token: str, dest: Path, size: int, tries: int = 3) -> None:
    """Download to *dest* via a .part file; a fresh signed URL per try (covers mid-stream drops)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(1, tries + 1):
        try:
            with _open(_presigned(syn, token), timeout=600) as r, tmp.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if tmp.stat().st_size == size:
                tmp.rename(dest)
                return
            print(f"  size mismatch on {dest.name} (try {i})", flush=True)
        except OSError as e:
            print(f"  {dest.name}: {e} (try {i})", flush=True)
        time.sleep(30 * i)
    sys.exit(f"giving up on {dest} after {tries} tries")


def main(dry_run: bool = False) -> None:
    token = None if dry_run else _token()
    n = 0
    for root, name in ROOTS.items():
        for syn, rel in _walk(root, Path(name)):
            dest = OUT / rel
            n += 1
            if dry_run:
                print(f"  would get {rel}")
                continue
            size = _call(f"/entity/{syn}/filehandles", token)["list"][0]["contentSize"]
            if dest.exists() and dest.stat().st_size == size:
                print(f"  exists: {rel}")
                continue
            _fetch(syn, token, dest, size)
            print(f"  got {rel} ({size / 1e6:.1f} MB)", flush=True)
    print(f"{n} files under {OUT}, sizes match Synapse")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
