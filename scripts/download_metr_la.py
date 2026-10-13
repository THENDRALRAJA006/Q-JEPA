"""
Download and verify METR-LA dataset from Zenodo.
DOI: 10.5281/zenodo.5724362 (License: CC BY 4.0)
Files:
- METR-LA.csv (~69 MB)
- adj_mx_METR-LA.pkl (~680 KB)
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

DEST_DIR = Path("data/metr_la")
DEST_DIR.mkdir(parents=True, exist_ok=True)

CSV_URL = "https://zenodo.org/records/5724362/files/METR-LA.csv?download=1"
CSV_PATH = DEST_DIR / "METR-LA.csv"

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        print(f"File {dest_path} already exists ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return

    print(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        total_len = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 512
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_len > 0:
                pct = downloaded / total_len * 100
                print(f"\r  {downloaded / (1024*1024):.1f}/{total_len / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / (1024*1024):.1f} MB downloaded", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\nDownload finished in {elapsed:.1f}s ({dest_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_file(CSV_URL, CSV_PATH)
