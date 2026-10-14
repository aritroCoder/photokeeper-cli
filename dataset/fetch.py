# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub", "pyarrow"]
# ///
"""Rebuild dataset/ from manifest.csv: `uv run dataset/fetch.py`.

The images are not redistributed with PhotoKeeper; each one is fetched from its source under
that source's own terms. Existing files are skipped, so a failed run can simply be rerun.
"""
import csv
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).parent
IMAGE_COLUMN = {"creative-graphic-design/Rico": "screenshot", "chainyo/rvl-cdip": "image", "lmms-lab-encoder/flickr30k": "image"}
UNSPLASH = "1aurent/unsplash-lite"
fs = HfFileSystem()


def read_column(repo, file, column):
    with fs.open(f"datasets/{repo}/{file}", "rb") as fh:
        return pq.read_table(fh, columns=[column]).column(0)


def unsplash_urls():
    urls = {}
    for i in range(3):
        photos = read_column(UNSPLASH, f"data/train-0000{i}-of-00003.parquet", "photo").flatten()
        urls.update(zip(photos[0].to_pylist(), photos[2].to_pylist()))  # photo.id -> photo.image_url
    return urls


def main():
    rows = list(csv.DictReader(open(ROOT / "manifest.csv")))
    todo = [r for r in rows if not (ROOT / r["path"]).exists()]
    by_parquet = defaultdict(list)
    urls = None
    for r in todo:
        if m := re.fullmatch(r"(.+\.parquet)\[(\d+)\]", r["location"]):
            by_parquet[(r["source"], m[1])].append((int(m[2]), r["path"]))
        elif r["source"] == UNSPLASH:
            urls = urls or unsplash_urls()
            write(r["path"], urllib.request.urlopen(f"{urls[r['location']]}?w=1600&fm=jpg").read())  # size the manifest was built from
        else:
            write(r["path"], fs.read_bytes(f"datasets/{r['source']}/{r['location']}"))
    for (repo, file), items in by_parquet.items():
        column = read_column(repo, file, IMAGE_COLUMN[repo])
        for index, path in items:
            write(path, column[index].as_py()["bytes"])
    print(f"{len(todo)} fetched, {len(rows) - len(todo)} already present")


def write(path, data):
    assert data, path
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(path)


if __name__ == "__main__":
    main()
