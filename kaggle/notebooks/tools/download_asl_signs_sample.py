"""Download train.csv, the sign map and a seeded per-sign sample of the asl-signs sequence files.

For running the ManuVox notebooks locally in FAST mode. Needs the `kaggle` package, a Kaggle API
token (~/.kaggle/access_token, ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN) and the competition
rules accepted on kaggle.com.

Kaggle rate-limits its per-file download endpoint to roughly one file a second, so this script
makes a single API call for the signed URL of the competition's zip archive and then reads only
the members it needs with HTTP range requests: the zip's central directory, train.csv, the sign
map and the sampled parquet files. Every member is checked against the CRC-32 stored in the zip
and written atomically. Files keep the competition's layout under <out>/asl-signs/, so the
notebooks read them with KAGGLE_INPUT_DIR=<out>. The default <out> is under data/, which git
ignores: the data stays on this machine. Reruns skip files that are already there.

    python kaggle/notebooks/tools/download_asl_signs_sample.py --per-sign 24
"""

import argparse
import csv
import io
import os
import random
import struct
import sys
import threading
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

COMPETITION = "asl-signs"
TOP_FILES = ("train.csv", "sign_to_prediction_index_map.json")
LOCAL_HEADER = struct.Struct("<4s5H3L2H")  # Signature ... file name length, extra field length.
SLACK = 1024  # Bytes fetched beyond the central directory's sizes, for the local extra field.


def archive_url():
    """The signed download URL of the competition's zip archive (one Kaggle API call)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = COMPETITION
        response = kaggle.competitions.competition_api_client.download_data_files(request)
    url = response.url
    response.close()  # Only the final (redirected) URL is needed, not the 50+ GB body.
    return url


class RangeReader(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, enough for zipfile's directory scan."""

    def __init__(self, url, session):
        self.url, self.session, self.position = url, session, 0
        head = session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        data = get_range(self.session, self.url, self.position, end)
        self.position += len(data)
        return data


def get_range(session, url, start, end, attempts=4):
    """Bytes [start, end) of the URL, retrying network errors with backoff."""
    for attempt in range(attempts):
        try:
            response = session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"expected 206 Partial Content, got {response.status_code}")
            return response.content
        except (requests.RequestException, RuntimeError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def member_bytes(session, url, info):
    """One zip member's uncompressed bytes, checked against its CRC-32."""
    start = info.header_offset
    name_length = len(info.orig_filename.encode("utf-8"))
    data = get_range(session, url, start, start + LOCAL_HEADER.size + name_length + SLACK + info.compress_size)
    fields = LOCAL_HEADER.unpack(data[: LOCAL_HEADER.size])
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"{info.filename}: no local file header at offset {start}")
    offset = LOCAL_HEADER.size + fields[-2] + fields[-1]
    if offset + info.compress_size > len(data):  # A local extra field longer than SLACK.
        data = get_range(session, url, start, start + offset + info.compress_size)
    raw = data[offset : offset + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        content = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"{info.filename}: unsupported compression {info.compress_type}")
    if zlib.crc32(content) != info.CRC or len(content) != info.file_size:
        raise RuntimeError(f"{info.filename}: CRC or size mismatch")
    return content


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(content)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/kaggle_input", type=Path)
    parser.add_argument("--per-sign", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.out / COMPETITION
    root.mkdir(parents=True, exist_ok=True)
    url = archive_url()
    session = requests.Session()
    with zipfile.ZipFile(RangeReader(url, session)) as archive:
        members = {info.filename: info for info in archive.infolist()}
    print(f"archive: {len(members):,} members, central directory read")

    for name in TOP_FILES:
        if not (root / name).exists():
            write_atomic(root / name, member_bytes(session, url, members[name]))
    with (root / "train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_sign = defaultdict(list)
    for row in rows:
        by_sign[row["sign"]].append(row["path"])
    rng = random.Random(args.seed)
    sample = [p for sign in sorted(by_sign) for p in rng.sample(sorted(by_sign[sign]), min(args.per_sign, len(by_sign[sign])))]
    todo = [p for p in sample if not (root / p).exists() or (root / p).stat().st_size != members[p].file_size]
    print(f"{len(rows):,} sequences listed; sample {len(sample):,} ({args.per_sign} per sign); {len(todo):,} to download", flush=True)

    local = threading.local()

    def fetch(relative):
        if not hasattr(local, "session"):
            local.session = requests.Session()
        write_atomic(root / relative, member_bytes(local.session, url, members[relative]))
        return relative

    started, failed = time.time(), []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, p): p for p in todo}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - report every failure, keep the rest going
                failed.append(f"{futures[future]}: {type(error).__name__}: {error}"[:200])
            if done % 500 == 0 or done == len(todo):
                print(f"{done:,}/{len(todo):,} files, {time.time() - started:.0f} s", flush=True)
    present = sum((root / p).exists() for p in sample)
    print(f"done: {present:,} of {len(sample):,} sampled files on disk; {len(failed)} failed")
    for line in failed[:10]:
        print("  failed:", line)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
