#!/usr/bin/env python3
"""Validate a possibly-CRLF SHA256SUMS file without altering filenames."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="manifests/clips.sha256")
    args = parser.parse_args()
    root = Path(args.root)
    lines = (root / args.manifest).read_bytes().splitlines()
    ok = bad = missing = 0
    bad_names: list[str] = []
    for raw in lines:
        raw = raw.rstrip(b"\r")
        try:
            expected, relative = raw.decode("utf-8").split("  ", 1)
        except (UnicodeDecodeError, ValueError):
            bad += 1
            bad_names.append("<malformed>")
            continue
        path = root / relative
        if not path.is_file():
            missing += 1
            bad_names.append(relative)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            ok += 1
        else:
            bad += 1
            bad_names.append(relative)
    print(f"entries={len(lines)} ok={ok} bad={bad} missing={missing}")
    if bad_names:
        print("first_bad=" + ",".join(bad_names[:10]))
    return 0 if len(lines) == ok and bad == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
