#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shards", type=int, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.shards):
        shard = rows[index::args.shards]
        (out_dir / f"shard_{index:02d}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard), encoding="utf-8"
        )
    print(json.dumps({"rows": len(rows), "shards": args.shards, "sizes": [len(rows[i::args.shards]) for i in range(args.shards)]}))


if __name__ == "__main__":
    main()
