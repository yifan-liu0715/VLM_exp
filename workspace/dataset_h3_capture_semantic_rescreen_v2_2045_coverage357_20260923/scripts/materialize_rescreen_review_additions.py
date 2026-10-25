#!/usr/bin/env python3
"""Create a compact, human-readable hardlink folder for rescreen additions."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result[:48] or "action"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    audit = json.loads((root / "reports/rescreen_v2_final_audit.json").read_text(encoding="utf-8"))
    added_ids = set(audit["added_ids"])
    candidates = {
        row["candidate_id"]: row
        for row in load_jsonl(root / "manifests/semantic_review_candidates_2045.jsonl")
    }
    exports = {row["clip_id"]: row for row in load_jsonl(root / "manifests/export_manifest.jsonl")}
    rows = sorted(
        (candidates[clip_id] for clip_id in added_ids),
        key=lambda row: (row["source_rel"], float(row["source_start_sec"]), row["candidate_id"]),
    )
    review_dir = root / "review_additions_14"
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    expected_names = set()
    for index, row in enumerate(rows, 1):
        clip_id = row["candidate_id"]
        name = f"{index:02d}_{slug(str(row['action_family']))}_{slug(str(row['action_label']))}_{clip_id[-8:]}.mp4"
        expected_names.add(name)
        source = Path(exports[clip_id]["output_video"])
        target = review_dir / name
        if target.exists():
            if not os.path.samefile(source, target):
                raise RuntimeError(f"review target is not the expected hardlink: {target}")
        else:
            os.link(source, target)
        manifest.append({
            "review_index": index,
            "review_file": name,
            "clip_id": clip_id,
            "source_rel": row["source_rel"],
            "source_start_sec": row["source_start_sec"],
            "source_end_sec": row["source_end_sec"],
            "duration_sec": row["duration_sec"],
            "action_family": row["action_family"],
            "action_label": row["action_label"],
            "review_status": row["review_status"],
        })
    unexpected = sorted(path.name for path in review_dir.glob("*.mp4") if path.name not in expected_names)
    if len(manifest) != 14 or unexpected:
        raise RuntimeError(f"review folder mismatch: count={len(manifest)} unexpected={unexpected}")
    (review_dir / "additions_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in manifest),
        encoding="utf-8",
    )
    print(json.dumps({"review_video_count": len(manifest), "unexpected_count": len(unexpected), "path": str(review_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
