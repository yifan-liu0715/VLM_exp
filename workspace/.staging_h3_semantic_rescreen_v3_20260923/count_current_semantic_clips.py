#!/usr/bin/env python3
"""Count current remote clips against the semantic manifest."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


ANIMAL_TERMS = (
    "animal", "pet", "cat", "dog", "rabbit", "kitten", "puppy", "schnauzer",
    "dachshund", "husky", "retriever", "sighthound", "canine", "feline",
)
EXPLICIT_TARGET_TYPES = {"pet", "animal", "live_animal"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def row_text(row: dict) -> str:
    return " ".join(str(row.get(key) or "") for key in (
        "action_family", "action_label", "target_type", "target_id", "target_description",
        "action_direction", "object_state_before", "object_state_after", "response_description",
    )).lower()


def main() -> int:
    root = Path(sys.argv[1])
    manifest = Path(sys.argv[2])
    rows = {str(row["clip_id"]): row for row in load_jsonl(manifest)}
    current_ids = {
        path.stem for path in (root / "clips").glob("*.mp4") if path.is_file()
    }
    active = [rows[clip_id] for clip_id in sorted(current_ids) if clip_id in rows]
    missing_from_manifest = sorted(current_ids - set(rows))
    animal_explicit = [
        row for row in active
        if str(row.get("action_family") or "").lower().startswith("animal_")
        or str(row.get("target_type") or "").lower() in EXPLICIT_TARGET_TYPES
    ]
    animal_text = [row for row in active if any(term in row_text(row) for term in ANIMAL_TERMS)]
    animal_sources = {
        str(row.get("source_sha256")) for row in active
        if str(row.get("action_family") or "").lower().startswith("animal_")
        or str(row.get("target_type") or "").lower() in EXPLICIT_TARGET_TYPES
    }
    source_broad = [row for row in active if str(row.get("source_sha256")) in animal_sources]
    result = {
        "current_mp4_count": len(current_ids),
        "manifest_row_count": len(rows),
        "active_manifest_rows": len(active),
        "missing_from_manifest_count": len(missing_from_manifest),
        "missing_from_manifest_examples": missing_from_manifest[:20],
        "explicit_animal_clip_count": len(animal_explicit),
        "explicit_animal_source_count": len(animal_sources),
        "animal_term_clip_count": len(animal_text),
        "animal_source_scope_clip_count": len(source_broad),
        "explicit_animal_family_counts": dict(collections.Counter(str(row.get("action_family") or "") for row in animal_explicit)),
        "explicit_animal_source_names": sorted({str(row.get("source_rel") or "") for row in animal_explicit}),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
