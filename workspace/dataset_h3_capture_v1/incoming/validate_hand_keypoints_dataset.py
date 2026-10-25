#!/usr/bin/env python3
"""Validate hand-keypoint JSON files and summarize one public dataset source."""
import argparse
import json
from collections import Counter
from pathlib import Path


def valid_payload(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            return False, "empty_frames", payload
        for frame in frames:
            if not isinstance(frame, dict):
                return False, "bad_frame", payload
            for hand in frame.get("hands", []):
                points = hand.get("points")
                if not isinstance(points, list) or len(points) != 21:
                    return False, "not_21_points", payload
                if any(not isinstance(point, list) or len(point) != 3 for point in points):
                    return False, "not_xyz", payload
        return True, "valid", payload
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, "unreadable_json", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--keypoint-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clip_dir = Path(args.clip_dir)
    keypoint_dir = Path(args.keypoint_dir)
    rows = []
    reasons = Counter()
    visible_rates = []
    two_hand_rates = []
    for row in manifest:
        sample_id = row["sample_id"]
        clip = clip_dir / f"{sample_id}.mp4"
        json_path = keypoint_dir / f"{sample_id}_hands.json"
        item = {"sample_id": sample_id, "clip_exists": clip.is_file(), "json_exists": json_path.is_file()}
        if json_path.is_file():
            ok, reason, payload = valid_payload(json_path)
            item["json_valid"] = ok
            reasons[reason] += 1
            if ok:
                summary = payload.get("summary", {})
                visible = summary.get("hand_visible_rate")
                two = summary.get("two_hand_rate")
                if isinstance(visible, (int, float)):
                    visible_rates.append(float(visible))
                if isinstance(two, (int, float)):
                    two_hand_rates.append(float(two))
                item.update({"hand_visible_rate": visible, "two_hand_rate": two})
        else:
            item["json_valid"] = False
            reasons["missing_json"] += 1
        rows.append(item)
    report = {
        "manifest_samples": len(manifest),
        "clips_present": sum(item["clip_exists"] for item in rows),
        "json_present": sum(item["json_exists"] for item in rows),
        "json_valid": sum(item["json_valid"] for item in rows),
        "hand_visible_ge_30": sum(rate >= 0.30 for rate in visible_rates),
        "hand_visible_ge_50": sum(rate >= 0.50 for rate in visible_rates),
        "hand_visible_ge_70": sum(rate >= 0.70 for rate in visible_rates),
        "hand_visible_ge_80": sum(rate >= 0.80 for rate in visible_rates),
        "two_hand_ge_50": sum(rate >= 0.50 for rate in two_hand_rates),
        "invalid_reasons": dict(reasons),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": report, "samples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
