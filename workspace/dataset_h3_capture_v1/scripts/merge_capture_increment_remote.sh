#!/usr/bin/env bash
# Safely merge the 2026-09-18 capture increment on the remote dataset host.
set -euo pipefail

ROOT="${ROOT:-/nvmedata/workspace2/users/nina/dataset_h3_capture_v1}"
STAGE="${STAGE:-$ROOT/incoming_20260918_increment_01}"
EXPECTED_INCREMENT_COUNT=109
BASELINE_COUNT=1900
FINAL_COUNT=2009
FIRST_NEW_ID=1901
LAST_NEW_ID=2009
EXPECTED_SOURCE_COUNT=336

MANIFESTS="$ROOT/manifests"
STAGE_MANIFESTS="$STAGE/manifests"
KEYPOINTS="$ROOT/hand_keypoints"
ARCHIVE="$MANIFESTS/archive/20260918_increment_01"
REPORTS="$ROOT/reports/capture_increment_20260918_increment_01"
REPLACEMENT_BACKUP="$STAGE/reports/replacement_backup"
INCREMENT_MANIFEST="$STAGE_MANIFESTS/capture_hand_detection_increment.jsonl"
PENDING_MANIFEST="$STAGE_MANIFESTS/capture_hand_detection_full_pending.jsonl"
PENDING_INVENTORY="$STAGE_MANIFESTS/source_inventory_full_pending.jsonl"
PENDING_HASHES="$STAGE_MANIFESTS/clips_full_pending.sha256"
INCREMENT_HASHES="$STAGE_MANIFESTS/clips_increment.sha256"
ROOT_MANIFEST="$MANIFESTS/capture_hand_detection_full.jsonl"
ROOT_INVENTORY="$MANIFESTS/source_inventory.jsonl"
ROOT_HASHES="$MANIFESTS/clips_full.sha256"

die() {
    printf 'merge_capture_increment_remote: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || die "required_nonempty_file_missing:$1"
}

link_or_verify() {
    local source=$1
    local destination=$2
    require_file "$source"
    mkdir -p "$(dirname "$destination")"
    if [[ -e "$destination" ]]; then
        [[ -f "$destination" ]] || die "destination_not_regular_file:$destination"
        cmp -s "$source" "$destination" || die "destination_content_conflict:$destination"
        [[ "$source" -ef "$destination" ]] || die "destination_is_not_hardlink:$destination"
        return
    fi
    ln "$source" "$destination"
}

copy_or_verify() {
    local source=$1
    local destination=$2
    require_file "$source"
    mkdir -p "$(dirname "$destination")"
    if [[ -e "$destination" ]]; then
        [[ -f "$destination" ]] || die "destination_not_regular_file:$destination"
        cmp -s "$source" "$destination" || die "destination_content_conflict:$destination"
        return
    fi
    local temporary
    temporary=$(mktemp "$(dirname "$destination")/.${destination##*/}.tmp.XXXXXX")
    cp "$source" "$temporary"
    mv "$temporary" "$destination"
}

replace_atomically() {
    local source=$1
    local destination=$2
    require_file "$source"
    mkdir -p "$(dirname "$destination")"
    if [[ -e "$destination" ]] && cmp -s "$source" "$destination"; then
        return
    fi
    local temporary
    temporary=$(mktemp "$(dirname "$destination")/.${destination##*/}.tmp.XXXXXX")
    cp "$source" "$temporary"
    mv -f "$temporary" "$destination"
}

backup_current_manifests() {
    copy_or_verify "$ROOT_MANIFEST" "$REPLACEMENT_BACKUP/capture_hand_detection_full.before_replace.jsonl"
    copy_or_verify "$ROOT_INVENTORY" "$REPLACEMENT_BACKUP/source_inventory.before_replace.jsonl"
    copy_or_verify "$ROOT_HASHES" "$REPLACEMENT_BACKUP/clips_full.before_replace.sha256"
}

rollback_main_manifests() {
    local rollback_failed=0
    if ! replace_atomically "$REPLACEMENT_BACKUP/capture_hand_detection_full.before_replace.jsonl" "$ROOT_MANIFEST"; then
        rollback_failed=1
    fi
    if ! replace_atomically "$REPLACEMENT_BACKUP/source_inventory.before_replace.jsonl" "$ROOT_INVENTORY"; then
        rollback_failed=1
    fi
    if ! replace_atomically "$REPLACEMENT_BACKUP/clips_full.before_replace.sha256" "$ROOT_HASHES"; then
        rollback_failed=1
    fi
    return "$rollback_failed"
}

replace_main_manifests() {
    backup_current_manifests
    if replace_atomically "$PENDING_MANIFEST" "$ROOT_MANIFEST" \
        && replace_atomically "$PENDING_INVENTORY" "$ROOT_INVENTORY" \
        && replace_atomically "$PENDING_HASHES" "$ROOT_HASHES"; then
        return 0
    fi
    if ! rollback_main_manifests; then
        die "main_manifest_replace_failed_and_rollback_failed"
    fi
    die "main_manifest_replace_failed_rolled_back"
}

require_file "$ROOT/scripts/validate_capture_full.py"
require_file "$ROOT/scripts/validate_capture_source_coverage.py"
for path in "$INCREMENT_MANIFEST" "$PENDING_MANIFEST" "$PENDING_INVENTORY" "$PENDING_HASHES" "$INCREMENT_HASHES" \
    "$ROOT_MANIFEST" "$ROOT_INVENTORY" "$ROOT_HASHES"; do
    require_file "$path"
done
[[ -d "$STAGE/clips" ]] || die "stage_clips_directory_missing:$STAGE/clips"
[[ -d "$STAGE/hand_keypoints" ]] || die "stage_keypoints_directory_missing:$STAGE/hand_keypoints"

merge_state=$(python3 - "$INCREMENT_MANIFEST" "$PENDING_MANIFEST" "$ROOT_MANIFEST" \
    "$EXPECTED_INCREMENT_COUNT" "$BASELINE_COUNT" "$FINAL_COUNT" "$FIRST_NEW_ID" "$LAST_NEW_ID" <<'PY'
import json
import re
import sys
from pathlib import Path

increment_path, pending_path, root_path = map(Path, sys.argv[1:4])
expected_increment, baseline, final, first_new, last_new = map(int, sys.argv[4:])
identifier = re.compile(r"^capture_(\d{6})$")

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def ids(items, label):
    values = []
    for row in items:
        sample_id = str(row.get("sample_id") or row.get("clip_id") or "")
        match = identifier.fullmatch(sample_id)
        if not match:
            raise SystemExit(f"{label}: invalid_sample_id:{sample_id!r}")
        values.append(int(match.group(1)))
    if len(values) != len(set(values)):
        raise SystemExit(f"{label}: duplicate_sample_id")
    return values

increment_rows = rows(increment_path)
pending_rows = rows(pending_path)
root_rows = rows(root_path)
increment_ids = ids(increment_rows, "increment_manifest")
pending_ids = ids(pending_rows, "pending_manifest")
root_ids = ids(root_rows, "root_manifest")
if len(increment_rows) != expected_increment or sorted(increment_ids) != list(range(first_new, last_new + 1)):
    raise SystemExit("increment_manifest: expected_109_continuous_ids_1901_2009")
if len(pending_rows) != final or sorted(pending_ids) != list(range(1, final + 1)):
    raise SystemExit("pending_manifest: expected_2009_continuous_ids_1_2009")
increment_names = {str(row.get("clip_file") or "") for row in increment_rows}
if len(increment_names) != expected_increment or any(not name or Path(name).name != name for name in increment_names):
    raise SystemExit("increment_manifest: invalid_or_duplicate_clip_file")
if len(root_rows) == baseline and sorted(root_ids) == list(range(1, baseline + 1)):
    root_clip_names = {str(row.get("clip_file") or "") for row in root_rows}
    collision_names = sorted(increment_names & root_clip_names)
    if collision_names:
        raise SystemExit(f"increment_manifest: clip_file_conflicts_with_old_manifest:{collision_names[:5]}")
    print("baseline")
elif len(root_rows) == final and sorted(root_ids) == list(range(1, final + 1)) and root_path.read_bytes() == pending_path.read_bytes():
    print("merged")
else:
    raise SystemExit("root_manifest: expected_baseline_1900_or_exact_pending_2009")
PY
) || die "manifest_preflight_failed"

[[ "$merge_state" == baseline || "$merge_state" == merged ]] || die "unexpected_merge_state:$merge_state"

stage_clip_count=$(find "$STAGE/clips" -maxdepth 1 -type f -name 'capture_*.mp4' -printf '.' | wc -c)
stage_json_count=$(find "$STAGE/hand_keypoints" -maxdepth 1 -type f -name 'capture_*_hands.json' -printf '.' | wc -c)
[[ "$stage_clip_count" -eq "$EXPECTED_INCREMENT_COUNT" ]] || die "stage_clip_count_expected_109_actual:$stage_clip_count"
[[ "$stage_json_count" -eq "$EXPECTED_INCREMENT_COUNT" ]] || die "stage_canonical_json_count_expected_109_actual:$stage_json_count"

if [[ "$merge_state" == baseline ]]; then
    copy_or_verify "$ROOT_MANIFEST" "$ARCHIVE/capture_hand_detection_full.before_merge.jsonl"
    copy_or_verify "$ROOT_INVENTORY" "$ARCHIVE/source_inventory.before_merge.jsonl"
    copy_or_verify "$ROOT_HASHES" "$ARCHIVE/clips_full.before_merge.sha256"
else
    require_file "$ARCHIVE/capture_hand_detection_full.before_merge.jsonl"
    require_file "$ARCHIVE/source_inventory.before_merge.jsonl"
    require_file "$ARCHIVE/clips_full.before_merge.sha256"
fi

while IFS=$'\t' read -r sample_id clip_file; do
    sample_id=${sample_id%$'\r'}
    clip_file=${clip_file%$'\r'}
    [[ -n "$sample_id" && -n "$clip_file" ]] || die "invalid_increment_asset_row"
    source_clip="$STAGE/clips/$clip_file"
    source_json="$STAGE/hand_keypoints/${sample_id}_hands.json"
    link_or_verify "$source_clip" "$ROOT/clips/$clip_file"
    link_or_verify "$source_json" "$KEYPOINTS/${sample_id}_hands.json"
    link_or_verify "$source_json" "$KEYPOINTS/flat/${sample_id}_hands.json"
    link_or_verify "$source_json" "$KEYPOINTS/shard_09/${sample_id}_hands.json"
done < <(python3 - "$INCREMENT_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(encoding="utf-8-sig").splitlines():
    if line.strip():
        row = json.loads(line)
        sample_id = str(row.get("sample_id") or row.get("clip_id") or "")
        clip_file = str(row.get("clip_file") or "")
        if Path(clip_file).name != clip_file:
            raise SystemExit(f"invalid_clip_file:{clip_file!r}")
        print(f"{sample_id}\t{clip_file}")
PY
)

mkdir -p "$STAGE/reports"
python3 "$ROOT/scripts/validate_capture_full.py" \
    --root "$ROOT" \
    --manifest "$PENDING_MANIFEST" \
    --out "$STAGE/reports/validate_capture_full_pending.json"
if ! python3 "$ROOT/scripts/validate_capture_source_coverage.py" \
    --inventory "$PENDING_INVENTORY" \
    --manifest "$PENDING_MANIFEST" \
    --expected-source-count "$EXPECTED_SOURCE_COUNT" \
    --out "$STAGE/reports/source_coverage_full_pending.json"; then
    die "source_coverage_validation_failed"
fi
python3 - "$STAGE/reports/source_coverage_full_pending.json" "$EXPECTED_SOURCE_COUNT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if report.get("expected_source_count") != expected:
    raise SystemExit("source_coverage_report_expected_count_missing")
if report.get("expected_source_count_mismatch") or report.get("error_count") != 0:
    raise SystemExit("source_coverage_report_has_errors")
PY

copy_or_verify "$INCREMENT_MANIFEST" "$MANIFESTS/capture_hand_detection_increment_20260918_increment_01.jsonl"
copy_or_verify "$INCREMENT_HASHES" "$MANIFESTS/clips_increment_20260918_increment_01.sha256"
copy_or_verify "$STAGE/reports/validate_capture_full_pending.json" "$REPORTS/validate_capture_full.json"
copy_or_verify "$STAGE/reports/source_coverage_full_pending.json" "$REPORTS/source_coverage.json"

if [[ "$merge_state" == baseline ]]; then
    replace_main_manifests
else
    cmp -s "$PENDING_MANIFEST" "$ROOT_MANIFEST" || die "merged_state_manifest_mismatch"
    cmp -s "$PENDING_INVENTORY" "$ROOT_INVENTORY" || die "merged_state_inventory_mismatch"
    cmp -s "$PENDING_HASHES" "$ROOT_HASHES" || die "merged_state_hashes_mismatch"
fi

python3 - "$ROOT_MANIFEST" "$ROOT_INVENTORY" "$ROOT/clips" "$KEYPOINTS" "$FINAL_COUNT" "$merge_state" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, inventory_path, clips_path, keypoints_path, expected, merge_state = sys.argv[1:]
expected = int(expected)
clips_path = Path(clips_path)
keypoints_path = Path(keypoints_path)
manifest_rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
inventory_rows = [json.loads(line) for line in Path(inventory_path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
if len(manifest_rows) != expected:
    raise SystemExit(f"final_manifest_count_expected_{expected}_actual_{len(manifest_rows)}")
if len({str(row.get('source_name') or '') for row in inventory_rows}) != len(inventory_rows):
    raise SystemExit("final_inventory_duplicate_source_name")
print(json.dumps({
    "final_manifest_count": len(manifest_rows),
    "final_source_count": len(inventory_rows),
    "canonical_clip_count": sum(path.is_file() for path in clips_path.glob("capture_*.mp4")),
    "canonical_keypoint_count": sum(path.is_file() for path in keypoints_path.glob("capture_*_hands.json")),
    "merge_state_at_start": merge_state,
}, ensure_ascii=False))
PY

printf 'merge_complete root=%s stage=%s started_from=%s\n' "$ROOT" "$STAGE" "$merge_state"
