#!/usr/bin/env python3
"""Build a pending semantic review pool without exporting media or running detection.

Presence records are navigation ROIs only.  V3, high-recall, and optional
manual decisions propose action cores; selection is deliberately separate so
only unique, export-eligible action hypotheses can fill the requested target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


POLICY_ID = "h3-semantic-recut-review-20260923"
MIN_WINDOW_SEC = 2.0
MAX_WINDOW_SEC = 20.0
DEFAULT_CONTEXT_SEC = 1.5
SHA_RE = re.compile(r"[0-9a-f]{64}")

ACTION_FAMILIES = (
    "shake", "pull_open", "pull", "close", "pick_up", "put_down", "rotate", "fold",
    "unfold", "drag", "pinch", "squeeze", "poke", "press", "bend", "straighten",
    "articulate", "transform", "gesture_prompt", "offer_feed",
)
ANIMAL_WORDS = ("cat", "cats", "dog", "dogs", "animal", "animals", "bird", "birds", "pet", "pets", "horse", "horses", "rabbit", "rabbits", "fish", "paw", "paws", "猫", "狗", "犬", "动物", "宠物", "鸟", "兔", "马", "鱼", "爪")
REAL_ANIMAL_WORDS = ("cat", "cats", "dog", "dogs", "bird", "birds", "pet", "pets", "horse", "horses", "rabbit", "rabbits", "fish", "猫", "狗", "犬", "宠物", "鸟", "兔", "马", "鱼")
BREED_WORDS = (
    "schnauzer", "husky", "sighthound", "golden retriever", "dachshund", "cavalier",
    "poodle", "poodles", "retriever", "retrievers", "beagle", "beagles", "牧羊犬", "雪纳瑞", "哈士奇", "贵宾犬", "猎犬",
)
PLUSH_WORDS = ("plush", "stuffed", "teddy", "doll", "毛绒", "玩偶", "布娃娃")
TOY_WORDS = ("toy", "玩具")
NON_LIVING_ANIMAL_MARKERS = ("ornament", "decoration", "figurine", "statue", "artificial", "artwork", "painting", "picture", "image", "print", "摆件", "装饰", "雕像", "仿真", "画", "画像", "图案")
ANIMAL_FAMILIES = {"animal_feed", "animal_eat", "animal_paw", "animal_approach", "animal_response", "animal_contact"}
OBJECT_FAMILY_PREFIXES = ("animal", "plush", "toy")
CANONICAL_FAMILIES = set(ACTION_FAMILIES) | {"interaction_unverified", *ANIMAL_FAMILIES, "plush_contact", "toy_contact"}
CANONICAL_FAMILIES.update(
    f"{prefix}_{action}" for prefix in OBJECT_FAMILY_PREFIXES for action in ACTION_FAMILIES
)
SEMANTIC_FIELDS = (
    "target_type", "target_id", "target_instance_id", "action_direction", "direction",
    "object_state_before", "object_state_after", "pre_state", "post_state", "action_round",
    "camera_motion", "target", "target_description", "response_type", "response_description",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _round(value: float) -> float:
    return round(value, 6)


def _source_rel(row: dict[str, Any]) -> str | None:
    raw = str(row.get("source_rel") or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or not path.parts or ":" in path.parts[0]:
        return None
    if any(part in {".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _source_sha(row: dict[str, Any]) -> str | None:
    value = str(row.get("source_sha256") or row.get("sha256") or "").lower()
    return value if SHA_RE.fullmatch(value) else None


def _row_id(row: dict[str, Any], prefix: str) -> str:
    value = row.get("clip_id") or row.get("candidate_id") or row.get("id")
    if value:
        return str(value)
    payload = json.dumps(row, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _interval(row: dict[str, Any], *, presence: bool = False) -> tuple[float, float] | None:
    starts = ("hand_presence_start_sec_abs", "hand_onset_sec_abs", "source_start_sec") if presence else ("source_start_sec", "hand_onset_sec_abs")
    ends = ("hand_presence_end_sec_abs", "hand_offset_sec_abs", "source_end_sec") if presence else ("source_end_sec", "hand_offset_sec_abs")
    start = next((value for key in starts if (value := _float(row.get(key))) is not None), None)
    end = next((value for key in ends if (value := _float(row.get(key))) is not None), None)
    return (start, end) if start is not None and end is not None and start < end else None


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def build_review_index(
    presence_rows: list[dict[str, Any]], v3_rows: list[dict[str, Any]], highrecall_rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    """Group all evidence by portable source identity without making approval claims."""
    index: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"presence": [], "v3": [], "highrecall": [], "invalid": []}
    )
    for kind, rows in (("presence", presence_rows), ("v3", v3_rows), ("highrecall", highrecall_rows)):
        for row in rows:
            source, sha = _source_rel(row), _source_sha(row)
            if source and sha:
                index[(source, sha)][kind].append(row)
            else:
                index[("__invalid__", kind)]["invalid"].append(
                    {"candidate_id": _row_id(row, kind), "row_id": _row_id(row, kind), "kind": kind,
                     "exportable": False, "export_eligible": False, "navigation_only": False,
                     "export_block_reason": "invalid_source_identity", "pending_user_review": True,
                     "review_status": "pending_user_review"}
                )
    return dict(index)


def _inventory(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source, sha = _source_rel(row), _source_sha(row)
        probe = row.get("probe") if isinstance(row.get("probe"), dict) else {}
        duration = _float(probe.get("duration_sec")) or _float(row.get("duration_sec"))
        if not source or not sha or duration is None or duration <= 0:
            continue
        key = (source, sha)
        if key in result:
            raise ValueError(f"duplicate inventory source identity: {source}")
        result[key] = {"duration": duration, "source_name": str(row.get("source_name") or Path(source).name)}
    return result


def _core(row: dict[str, Any]) -> tuple[float, float] | None:
    if "core_start_sec_abs" in row or "core_end_sec_abs" in row:
        start, end = _float(row.get("core_start_sec_abs")), _float(row.get("core_end_sec_abs"))
        return (start, end) if start is not None and end is not None and start < end else None
    source_start = _float(row.get("source_start_sec"))
    relative_start, relative_end = _float(row.get("action_start_sec_rel")), _float(row.get("action_end_sec_rel"))
    if source_start is not None and relative_start is not None and relative_end is not None:
        start, end = source_start + relative_start, source_start + relative_end
        return (start, end) if start < end else None
    return _interval(row)


def _row_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    return " ".join(str(row.get(field) or "") for field in fields).lower()


def _contains_term(text: str, term: str) -> bool:
    """Match ASCII terms at word boundaries; allow Chinese substring terms."""
    if any(ord(character) > 127 for character in term):
        return term in text
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _base_action_family(text: str) -> str:
    if _contains_any(text, ("zip up", "zipped up", "zip close", "close zip", "zipped", "zips", "closing", "closes", "closed", "拉上", "合上")):
        return "close"
    if _contains_any(text, ("unzip", "unzips", "unzipped", "unzipping", "opened", "opens", "opening", "打开", "拉开")):
        return "pull_open"
    if _contains_any(text, ("unscrew", "unscrews", "unscrewed", "unscrewing", "screw", "screws", "screwed", "screwing", "turn", "turns", "turned", "turning", "rotated", "rotates", "rotating", "rotate", "旋转", "转动")):
        return "rotate"
    if _contains_any(text, ("place", "places", "placed", "placing", "set down", "lower", "lowers", "lowered", "lowering", "放下")):
        return "put_down"
    if _contains_any(text, ("pull", "pulls", "pulled", "pulling", "拉")) and _contains_any(text, ("open", "door", "drawer", "打开")):
        return "pull_open"
    if _contains_any(text, ("pull", "pulls", "pulled", "pulling", "拉")):
        return "pull"
    if _contains_term(text, "put") and _contains_any(text, ("down", "place", "set")):
        return "put_down"
    if _contains_any(text, ("pick up", "lift", "lifts", "lifted", "lifting", "拿起", "举起")):
        return "pick_up"
    if _contains_any(text, ("unfold", "unfolds", "unfolded", "unfolding", "展开")):
        return "unfold"
    if _contains_any(text, ("fold", "folds", "folded", "folding", "折叠")):
        return "fold"
    if _contains_any(text, ("straighten", "straightens", "straightened", "straightening", "拉直")):
        return "straighten"
    aliases = (
        ("transform", ("transform", "transforms", "transformed", "transforming", "变形")),
        ("shake", ("shake", "shakes", "shook", "shaken", "shaking", "摇晃", "摇动")), ("close", ("close", "closes", "closed", "closing", "关闭", "合上")),
        ("rotate", ("rotate", "rotates", "rotated", "rotating", "turn", "turns", "turned", "turning", "旋转", "转动")), ("drag", ("drag", "drags", "dragged", "dragging", "拖动")),
        ("pinch", ("pinch", "pinches", "pinched", "pinching", "捏")), ("squeeze", ("squeeze", "squeezes", "squeezed", "squeezing", "挤压")),
        ("poke", ("poke", "pokes", "poked", "poking", "戳")), ("press", ("press", "presses", "pressed", "pressing", "按压")),
        ("bend", ("bend", "bends", "bent", "bending", "弯曲")), ("articulate", ("articulate", "articulates", "articulated", "articulating", "关节")),
    )
    family = next((name for name, words in aliases if _contains_any(text, words)), "interaction_unverified")
    return "pull_open" if family == "interaction_unverified" and _contains_any(text, ("open", "打开")) else family


def _action_family(row: dict[str, Any]) -> str:
    explicit = str(row.get("action_family") or "").strip().lower()
    target_text = _row_text(row, ("target", "target_type", "target_description", "target_id", "target_instance_id"))
    response_text = _row_text(row, ("response_type", "response_description"))
    action_text = _row_text(row, ("action_family", "action_label", "action", "cause_description"))
    state_text = _row_text(row, ("object_state_before", "object_state_after", "pre_state", "post_state"))
    namespace_text = f"{action_text} {target_text} {response_text}"
    plush = _contains_any(namespace_text, PLUSH_WORDS)
    toy = _contains_any(namespace_text, TOY_WORDS)
    non_living_animal = _contains_any(target_text, NON_LIVING_ANIMAL_MARKERS)
    real_animal = _contains_any(namespace_text, (*REAL_ANIMAL_WORDS, *BREED_WORDS))
    generic_animal = _contains_any(namespace_text, ANIMAL_WORDS)
    namespace = "plush" if plush else "toy" if toy else "animal" if real_animal and not non_living_animal else "animal" if generic_animal and not non_living_animal else None
    if explicit in CANONICAL_FAMILIES and not (
        explicit in ACTION_FAMILIES and explicit not in {"gesture_prompt", "offer_feed"} and namespace is not None
    ):
        return explicit

    if _contains_any(action_text, ("pump", "pumps", "pumped", "pumping")) and (
        _contains_any(action_text, ("press", "presses", "pressed", "pressing"))
        or _contains_any(state_text, ("press", "presses", "pressed", "pressing"))
    ):
        return "press"

    base = _base_action_family(action_text)
    # Response text signals target state/namespace but must not create a new
    # action verb, for example an "open hand" response is not pull_open.
    context_text = f"{action_text} {target_text}"
    if base == "interaction_unverified":
        base = _base_action_family(context_text)
    semantic_text = f"{action_text} {response_text}"
    if namespace == "animal":
        if _contains_any(semantic_text, ("paw", "爪", "抬爪")):
            return "animal_paw"
        if _contains_any(action_text, ("pet", "stroke", "scratch", "rub", "抚摸", "抓挠", "摩擦", "摸", "触摸")):
            return "animal_contact"
        if _contains_any(action_text, ("offer open hand", "offer open palm", "open hand", "open palm")):
            return "gesture_prompt"
        if _contains_any(semantic_text, ("offer", "feed", "喂", "投喂")):
            return "animal_feed"
        if _contains_any(semantic_text, ("eat", "吃")):
            return "animal_eat"
        if _contains_any(semantic_text, ("approach", "靠近")):
            return "animal_approach"
        if _contains_any(semantic_text, ("respond", "response", "retreat", "reaction", "回应", "反应", "后退")):
            return "animal_response"
        if _contains_any(semantic_text, ("pet", "stroke", "scratch", "rub", "抚摸", "抓挠", "摩擦", "摸", "触摸")):
            return "animal_contact"
        if _contains_any(semantic_text, ("prompt", "present palm", "beckon", "gesture", "提示", "伸手", "招手", "手掌")):
            return "gesture_prompt"
        return "animal_contact" if base == "interaction_unverified" else f"animal_{base}"
    if namespace == "plush":
        return "plush_contact" if base == "interaction_unverified" else f"plush_{base}"
    if namespace == "toy":
        return "toy_contact" if base == "interaction_unverified" else f"toy_{base}"
    return base


def _is_animal_or_plush_candidate(row: dict[str, Any]) -> bool:
    family = str(row.get("action_family") or "")
    if family.startswith(("animal_", "plush_", "toy_")):
        return True
    text = " ".join(
        str(row.get(key) or "")
        for key in ("target", "target_type", "target_description", "target_id", "target_instance_id", "response_type", "response_description", "action_label")
    ).lower()
    return any(word in text for word in (*ANIMAL_WORDS, *PLUSH_WORDS, *TOY_WORDS))


def _window(row: dict[str, Any], core: tuple[float, float], duration: float, context_sec: float) -> tuple[float, float] | None:
    if core[1] - core[0] > MAX_WINDOW_SEC + 1e-9 or duration < MIN_WINDOW_SEC:
        return None
    start, end = _float(row.get("pre_start_sec_abs")), _float(row.get("post_end_sec_abs"))
    source_start = _float(row.get("source_start_sec"))
    if start is None and source_start is not None and (relative := _float(row.get("pre_start_sec_rel"))) is not None:
        start = source_start + relative
    if end is None and source_start is not None and (relative := _float(row.get("post_end_sec_rel"))) is not None:
        end = source_start + relative
    start = max(0.0, core[0] - context_sec) if start is None else max(0.0, start)
    end = min(duration, core[1] + context_sec) if end is None else min(duration, end)
    start, end = min(start, core[0]), max(end, core[1])
    if end - start > MAX_WINDOW_SEC:
        margin = MAX_WINDOW_SEC - (core[1] - core[0])
        start = max(0.0, core[0] - margin / 2)
        end = min(duration, start + MAX_WINDOW_SEC)
        start = max(0.0, end - MAX_WINDOW_SEC)
    if end - start < MIN_WINDOW_SEC:
        # Expand around the core while clamping to the source.  The ordering
        # below guarantees that clamping at either edge never drops the core.
        start = max(0.0, min(start, core[0], duration - MIN_WINDOW_SEC))
        end = min(duration, max(end, core[1], start + MIN_WINDOW_SEC))
        if end < core[1]:
            end = core[1]
            start = max(0.0, end - MIN_WINDOW_SEC)
        if start > core[0]:
            start = core[0]
            end = min(duration, start + MIN_WINDOW_SEC)
    if not MIN_WINDOW_SEC - 1e-6 <= end - start <= MAX_WINDOW_SEC + 1e-6:
        return None
    return _round(start), _round(end)


def _phase_fields(window: tuple[float, float], core: tuple[float, float]) -> dict[str, float]:
    action_start, action_end = _round(core[0] - window[0]), _round(core[1] - window[0])
    duration = _round(window[1] - window[0])
    return {
        "pre_start_sec_rel": 0.0, "pre_end_sec_rel": action_start,
        "action_start_sec_rel": action_start, "action_end_sec_rel": action_end,
        "response_start_sec_rel": action_end, "response_end_sec_rel": action_end,
        "post_start_sec_rel": action_end, "post_end_sec_rel": duration,
    }


def _semantic_identity(row: dict[str, Any], family: str) -> tuple[str, ...]:
    fields = ("target_id", "target_instance_id", "target_type", "action_direction", "direction", "action_round")
    placeholders = {"", "none", "null", "unknown", "unverified", "unspecified", "n/a", "na", "nan"}
    def normalize(value: Any) -> str:
        text = str(value or "").strip().lower()
        return "<missing>" if text in placeholders else text
    return (family, *(normalize(row.get(field)) for field in fields))


def _candidate_id(source_sha: str, core: tuple[float, float], identity: tuple[str, ...]) -> str:
    token = f"{source_sha}|{core[0]:.6f}|{core[1]:.6f}|{'|'.join(identity)}".encode("utf-8")
    return f"semantic_review_{hashlib.sha256(token).hexdigest()[:20]}"


def _block_reason(row: dict[str, Any]) -> str | None:
    if row.get("navigation_only") is True:
        return "navigation_only_evidence_row"
    if row.get("export_eligible") is False or row.get("exportable") is False:
        return "evidence_marked_non_exportable"
    if row.get("requires_manual_action_segmentation") is True:
        return "requires_manual_action_segmentation"
    return None


def _pending_action(
    row: dict[str, Any], kind: str, source: str, sha: str, info: dict[str, Any], reason: str
) -> dict[str, Any]:
    row_id = _row_id(row, kind)
    candidate = {
        "schema_version": 1, "policy_id": POLICY_ID, "candidate_id": f"semantic_pending_{row_id}", "clip_id": f"semantic_pending_{row_id}",
        "source_rel": source, "source_name": info["source_name"], "source_sha256": sha,
        "review_status": "pending_user_review", "pending_user_review": True, "feedback_status": "unverified",
        "semantic_feedback_verified": False, "navigation_only": bool(row.get("navigation_only")),
        "exportable": False, "export_eligible": False, "export_block_reason": reason,
        "proposal_origin": "explicit_semantic_split_decision" if kind == "decision" else f"{kind}_evidence_recut",
        "action_family": _action_family(row), "action_label": str(row.get("action_label") or row.get("action") or "unresolved_action"),
        "presence_parent_ids": [], "v3_parent_ids": [], "highrecall_parent_ids": [], "decision_parent_ids": [],
        "lineage_parent_ids": [row_id], "pending_review_flags": {"manual_review_required": True, "action_core_rejected": True},
    }
    candidate[{"v3": "v3_parent_ids", "highrecall": "highrecall_parent_ids", "decision": "decision_parent_ids"}[kind]] = [row_id]
    candidate.update({field: row[field] for field in SEMANTIC_FIELDS if field in row})
    return candidate


def generate_candidates(
    review_index: dict[tuple[str, str], dict[str, list[dict[str, Any]]]], inventory_rows: list[dict[str, Any]],
    *, decisions_rows: list[dict[str, Any]] | None = None, context_sec: float = DEFAULT_CONTEXT_SEC,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate valid and pending candidates; no row is training-approved."""
    if not math.isfinite(context_sec) or context_sec < 0:
        raise ValueError("context_sec must be finite and non-negative")
    inventory = _inventory(inventory_rows)
    candidates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for entry in review_index.values():
        unresolved.extend(entry["invalid"])
    decisions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions_rows or []:
        source, sha = _source_rel(row), _source_sha(row)
        if source and sha:
            decisions[(source, sha)].append(row)
        else:
            unresolved.append({"candidate_id": f"semantic_pending_{_row_id(row, 'decision')}", "row_id": _row_id(row, "decision"),
                               "kind": "decision", "exportable": False, "export_eligible": False,
                               "export_block_reason": "invalid_source_identity", "pending_user_review": True,
                               "review_status": "pending_user_review"})
    slow_count = 0
    long_interval_splits = 0
    for key in sorted(set(review_index) | set(decisions)):
        if key not in inventory:
            unresolved.append({"candidate_id": f"semantic_pending_missing_{hashlib.sha256(repr(key).encode()).hexdigest()[:16]}",
                               "source_rel": key[0], "source_sha256": key[1], "exportable": False, "export_eligible": False,
                               "export_block_reason": "source_missing_from_inventory", "pending_user_review": True,
                               "review_status": "pending_user_review"})
            continue
        source, sha = key
        info = inventory[key]
        entry = review_index.get(key, {"presence": [], "v3": [], "highrecall": [], "invalid": []})
        presence = sorted(((row, _interval(row, presence=True)) for row in entry["presence"]), key=lambda item: _row_id(item[0], "presence"))
        presence = [(row, interval) for row, interval in presence if interval]
        evidence = [("decision", row) for row in decisions.get(key, [])] + [("v3", row) for row in entry["v3"]] + [("highrecall", row) for row in entry["highrecall"]]
        evidence.sort(key=lambda item: (item[0], _row_id(item[1], item[0])))
        covered_presence: set[str] = set()
        for kind, row in evidence:
            row_id = _row_id(row, kind)
            blocked = _block_reason(row)
            core = _core(row)
            if blocked:
                pending = _pending_action(row, kind, source, sha, info, blocked)
                candidates.append(pending)
                unresolved.append({"row_id": row_id, "kind": kind, "reason": blocked})
                continue
            if not core:
                candidates.append(_pending_action(row, kind, source, sha, info, "invalid_core_boundary"))
                unresolved.append({"row_id": row_id, "kind": kind, "reason": "invalid_core_boundary"})
                continue
            if not (0 <= core[0] < core[1] <= info["duration"] + 1e-6):
                candidates.append(_pending_action(row, kind, source, sha, info, "core_outside_source"))
                unresolved.append({"row_id": row_id, "kind": kind, "reason": "core_outside_source"})
                continue
            parent_ids = sorted(_row_id(parent, "presence") for parent, interval in presence if _overlaps(core, interval))
            if not parent_ids:
                candidates.append(_pending_action(row, kind, source, sha, info, "core_does_not_intersect_presence"))
                unresolved.append({"row_id": row_id, "kind": kind, "reason": "core_does_not_intersect_presence"})
                continue
            window = _window(row, core, info["duration"], context_sec)
            if not window:
                candidates.append(_pending_action(row, kind, source, sha, info, "core_or_window_outside_2_to_20_sec"))
                unresolved.append({"row_id": row_id, "kind": kind, "reason": "core_or_window_outside_2_to_20_sec"})
                continue
            covered_presence.update(parent_ids)

            def supporting_ids(rows: list[dict[str, Any]], prefix: str) -> list[str]:
                return sorted(_row_id(parent, prefix) for parent in rows if (interval := _interval(parent)) and _overlaps(core, interval))

            family = _action_family(row)
            identity = _semantic_identity(row, family)
            lineage = {
                "presence_parent_ids": parent_ids,
                "v3_parent_ids": supporting_ids(entry["v3"], "v3"),
                "highrecall_parent_ids": supporting_ids(entry["highrecall"], "highrecall"),
                "decision_parent_ids": supporting_ids(decisions.get(key, []), "decision"),
            }
            lineage_key = {"v3": "v3_parent_ids", "highrecall": "highrecall_parent_ids", "decision": "decision_parent_ids"}[kind]
            if row_id not in lineage[lineage_key]:
                lineage[lineage_key].append(row_id)
                lineage[lineage_key].sort()
            candidate = {
                "schema_version": 1, "policy_id": POLICY_ID,
                "candidate_id": _candidate_id(sha, core, identity), "clip_id": _candidate_id(sha, core, identity),
                "source_rel": source, "source_name": info["source_name"], "source_sha256": sha, "source_duration_sec": _round(info["duration"]),
                "source_start_sec": window[0], "source_end_sec": window[1], "duration_sec": _round(window[1] - window[0]),
                "core_start_sec_abs": _round(core[0]), "core_end_sec_abs": _round(core[1]), "action_family": family,
                "action_label": str(row.get("action_label") or row.get("action") or family), **lineage,
                "lineage_parent_ids": sorted({parent for values in lineage.values() for parent in values}),
                "proposal_parent_id": row_id,
                "proposal_origin": "explicit_semantic_split_decision" if kind == "decision" else f"{kind}_evidence_recut",
                "evidence_kind": kind, "review_status": "pending_user_review", "pending_user_review": True,
                "feedback_status": "unverified", "semantic_feedback_verified": False, "navigation_only": False,
                "phase_boundaries_status": "provisional_semantic_core_pending_manual_review",
                "pending_review_flags": {"semantic_boundary_provisional": True, "manual_review_required": True},
                "exportable": True, "export_eligible": True, "export_block_reason": None,
                "replaces_clip_ids": sorted(str(value) for value in row.get("replaces_clip_ids", []) or []),
                "supersedes_clip_ids": sorted(str(value) for value in row.get("supersedes_clip_ids", []) or []),
                **_phase_fields(window, core),
            }
            candidate.update({field: row[field] for field in SEMANTIC_FIELDS if field in row})
            if kind == "decision":
                candidate["decision_id"] = str(row.get("decision_id") or row_id)
                candidate["possible_alternate_decision_ids"] = sorted(
                    str(value) for value in row.get("possible_alternate_decision_ids", []) or []
                )
            candidate.setdefault("camera_motion", "unassessed_pending_user_review")
            candidates.append(candidate)
            if kind == "decision" and any(interval[1] - interval[0] > core[1] - core[0] + 1e-6 for _, interval in presence):
                long_interval_splits += 1
        for row, interval in presence:
            row_id = _row_id(row, "presence")
            if row_id in covered_presence:
                continue
            slow = bool(row.get("slow_motion")) or _float(row.get("motion_score")) == 0.0
            slow_count += int(slow)
            candidates.append({
                "schema_version": 1, "policy_id": POLICY_ID, "candidate_id": _candidate_id(sha, interval, ("presence_navigation", row_id)),
                "clip_id": _candidate_id(sha, interval, ("presence_navigation", row_id)), "source_rel": source,
                "source_name": info["source_name"], "source_sha256": sha, "source_duration_sec": _round(info["duration"]),
                "source_start_sec": _round(interval[0]), "source_end_sec": _round(interval[1]), "duration_sec": _round(interval[1] - interval[0]),
                "core_start_sec_abs": None, "core_end_sec_abs": None, "action_family": "presence_navigation",
                "action_label": "hand_presence_navigation_only", "presence_parent_ids": [row_id], "v3_parent_ids": [],
                "highrecall_parent_ids": [], "decision_parent_ids": [], "lineage_parent_ids": [row_id],
                "proposal_origin": "presence_navigation_only", "review_status": "pending_user_review", "pending_user_review": True,
                "feedback_status": "unverified", "semantic_feedback_verified": False, "navigation_only": True,
                "slow_motion_retained": slow, "phase_boundaries_status": "no_semantic_core_navigation_only",
                "pending_review_flags": {"navigation_only_presence": True, "semantic_boundary_missing": True, "manual_review_required": True},
                "exportable": False, "export_eligible": False, "export_block_reason": "navigation_only_presence_row",
            })
    report = {
        "schema_version": 1, "policy_id": POLICY_ID, "generated_candidate_count": len(candidates),
        "unresolved_count": len(unresolved), "unresolved": unresolved, "slow_motion_retained_count": slow_count,
        "long_interval_split_count": long_interval_splits,
        "hand_detection_run": False, "training_run": False,
    }
    return candidates, report


def _dedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_sha256"), _round(float(row["core_start_sec_abs"])), _round(float(row["core_end_sec_abs"])),
        *_semantic_identity(row, str(row.get("action_family") or "interaction_unverified")),
    )


def _export_window_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    start, end = _float(row.get("source_start_sec")), _float(row.get("source_end_sec"))
    if start is None or end is None or start >= end:
        return None
    return (row.get("source_sha256"), row.get("source_rel"), _round(start), _round(end))


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    origin = str(row.get("proposal_origin") or "")
    family = str(row.get("action_family") or "")
    if origin == "explicit_semantic_split_decision":
        tier = 0
    elif origin.startswith("v3"):
        tier = 2 if family == "interaction_unverified" else 1
    else:
        tier = 3
    special = family.startswith(("animal_", "plush_", "toy_")) or family in {"offer_feed", "gesture_prompt"}
    before, after = row.get("object_state_before") or row.get("pre_state"), row.get("object_state_after") or row.get("post_state")
    paired_state_change = bool(before and after and before != after)
    support = len(set(row.get("v3_parent_ids", []) + row.get("highrecall_parent_ids", []) + row.get("decision_parent_ids", [])))
    return (tier, -int(special), -int(paired_state_change), -support, row.get("source_rel", ""),
            row.get("core_start_sec_abs", -1), row.get("core_end_sec_abs", -1), row.get("candidate_id", ""))


def reconcile_candidates(rows: list[dict[str, Any]], target_count: int = 2050) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select at most target unique valid actions; retain navigation/overflow as pending."""
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    pending: list[dict[str, Any]] = []
    retired: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    decision_rows = [row for row in rows if row.get("proposal_origin") == "explicit_semantic_split_decision"]
    aliases_by_decision: dict[str, set[str]] = {}
    decision_by_alias: dict[str, str] = {}
    for row in decision_rows:
        canonical = str(row.get("candidate_id"))
        aliases = {canonical}
        if row.get("proposal_parent_id"):
            aliases.add(str(row["proposal_parent_id"]))
        if row.get("decision_id"):
            aliases.add(str(row["decision_id"]))
        aliases_by_decision[canonical] = aliases
        for alias in aliases:
            decision_by_alias[alias] = canonical

    # Resolve each raw replacement reference through generated and original
    # aliases before evaluating self references or decision-to-decision cycles.
    refs_by_decision: dict[str, list[tuple[str, str | None]]] = {}
    for row in decision_rows:
        canonical = str(row.get("candidate_id"))
        refs_by_decision[canonical] = [
            (str(ref), decision_by_alias.get(str(ref)))
            for ref in [*(row.get("replaces_clip_ids", []) or []), *(row.get("supersedes_clip_ids", []) or [])]
        ]
    decision_edges = {
        (source, target)
        for source, refs in refs_by_decision.items()
        for _, target in refs
        if target is not None and target != source
    }
    replacement_aliases: set[str] = set()
    for source, refs in refs_by_decision.items():
        for raw_ref, target in refs:
            if target == source:
                continue
            # A<->B is an ambiguous cycle, so neither side retires the other.
            if target is not None and (target, source) in decision_edges:
                continue
            replacement_aliases.add(raw_ref)
            if target is not None:
                replacement_aliases.update(aliases_by_decision[target])
    composite_comparison_aliases: set[str] = set()
    decision_by_candidate_id = {str(row["candidate_id"]): row for row in decision_rows}
    for row in decision_rows:
        alternate_rows = []
        for alternate in row.get("possible_alternate_decision_ids", []) or []:
            canonical = decision_by_alias.get(str(alternate))
            if canonical and canonical in decision_by_candidate_id:
                alternate_rows.append(decision_by_candidate_id[canonical])
        core = _core(row)
        if len({candidate["candidate_id"] for candidate in alternate_rows}) < 2 or not core:
            continue
        alternate_cores = [_core(candidate) for candidate in alternate_rows]
        if all(
            alternate_core is not None
            and row.get("source_sha256") == candidate.get("source_sha256")
            and core[0] <= alternate_core[0] < alternate_core[1] <= core[1]
            for candidate, alternate_core in zip(alternate_rows, alternate_cores)
        ):
            composite_comparison_aliases.update(aliases_by_decision[str(row["candidate_id"])])
    for row in rows:
        valid = row.get("export_eligible", True) is True and row.get("exportable", True) is True and not row.get("navigation_only")
        if not valid:
            pending.append(row)
            continue
        row_identity = {str(row.get("proposal_parent_id")), str(row.get("candidate_id"))}
        if row_identity.intersection(composite_comparison_aliases):
            pending.append({
                **row,
                "selected_for_export": False,
                "exportable": False,
                "export_eligible": False,
                "export_block_reason": "superseded_by_explicit_split_decisions",
            })
            continue
        if row_identity.intersection(replacement_aliases):
            retired.append({**row, "retirement_reason": "replaced_by_semantic_decision"})
        else:
            action_rows.append(row)
    unique: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen_windows: dict[tuple[Any, ...], dict[str, Any]] = {}
    partial_overlap_count = 0
    source_cores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in sorted(action_rows, key=_rank):
        key = _dedup_key(row)
        if key in seen:
            retired.append({**row, "retirement_reason": "exact_core_boundary_duplicate", "retired_by_candidate_id": seen[key]["candidate_id"]})
            continue
        window_key = _export_window_key(row)
        if window_key is not None and window_key in seen_windows:
            retired.append({
                **row,
                "retirement_reason": "exact_export_window_duplicate",
                "retired_by_candidate_id": seen_windows[window_key]["candidate_id"],
            })
            continue
        interval = (float(row["core_start_sec_abs"]), float(row["core_end_sec_abs"]))
        if any(_overlaps(interval, other) for other in source_cores[row["source_rel"]]):
            partial_overlap_count += 1
        source_cores[row["source_rel"]].append(interval)
        seen[key] = row
        if window_key is not None:
            seen_windows[window_key] = row
        unique.append(row)
    eligible_source_coverage_count = len({row["source_rel"] for row in unique})
    source_first: list[dict[str, Any]] = []
    represented_sources: set[str] = set()
    for row in unique:
        source = row["source_rel"]
        if source not in represented_sources:
            source_first.append(row)
            represented_sources.add(source)
    if target_count < len(source_first):
        selected = source_first[:target_count]
    else:
        selected_ids = {row["candidate_id"] for row in source_first}
        selected = source_first + [
            row for row in unique if row["candidate_id"] not in selected_ids
        ][:target_count - len(source_first)]
    selected.sort(key=_rank)
    selected_ids = {row["candidate_id"] for row in selected}
    for row in unique:
        if row["candidate_id"] in selected_ids:
            continue
        pending.append({**row, "selected_for_export": False, "exportable": False, "export_eligible": False,
                        "export_block_reason": "overflow_not_selected"})
    for row in selected:
        row["selected_for_export"] = True
        row["exportable"] = True
        row["export_eligible"] = True
    report = {
        "generated_candidate_count": len(rows), "unique_export_eligible_count": len(unique), "selected_candidate_count": len(selected),
        "candidate_count": len(selected), "pending_count": len(pending), "retired_count": len(retired),
        "exact_core_boundary_deduplicated_count": sum(row.get("retirement_reason") == "exact_core_boundary_duplicate" for row in retired),
        "exact_export_window_deduplicated_count": sum(row.get("retirement_reason") == "exact_export_window_duplicate" for row in retired),
        "exact_boundary_deduplicated_count": sum(
            row.get("retirement_reason") in {"exact_core_boundary_duplicate", "exact_export_window_duplicate"}
            for row in retired
        ),
        "exact_core_duplicate_groups_after_normalization": sum(row.get("retirement_reason") == "exact_core_boundary_duplicate" for row in retired),
        "replacement_lineage_count": sum(row.get("retirement_reason") == "replaced_by_semantic_decision" for row in retired),
        "partial_overlap_retained_count": partial_overlap_count, "target_count": target_count,
        "target_met": len(selected) == target_count, "deficit_count": max(0, target_count - len(selected)),
        "over_target_count": max(0, len(unique) - target_count), "replacement_needed_count": max(0, target_count - len(selected)),
        "eligible_source_coverage_count": eligible_source_coverage_count,
        "selected_source_coverage_count": len({row["source_rel"] for row in selected}),
        "selected_source_coverage_deficit_count": max(
            0, eligible_source_coverage_count - len({row["source_rel"] for row in selected})
        ),
        "pending_candidate_ids": [str(row.get("candidate_id")) for row in pending], "_pending_rows": pending,
    }
    return selected, retired, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--presence", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--high-recall", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=2050)
    parser.add_argument("--context-sec", type=float, default=DEFAULT_CONTEXT_SEC)
    args = parser.parse_args(argv)
    decision_path = args.decisions or (args.work_root / "manifests" / "semantic_split_decisions.jsonl")
    decisions = load_jsonl(decision_path) if decision_path.exists() else []
    index = build_review_index(load_jsonl(args.presence), load_jsonl(args.v3), load_jsonl(args.high_recall))
    generated, generated_report = generate_candidates(index, load_jsonl(args.inventory), decisions_rows=decisions, context_sec=args.context_sec)
    selected, retired, selection_report = reconcile_candidates(generated, args.target_count)
    pending = selection_report.pop("_pending_rows")
    selection_report.pop("pending_candidate_ids")
    manifests = args.work_root / "manifests"
    write_jsonl(manifests / "semantic_review_candidates_2050.jsonl", selected)
    write_jsonl(manifests / "semantic_review_retired.jsonl", retired)
    write_jsonl(manifests / "semantic_review_pending.jsonl", pending)
    report = {
        "schema_version": 1, "policy_id": POLICY_ID, **selection_report,
        "long_interval_split_count": generated_report.get("long_interval_split_count", 0),
        "source_coverage_count": selection_report["selected_source_coverage_count"],
        "action_family_counts": dict(sorted(Counter(row.get("action_family", "unresolved") for row in selected).items())),
        "animal_or_plush_candidate_count": sum(_is_animal_or_plush_candidate(row) for row in selected),
        "slow_motion_retained_count": sum(bool(row.get("slow_motion_retained")) for row in pending),
        "navigation_pending_count": sum(bool(row.get("navigation_only")) for row in pending),
        "overflow_pending_count": sum(row.get("export_block_reason") == "overflow_not_selected" for row in pending),
        "unresolved_count": sum(
            bool(row.get("export_block_reason"))
            and row.get("export_block_reason") not in {"navigation_only_presence_row", "overflow_not_selected"}
            for row in pending
        ),
        "invalid_core_or_phase_count": sum(
            not row.get("navigation_only") and (
                row.get("export_block_reason") in {"invalid_core_boundary", "core_outside_source", "core_or_window_outside_2_to_20_sec"}
                or any(field not in row for field in ("pre_start_sec_rel", "action_start_sec_rel", "action_end_sec_rel", "post_end_sec_rel"))
            )
            for row in pending
        ) + sum(
            any(field not in row for field in ("pre_start_sec_rel", "action_start_sec_rel", "action_end_sec_rel", "post_end_sec_rel"))
            for row in selected
        ),
        "hand_detection_run": False, "training_run": False, "decisions_path": str(decision_path) if decision_path.exists() else None,
    }
    reports = args.work_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "semantic_review_candidates_2050.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated_count": generated_report["generated_candidate_count"], "candidate_count": len(selected), "pending_count": len(pending), "retired_count": len(retired), "target_met": report["target_met"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
