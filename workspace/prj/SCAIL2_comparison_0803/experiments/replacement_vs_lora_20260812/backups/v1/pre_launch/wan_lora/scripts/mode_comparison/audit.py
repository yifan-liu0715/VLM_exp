"""Pure validation helpers for model comparison artifacts."""

from __future__ import annotations

import re
from fractions import Fraction


FAIRNESS_FIELDS = (
    "prompt",
    "width",
    "height",
    "num_frames",
    "fps",
    "solver",
    "steps",
    "guidance_scale",
    "shift",
    "seed",
)


def normalize_probe(stream: dict) -> dict[str, object]:
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(str(stream["avg_frame_rate"]))),
        "frames": int(stream["nb_read_frames"]),
    }


def scan_log_errors(text: str) -> list[str]:
    patterns = {
        "traceback": r"\btraceback \(most recent call last\)",
        "cuda_oom": r"cuda out of memory|torch\.outofmemoryerror",
        "non_finite": r"non[- ]finite|\bnan\b|\binf(?:inity)?\b",
        "no_space": r"no space left on device",
        "segmentation_fault": r"segmentation fault|core dumped",
    }
    lowered = text.lower()
    return [name for name, pattern in patterns.items() if re.search(pattern, lowered)]


def parse_physical_gpu(text: str) -> str:
    matches = re.findall(r"physical_gpu\s*=\s*([0-9]+)", text)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one physical_gpu record, found {unique}")
    if unique[0] == "0":
        raise ValueError("physical GPU0 is forbidden")
    return unique[0]


def compare_fairness_fields(scail: dict, lora: dict) -> dict[str, dict[str, object]]:
    differences = {}
    for key in FAIRNESS_FIELDS:
        if key not in scail and key not in lora:
            continue
        if scail.get(key) != lora.get(key):
            differences[key] = {
                "scail2": scail.get(key),
                "wan_lora": lora.get(key),
            }
    return differences
