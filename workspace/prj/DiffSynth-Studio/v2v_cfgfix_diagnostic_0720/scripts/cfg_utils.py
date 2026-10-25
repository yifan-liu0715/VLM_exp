from __future__ import annotations


def apply_classifier_free_guidance(conditional, unconditional, guidance_scale: float):
    """Combine conditional and unconditional predictions using standard CFG."""
    if guidance_scale <= 1.0:
        raise ValueError("guidance_scale must be greater than 1 for CFG")
    return unconditional + guidance_scale * (conditional - unconditional)

