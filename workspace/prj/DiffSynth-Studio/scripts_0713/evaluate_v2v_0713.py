#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
PAIRS = PROJECT / "data_0713/v2v_quick_0713/v2v_pairs_0713.csv"
RESULTS = PROJECT / "validation_results_0713"
METRICS = RESULTS / "metrics_0713"


def frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    output = []
    while len(output) < 33:
        ok, frame = cap.read()
        if not ok:
            break
        output.append(cv2.resize(frame, (576, 320), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(output) != 33:
        raise RuntimeError(f"Expected 33 frames, got {len(output)}: {path}")
    return output


def structural_similarity(gray_a, gray_b):
    """Standard local-window SSIM implemented with OpenCV Gaussian filters."""
    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def score_pair(candidate, target):
    ssim, psnr, sharpness = [], [], []
    for cand, ref in zip(candidate, target):
        gray_c = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ssim.append(structural_similarity(gray_c, gray_r))
        mse = np.mean((cand.astype(np.float32) - ref.astype(np.float32)) ** 2)
        psnr.append(100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse)))
        sharpness.append(cv2.Laplacian(gray_c, cv2.CV_64F).var())
    return {"frame_ssim": float(np.mean(ssim)), "psnr_db": float(np.mean(psnr)), "sharpness": float(np.mean(sharpness))}


def clip_scorer():
    try:
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(Path.home() / ".cache/clip"))
        model.eval()
        return model, preprocess, device, torch, clip
    except Exception as exc:
        print(f"[warning] CLIP unavailable: {exc}")
        return None


def semantic_score(video_frames, prompt, bundle):
    if bundle is None:
        return None
    from PIL import Image
    model, preprocess, device, torch, clip = bundle
    indices = np.linspace(0, len(video_frames) - 1, 8).round().astype(int)
    images = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2RGB))) for i in indices]).to(device)
    tokens = clip.tokenize([prompt], truncate=True).to(device)
    with torch.inference_mode():
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return float((image_features @ text_features.T).mean().item())


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    with PAIRS.open("r", encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    clip_bundle = clip_scorer()
    rows = []
    for pair in pairs:
        target = frames(Path(pair["target_video"]))
        for variant, path in [
            ("target", Path(pair["target_video"])),
            ("source", Path(pair["source_video"])),
            ("base", RESULTS / f"{pair['sample_id']}_base.mp4"),
            ("lora", RESULTS / f"{pair['sample_id']}_lora.mp4"),
        ]:
            if not path.is_file():
                raise FileNotFoundError(path)
            video = frames(path)
            metric = score_pair(video, target)
            metric["clip_prompt_similarity"] = semantic_score(video, pair["prompt"], clip_bundle)
            rows.append({"sample_id": pair["sample_id"], "variant": variant, "video": str(path), **metric})
    csv_path = METRICS / "v2v_metrics_0713.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for variant in ("target", "source", "base", "lora"):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate.append({
            "variant": variant, "samples": len(selected),
            "frame_ssim": float(np.mean([row["frame_ssim"] for row in selected])),
            "psnr_db": float(np.mean([row["psnr_db"] for row in selected])),
            "sharpness": float(np.mean([row["sharpness"] for row in selected])),
            "clip_prompt_similarity": None if any(row["clip_prompt_similarity"] is None for row in selected) else float(np.mean([row["clip_prompt_similarity"] for row in selected])),
        })
    (METRICS / "v2v_metrics_aggregate_0713.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    report = ["# V2V quick validation 0713", "", "Target is the standardized original video. Source is the recorded deterministic degraded video.", "", "| Variant | Frame SSIM ↑ | PSNR dB ↑ | Sharpness ↑ | CLIP prompt similarity ↑ |", "|---|---:|---:|---:|---:|"]
    for row in aggregate:
        clip_value = "N/A" if row["clip_prompt_similarity"] is None else f"{row['clip_prompt_similarity']:.4f}"
        report.append(f"| {row['variant']} | {row['frame_ssim']:.4f} | {row['psnr_db']:.2f} | {row['sharpness']:.2f} | {clip_value} |")
    report += ["", "Interpretation: compare base and lora only under identical source video, prompt, seed, denoising strength, and inference settings. This is a native input_video V2V restoration smoke test, not a paired source-conditioned architecture."]
    (METRICS / "V2V_RESULT_0713.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
