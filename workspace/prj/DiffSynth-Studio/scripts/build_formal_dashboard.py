#!/usr/bin/env python3
"""Build a standalone HTML dashboard for formal checkpoint validation."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_RESULTS = PROJECT_ROOT / "validation_results/formal_val_v2"
DEFAULT_SUBSET = PROJECT_ROOT / "data_splits/source_disjoint_v2/validation_subset_15.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def metric_bar(value: float, low: float = 0.0, high: float = 1.0) -> str:
    width = max(0.0, min(100.0, 100.0 * (value - low) / max(high - low, 1e-12)))
    return (
        '<div class="metric"><span>{:.3f}</span>'
        '<i><b style="width:{:.1f}%"></b></i></div>'
    ).format(value, width)


def ranking_rows(rows: list[dict]) -> str:
    rendered = []
    for row in rows:
        badges = []
        badges.append(
            '<em class="badge good">guardrail</em>'
            if row["eligible_guardrails"]
            else '<em class="badge bad">failed</em>'
        )
        if row.get("eligible_guardrails_ci"):
            badges.append('<em class="badge robust">95% CI robust</em>')
        if row["pareto_front"]:
            badges.append('<em class="badge pareto">Pareto</em>')
        rendered.append(
            "<tr>"
            f"<td>{row['rank']}</td>"
            f"<td><code>{html.escape(Path(row['checkpoint']).name)}</code><div>{''.join(badges)}</div></td>"
            f"<td>{metric_bar(row['sharpness_retention'], 0.75, 1.05)}</td>"
            f"<td>{metric_bar(row['motion_retention'], 0.50, 1.20)}</td>"
            f"<td>{metric_bar(row['warp_error_reduction'], -0.30, 0.40)}</td>"
            f"<td>{metric_bar(row['target_motion_mae_reduction'], -0.30, 0.40)}</td>"
            f"<td>{row['reference_ssim_delta']:+.4f}</td>"
            f"<td><strong>{row['exploratory_selection_score']:.3f}</strong></td>"
            "</tr>"
        )
    return "\n".join(rendered)


def subgroup_section(results: Path) -> str:
    path = results / "subgroup_metrics.json"
    if not path.is_file():
        return ""
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    body = []
    for row in rows:
        gate = (
            '<em class="badge good">pass</em>'
            if row["passes_group_guardrails"]
            else '<em class="badge bad">fail</em>'
        )
        body.append(
            "<tr>"
            f"<td><code>{html.escape(row['variant'])}</code></td>"
            f"<td>{html.escape(row['group_type'])}={html.escape(str(row['group']))}</td>"
            f"<td>{row['samples']}</td>"
            f"<td>{row['sharpness_retention']:.3f}</td>"
            f"<td>{row['motion_retention']:.3f}</td>"
            f"<td>{row['warp_error_reduction']:.3f}</td>"
            f"<td>{row['target_motion_mae_reduction']:.3f}</td>"
            f"<td>{row['reference_ssim_delta']:+.4f}</td>"
            f"<td>{gate}</td>"
            "</tr>"
        )
    return (
        "<h2>Subgroup veto audit</h2>"
        "<p>分别检查分辨率桶和内容域；子组失败是模型选择的警告，而不是由整体均值抵消。</p>"
        '<div class="table-wrap"><table><thead><tr><th>Checkpoint</th><th>Group</th><th>N</th>'
        "<th>Sharpness</th><th>Motion</th><th>Warp reduction</th>"
        "<th>Target MAE reduction</th><th>Ref SSIM Δ</th><th>Gate</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def dover_section(results: Path) -> str:
    path = results / "dover_mobile/dover_aggregate.csv"
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    body = "".join(
        "<tr>"
        f"<td><code>{html.escape(row['variant'])}</code></td>"
        f"<td>{int(float(row['samples']))}</td>"
        f"<td>{float(row['mean_dover_technical']):.4f}</td>"
        f"<td>{float(row['mean_dover_aesthetic']):.4f}</td>"
        f"<td><strong>{float(row['mean_dover_fused']):.4f}</strong></td>"
        "</tr>"
        for row in rows
    )
    return (
        "<h2>DOVER-Mobile learned quality</h2>"
        "<p>官方 DOVER-Mobile 的技术、审美和融合分数；仅作次级证据，盲评优先。</p>"
        '<div class="table-wrap"><table><thead><tr><th>Variant</th><th>N</th>'
        "<th>Technical</th><th>Aesthetic</th><th>Fused</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def main() -> None:
    args = parse_args()
    results = args.results_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results / "formal_validation_dashboard.html"
    )
    ranking_path = results / "checkpoint_ranking.json"
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Checkpoint ranking is not ready: {ranking_path}")
    ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    rows = ranking_payload["ranking"]
    subset = json.loads(args.subset.read_text(encoding="utf-8"))["samples"]
    if not rows or not subset:
        raise RuntimeError("Ranking and validation subset must be non-empty")

    client_payload = {
        "variants": [row["label"] for row in rows],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "bucket": sample["bucket"],
                "prompt": sample["caption"],
            }
            for sample in subset
        ],
    }
    escaped_guardrails = html.escape(json.dumps(ranking_payload["guardrails"], ensure_ascii=False))
    subgroup_html = subgroup_section(results)
    dover_html = dover_section(results)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan2.2 LoRA Formal Validation</title>
<style>
:root{{--bg:#0b1020;--panel:#151c31;--panel2:#1d2741;--text:#edf2ff;--muted:#9eaccb;--line:#2c395d;--accent:#7aa2ff;--green:#4ed6a8;--red:#ff7788;--gold:#ffc857}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#080c18,#101831 60%,#151b31);color:var(--text);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:36px 28px 80px}} h1{{font-size:34px;margin:0 0 5px}} h2{{margin-top:38px;font-size:22px}} p{{color:var(--muted)}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin:24px 0}} .card{{background:rgba(21,28,49,.92);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 32px #0003}}
.card strong{{display:block;font-size:27px;color:white}} .card span{{color:var(--muted)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--panel)}} table{{width:100%;border-collapse:collapse;min-width:1050px}} th,td{{padding:12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}} th{{position:sticky;top:0;background:#1b2540;color:#cbd7f4}} tr:hover{{background:#1a2440}}
code{{color:#b8ccff}} .badge{{display:inline-block;margin:5px 4px 0 0;padding:2px 7px;border-radius:20px;font-size:11px;font-style:normal}} .good{{background:#174c41;color:#7ff0ce}} .bad{{background:#552532;color:#ff9aaa}} .pareto{{background:#594518;color:#ffd66f}} .robust{{background:#263f6e;color:#a9c7ff}}
.metric{{min-width:120px}} .metric span{{display:block;font-variant-numeric:tabular-nums}} .metric i{{display:block;height:5px;background:#2b3656;border-radius:5px;overflow:hidden}} .metric b{{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--green))}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}} select{{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:9px;padding:9px 12px;max-width:100%}}
.prompt{{padding:13px 16px;border-left:3px solid var(--accent);background:#111a30;color:#c7d4ef;border-radius:5px;margin:12px 0 18px}}
.videos{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .video-card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}} .video-card h3{{margin:0 0 10px}} video{{display:block;width:100%;background:#05070d;border-radius:9px;aspect-ratio:576/320}}
.note{{border:1px solid #46547a;background:#141d35;border-radius:12px;padding:14px 16px}} @media(max-width:850px){{.summary{{grid-template-columns:1fr 1fr}}.videos{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}
</style>
</head>
<body><main>
<h1>Wan2.2-TI2V-5B LoRA 正式验证</h1>
<p>source-disjoint v2 · 576×320 · 33 frames · seed 42 · tiled · 相同生成条件</p>
<section class="summary">
 <div class="card"><strong>{len(rows)}</strong><span>checkpoints</span></div>
 <div class="card"><strong>{len(subset)}</strong><span>held-out samples</span></div>
 <div class="card"><strong>{sum(bool(row.get('eligible_guardrails_ci')) for row in rows)}</strong><span>95% CI robust</span></div>
 <div class="card"><strong>{sum(bool(row['pareto_front']) for row in rows)}</strong><span>Pareto candidates</span></div>
</section>
<h2>Checkpoint 排名</h2>
<p>数值分数仅用于探索性排序；guardrail、Pareto 状态和隐藏标签盲评优先。Guardrails: <code>{escaped_guardrails}</code></p>
<div class="table-wrap"><table><thead><tr><th>#</th><th>Checkpoint</th><th>Sharpness retention</th><th>Motion retention</th><th>Warp reduction</th><th>Target-motion MAE reduction</th><th>Ref SSIM Δ</th><th>Score</th></tr></thead><tbody>
{ranking_rows(rows)}
</tbody></table></div>
{subgroup_html}
{dover_html}
<h2>逐样本视频浏览</h2>
<div class="controls"><label>样本 <select id="sample"></select></label><label>Checkpoint <select id="variant"></select></label></div>
<div id="prompt" class="prompt"></div>
<div class="videos">
 <div class="video-card"><h3>Base</h3><video id="base" controls muted loop playsinline></video></div>
 <div class="video-card"><h3 id="variantTitle">LoRA</h3><video id="lora" controls muted loop playsinline></video></div>
</div>
<h2>解释边界</h2>
<div class="note">本页不把代理指标等同于最终质量。运动幅度下降可能是抖动减少，也可能是动作塌缩；必须联合目标运动误差、清晰度、时序稳定性及随机化盲评。验证集与训练集按源视频组隔离，但 15 个样本仍属于小样本研究，应报告置信区间并在独立 test split 上复核最终 checkpoint。</div>
</main>
<script>
const DATA={json.dumps(client_payload, ensure_ascii=False)};
const sampleSelect=document.getElementById('sample'); const variantSelect=document.getElementById('variant');
DATA.samples.forEach((s,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{s.sample_id}} (${{s.bucket}})`;sampleSelect.appendChild(o)}});
DATA.variants.forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;variantSelect.appendChild(o)}});
function refresh(){{const s=DATA.samples[Number(sampleSelect.value||0)];const v=variantSelect.value||DATA.variants[0];document.getElementById('prompt').textContent=s.prompt;document.getElementById('variantTitle').textContent=v;document.getElementById('base').src=`${{s.sample_id}}_base.mp4`;document.getElementById('lora').src=`${{s.sample_id}}_${{v}}.mp4`;}}
sampleSelect.addEventListener('change',refresh);variantSelect.addEventListener('change',refresh);refresh();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    print(f"Dashboard written to {output}")


if __name__ == "__main__":
    main()
