from __future__ import annotations
import argparse, json, re
from pathlib import Path


def strict_parse_json(raw: str):
    if not isinstance(raw, str):
        return None
    raw = re.sub(r'<think>.*?</think>', '', raw.strip(), flags=re.S).strip()
    candidates = []
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, flags=re.S | re.I)
    if fence:
        candidates.append(fence.group(1))
    if raw.startswith('{'):
        try:
            obj, end = json.JSONDecoder().raw_decode(raw)
            candidates.append(raw[:end])
        except Exception:
            pass
    for start in [m.start() for m in re.finditer(r'\{', raw)]:
        depth = 0; in_string = False; escaped = False
        for pos in range(start, len(raw)):
            ch = raw[pos]
            if in_string:
                if escaped: escaped = False
                elif ch == '\\': escaped = True
                elif ch == '"': in_string = False
                continue
            if ch == '"': in_string = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start:pos + 1]); break
    seen = set()
    for text in candidates:
        if text in seen: continue
        seen.add(text)
        try: obj = json.loads(text)
        except Exception: continue
        if isinstance(obj, dict) and 'actions' in obj and ('action_count' in obj or 'semantic_state' in obj) and ('first_person_hand' in obj or 'animal_visible' in obj):
            return obj
    return None


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=Path, required=True)
    ap.add_argument('--tail-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--expected', type=int, default=597)
    args = ap.parse_args()
    rows = []
    rows.extend(read_rows(args.pilot))
    for path in sorted(args.tail_dir.glob('reviews_worker_*.jsonl')):
        rows.extend(read_rows(path))
    rows.sort(key=lambda x: (int(x.get('queue_index', 10**9)), str(x.get('backend_version', ''))))
    by_idx = {}; duplicates = []
    for row in rows:
        idx = row.get('queue_index')
        if not isinstance(idx, int): continue
        if idx in by_idx:
            duplicates.append(idx)
            continue
        by_idx[idx] = row
    canonical = []
    malformed = []
    completed = []
    decisions = []
    for idx in range(args.expected):
        row = by_idx.get(idx)
        if row is None:
            continue
        parsed = strict_parse_json(row.get('raw_response', ''))
        out = dict(row)
        out['parser_version'] = 'strict_top_level_v1'
        out['parsed_review'] = parsed
        out['review_status'] = 'completed' if parsed is not None else 'malformed'
        if parsed is None:
            out['raw_review_status'] = row.get('review_status')
            malformed.append(out)
            state = 'semantic_backend_failed'
            reason = 'malformed_or_truncated_top_level_json_requires_retry'
        else:
            completed.append(out)
            state = 'review_needed'
            reason = 'sidecar_proposal_requires_deterministic_temporal_privacy_and_human_review'
        canonical.append(out)
        decisions.append({
            'queue_index': idx, 'source': out.get('source'), 'file': out.get('file'),
            'start_s': out.get('source_start_s'), 'end_s': out.get('source_end_s'),
            'backend_name': out.get('backend_name'), 'backend_version': out.get('backend_version'),
            'semantic_review_path': str(args.out_dir / 'semantic_review.jsonl'),
            'state': state, 'status': state, 'semantic_reviewed': False,
            'accepted_for_training': False, 'training_ready': False,
            'promotion_status': 'not_promoted', 'decision_reason': reason,
        })
    missing = [i for i in range(args.expected) if i not in by_idx]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'semantic_review.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in canonical) + ('\n' if canonical else ''), encoding='utf-8')
    (args.out_dir / 'semantic_decisions.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in decisions) + ('\n' if decisions else ''), encoding='utf-8')
    (args.out_dir / 'semantic_retry_queue.jsonl').write_text('\n'.join(json.dumps({k: x.get(k) for k in ('queue_index','source','file','source_start_s','source_end_s')} | {'reason':'malformed_or_truncated_top_level_json'}, ensure_ascii=False) for x in malformed) + ('\n' if malformed else ''), encoding='utf-8')
    report = {
        'expected': args.expected, 'input_rows': len(rows), 'canonical_rows': len(canonical),
        'completed_valid': len(completed), 'malformed_retry': len(malformed),
        'missing_indices': missing, 'duplicate_indices': sorted(set(duplicates)),
        'state_counts': {'review_needed': len(completed), 'semantic_backend_failed': len(malformed)},
        'parser_version': 'strict_top_level_v1', 'promotion_performed': False,
    }
    (args.out_dir / 'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__': main()
