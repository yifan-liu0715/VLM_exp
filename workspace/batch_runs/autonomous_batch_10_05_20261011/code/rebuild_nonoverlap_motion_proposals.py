#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

R = Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
OLD_OUT = R / 'motion_proposal_clips'
OUT = R / 'motion_proposal_clips_v2_nonoverlap'
MANIFEST_OLD = R / 'motion_proposal_manifest.jsonl'
MANIFEST = R / 'motion_proposal_manifest_v2_nonoverlap.jsonl'
REPORT = R / 'reports' / 'overlap_dedup_v2.json'
QUEUE = R / 'uncertain' / 'motion_proposal_review_needed_v2.md'
PROGRESS = R / 'progress_motion_proposals_v2.json'
WORKERS = int(os.environ.get('MOTION_REBUILD_WORKERS', '4'))
THREADS = int(os.environ.get('MOTION_REBUILD_FFMPEG_THREADS', '2'))


def now():
    return datetime.datetime.now().astimezone().isoformat()


def run(cmd, timeout=1800):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-3000:])
    return p.stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def probe(path: Path):
    data = json.loads(run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                           '-show_format', '-of', 'json', str(path)]))
    streams = data.get('streams', [])
    vs = [s for s in streams if s.get('codec_type') == 'video']
    if not vs:
        raise RuntimeError('no video stream')
    v = vs[0]
    frames = int(v.get('nb_read_frames') or 0)
    dur = float((data.get('format') or {}).get('duration') or 0.0)
    return {
        'width': int(v.get('width') or 0), 'height': int(v.get('height') or 0),
        'codec': v.get('codec_name'), 'pix_fmt': v.get('pix_fmt'),
        'fps': v.get('avg_frame_rate'), 'r_fps': v.get('r_frame_rate'),
        'frames': frames, 'duration_s': dur,
        'audio': any(s.get('codec_type') == 'audio' for s in streams),
    }


def validate(path: Path, expected: float):
    fmt = probe(path)
    run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
         '-threads', str(THREADS), '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'], timeout=1800)
    expected_fmt = (1280, 720, 'h264', 'yuv420p', '16/1', '16/1', False)
    got = (fmt['width'], fmt['height'], fmt['codec'], fmt['pix_fmt'], fmt['fps'], fmt['r_fps'], fmt['audio'])
    if got != expected_fmt:
        raise RuntimeError(f'format mismatch: {fmt}')
    if abs(fmt['duration_s'] - expected) > 0.15:
        raise RuntimeError(f'duration mismatch: {fmt["duration_s"]} vs {expected}')
    if fmt['frames'] < 1:
        raise RuntimeError('zero decoded frames')
    return fmt


def fixed_crop(row):
    dw = int(row.get('display_width') or row.get('width') or 0)
    dh = int(row.get('display_height') or row.get('height') or 0)
    if dw <= 0 or dh <= 0 or dw / dh >= (16 / 9) - 1e-6:
        return None
    cw = min(dw, int((dh * 16 / 9) // 2 * 2))
    ch = int((cw * 9 / 16) // 2 * 2)
    cw = max(2, cw - (cw % 2)); ch = max(2, ch - (ch % 2))
    cx = max(0, (dw - cw) // 2); cy = max(0, (dh - ch) // 2)
    return (cx - cx % 2, cy - cy % 2, cw, ch)


def vf_for(row):
    crop = fixed_crop(row)
    if crop:
        cx, cy, cw, ch = crop
        return f'crop={cw}:{ch}:{cx}:{cy},fps=16,scale=1280:720:flags=lanczos,setsar=1', [cw, ch, cx, cy]
    return 'fps=16,scale=1280:720:flags=lanczos,setsar=1', None


def output_name(sid, idx, start, end):
    sm = round(float(start) * 1000); em = round(float(end) * 1000)
    return f'{sid}__mp_{idx:04d}__{sm:09d}-{em:09d}__motion_proposal.mp4'


def load_sources():
    return {r['source_id']: r for r in json.loads((R / 'full_inventory.json').read_text(encoding='utf-8'))}


def load_proposals():
    out = []
    for p in sorted((R / 'proposals').glob('*/motion_proposals.json')):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        for i, item in enumerate(data, 1):
            x = dict(item)
            x['_proposal_file'] = str(p)
            x['_proposal_index'] = i
            out.append(x)
    return out


def load_old():
    d = {}
    if not MANIFEST_OLD.exists():
        return d
    for line in MANIFEST_OLD.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            x = json.loads(line)
            key = (x.get('source_id'), round(float(x.get('source_start_s') or 0), 6), round(float(x.get('source_end_s') or 0), 6))
            d[key] = x
        except Exception:
            pass
    return d


def cluster_proposals(proposals):
    by = {}
    for x in proposals:
        by.setdefault(x.get('source_id'), []).append(x)
    clusters = []
    overlap_pairs = []
    for sid, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda x: (float(x.get('start_s') or 0), float(x.get('end_s') or 0), int(x.get('_proposal_index') or 0)))
        cur = None
        for x in rows:
            a = float(x.get('start_s') or 0); b = float(x.get('end_s') or 0)
            if cur is None or a >= cur['end'] - 1e-9:
                if cur is not None:
                    clusters.append(cur)
                cur = {'source_id': sid, 'start': a, 'end': b, 'members': [x]}
            else:
                for y in cur['members']:
                    ov = min(float(y.get('end_s') or 0), b) - max(float(y.get('start_s') or 0), a)
                    if ov > 0.001:
                        overlap_pairs.append({'source_id': sid, 'a': int(y.get('_proposal_index') or 0), 'b': int(x.get('_proposal_index') or 0), 'overlap_s': round(ov, 6)})
                cur['end'] = max(cur['end'], b)
                cur['members'].append(x)
        if cur is not None:
            clusters.append(cur)
    clusters.sort(key=lambda c: (c['source_id'], c['start'], c['end']))
    return clusters, overlap_pairs


def build_row(cluster, idx, sources, old_map):
    sid = cluster['source_id']; members = cluster['members']; first = members[0]
    srcrow = sources[sid]
    start = round(cluster['start'], 6); end = round(cluster['end'], 6)
    old = old_map.get((sid, round(start, 6), round(end, 6))) if len(members) == 1 else None
    row = dict(old) if old else {
        'source_id': sid, 'source_path': srcrow.get('local_source'),
        'remote_source': srcrow.get('source'), 'processing_source': first.get('processing_source') or first.get('source'),
        'source_sha256': srcrow.get('sha256'), 'source_duration_s': float(srcrow.get('duration') or 0),
        'source_orientation': 'landscape' if srcrow.get('landscape', True) else 'portrait',
        'proposal_state': 'motion_proposal', 'proposal_reason': 'mechanical non-overlap proposal; semantic review required',
        'action': 'unreviewed_hand_activity', 'response': 'unreviewed_animal_response',
        'caption_verified': False, 'semantic_reviewed': False, 'privacy_reviewed': False,
        'accepted_for_training': False, 'keypoints_started': False, 'training_started': False,
        'state': 'motion_proposal', 'status': 'motion_proposal', 'training_ready': False,
        'temporal_envelope': 'motion_proposal', 'face_review_status': 'pending', 'human_reviewed': False,
        'created_at': now(),
    }
    row.update({
        'event_index': idx, 'source_start_s': start, 'source_end_s': end,
        'source_id': sid, 'source_duration_s': float(srcrow.get('duration') or row.get('source_duration_s') or 0),
        'source_path': srcrow.get('local_source', row.get('source_path')),
        'remote_source': srcrow.get('source', row.get('remote_source')),
        'processing_source': first.get('processing_source') or first.get('source') or row.get('processing_source'),
        'proposal_file': [m.get('_proposal_file') for m in members],
        'proposal_component_indices': [int(m.get('_proposal_index') or 0) for m in members],
        'overlap_cluster': len(members) > 1,
        'overlap_cluster_size': len(members),
        'review_reason': 'overlap_cluster_merged_for_source_context_resplit' if len(members) > 1 else row.get('review_reason', 'motion proposal requires source-context semantic review'),
        'state': 'motion_proposal', 'status': 'motion_proposal', 'semantic_reviewed': False,
        'privacy_reviewed': False, 'accepted_for_training': False, 'keypoints_started': False, 'training_started': False,
        'long_motion_proposal': (end - start) > 15.0, 'short_motion_proposal': (end - start) < 3.0,
        'portrait_reframe_review': not bool(srcrow.get('landscape', True)),
        'output_path': str(OUT / sid / output_name(sid, idx, start, end)),
        'crop': None,
    })
    row['output_path'] = str(OUT / sid / output_name(sid, idx, start, end))
    return row, old


def process(item):
    row, old = item
    sid = row['source_id']; start = float(row['source_start_s']); end = float(row['source_end_s']); duration = end - start
    out = Path(row['output_path']); out.parent.mkdir(parents=True, exist_ok=True)
    reused = False
    try:
        oldpath = Path(old.get('output_path')) if old and old.get('output_path') else None
        if oldpath and oldpath.is_file() and abs(float(old.get('source_start_s') or 0) - start) < 1e-6 and abs(float(old.get('source_end_s') or 0) - end) < 1e-6 and oldpath != out:
            try:
                if out.exists(): out.unlink()
                os.link(oldpath, out)
                reused = True
            except Exception:
                shutil.copy2(oldpath, out)
                reused = True
        if not out.exists():
            src = Path(row['processing_source'])
            if not src.is_file():
                raise RuntimeError(f'processing source missing: {src}')
            # Use inventory dimensions to keep the same fixed 16:9 rule.
            # Source metadata is carried in row only for the crop helper below.
            f = row.get('_vf') or 'fps=16,scale=1280:720:flags=lanczos,setsar=1'
            tmp = out.with_suffix('.tmp.mp4')
            cmd = ['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads',str(THREADS),'-filter_threads',str(THREADS),'-ss',f'{start:.6f}','-i',str(src),'-t',f'{duration:.6f}','-map','0:v:0','-vf',f,'-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads',str(THREADS),'-vsync','cfr','-metadata:s:v:0','rotate=0','-movflags','+faststart','-y',str(tmp)]
            run(cmd)
            fmt = validate(tmp, duration)
            tmp.replace(out)
        fmt = validate(out, duration)
        row.update({'output_sha256': sha256(out), 'encoded_format_verified': True, 'decoded_frame_count_verified': True, 'normalized_frame_count': fmt['frames'], 'encoded_width': fmt['width'], 'encoded_height': fmt['height'], 'encoded_fps': 16, 'encoded_duration_s': fmt['duration_s'], 'format': fmt, 'error': None, 'reused_from_v1': reused})
    except Exception as e:
        row.update({'encoded_format_verified': False, 'decoded_frame_count_verified': False, 'error': str(e), 'state': 'review_needed', 'status': 'review_needed', 'reused_from_v1': reused})
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True); (R/'reports').mkdir(exist_ok=True); (R/'uncertain').mkdir(exist_ok=True)
    sources = load_sources(); proposals = load_proposals(); old_map = load_old()
    clusters, overlap_pairs = cluster_proposals(proposals)
    rows=[]
    for idx, c in enumerate(clusters, 1):
        row, old = build_row(c, idx, sources, old_map)
        f, crop = vf_for(sources[c['source_id']])
        row['_vf'] = f; row['crop'] = crop
        rows.append((row, old))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs=[ex.submit(process, x) for x in rows]
        for n, fut in enumerate(concurrent.futures.as_completed(futs),1):
            results.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(json.dumps({'done':n,'total':len(futs)},ensure_ascii=False),flush=True)
    results.sort(key=lambda x:(str(x.get('source_id')),int(x.get('event_index') or 0)))
    for r in results: r.pop('_vf', None)
    MANIFEST.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    report={
        'created_at': now(), 'original_proposals': len(proposals), 'nonoverlap_windows': len(results),
        'reduction': len(proposals)-len(results), 'overlap_pairs': len(overlap_pairs),
        'overlap_clusters': sum(1 for r in results if r.get('overlap_cluster')),
        'encoded_success': sum(bool(r.get('encoded_format_verified')) for r in results),
        'encoded_failed': sum(not bool(r.get('encoded_format_verified')) for r in results),
        'reused_from_v1': sum(bool(r.get('reused_from_v1')) for r in results),
        'total_output_duration_s': round(sum(float(r.get('encoded_duration_s') or 0) for r in results),3),
        'overlap_examples': overlap_pairs[:100], 'manifest': str(MANIFEST), 'output_root': str(OUT),
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 非重叠机械候选切片（skill 第二遍复核队列）','',
           f'- 原始机械候选：{len(proposals)} 条', f'- 非重叠复核窗口：{len(results)} 条',
           f'- 合并的重叠候选组：{report["overlap_clusters"]} 组',
           '', '这些仍然是 `motion_proposal`，不是语义已接受数据。重叠候选已合并成一个原片上下文窗口，人工审核时应在窗口内重新拆分不同动作；不能把合并窗口直接作为最终训练片段。', '']
    for r in results:
        if r.get('overlap_cluster') or r.get('long_motion_proposal') or r.get('short_motion_proposal') or r.get('portrait_reframe_review') or r.get('error'):
            lines.append(f'- `{r.get("source_id")}` / {r.get("event_index")}: {r.get("source_start_s")}–{r.get("source_end_s")}s; components={r.get("proposal_component_indices")}; reason={r.get("review_reason") or r.get("error")}')
    QUEUE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    PROGRESS.write_text(json.dumps({'state':'motion_proposals_v2_nonoverlap_review_needed', **report, 'semantic_reviewed':0, 'privacy_reviewed':0, 'accepted_for_training':0, 'keypoints_started':False, 'training_started':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
