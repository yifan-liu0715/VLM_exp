#!/usr/bin/env python3
"""Export remote hand-presence intervals as review-only motion-proposal clips.

This is intentionally separate from semantic promotion: every row remains
motion_proposal/review_needed until visual semantic, privacy and loader review.
It never changes source videos and is resumable by output+manifest identity.
"""
from __future__ import annotations
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import subprocess
import threading
from pathlib import Path

R = Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
OUT = R / 'motion_proposal_clips'
MANIFEST = R / 'motion_proposal_manifest.jsonl'
PROGRESS = R / 'progress_motion_proposals.json'
LOG = R / 'logs' / 'export_motion_proposals.log'
WORKERS = int(os.environ.get('MOTION_EXPORT_WORKERS', '4'))
FFMPEG_THREADS = int(os.environ.get('MOTION_EXPORT_FFMPEG_THREADS', '2'))

write_lock = threading.Lock()

def now():
    return datetime.datetime.now().astimezone().isoformat()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def run(cmd, timeout=1800):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-3000:])
    return p.stdout

def source_rows():
    rows = json.loads((R / 'full_inventory.json').read_text(encoding='utf-8'))
    return {r['source_id']: r for r in rows}

def load_proposals():
    rows = []
    for p in sorted((R / 'proposals').glob('*/motion_proposals.json')):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            print(f'PROPOSAL_READ_ERROR {p}: {e}', flush=True)
            continue
        if not isinstance(data, list):
            continue
        for i, item in enumerate(data, 1):
            item = dict(item)
            item['_proposal_file'] = str(p)
            item['_proposal_index'] = i
            rows.append(item)
    return rows

def fixed_crop(row):
    """Return a fixed display-space 16:9 crop for portrait candidates."""
    dw = int(row.get('display_width') or row.get('width') or 0)
    dh = int(row.get('display_height') or row.get('height') or 0)
    if dw <= 0 or dh <= 0 or dw / dh >= (16 / 9) - 1e-6:
        return None
    cw = min(dw, int((dh * 16 / 9) // 2 * 2))
    ch = int((cw * 9 / 16) // 2 * 2)
    cw = max(2, cw - (cw % 2))
    ch = max(2, ch - (ch % 2))
    cx = max(0, (dw - cw) // 2)
    cy = max(0, (dh - ch) // 2)
    cx -= cx % 2
    cy -= cy % 2
    return (cw, ch, cx, cy)

def make_filter(row):
    crop = fixed_crop(row)
    if crop:
        cw, ch, cx, cy = crop
        return f'crop={cw}:{ch}:{cx}:{cy},fps=16,scale=1280:720:flags=lanczos,setsar=1', crop
    return 'fps=16,scale=1280:720:flags=lanczos,setsar=1', None

def output_name(sid, idx, start, end):
    sm = round(float(start) * 1000)
    em = round(float(end) * 1000)
    return f'{sid}__mp_{idx:04d}__{sm:09d}-{em:09d}__motion_proposal.mp4'

def probe(path):
    data = json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
    vs = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
    if not vs:
        raise RuntimeError('no video stream')
    v = vs[0]
    count = int(v.get('nb_read_frames') or 0)
    dur = float((data.get('format') or {}).get('duration') or 0.0)
    fmt = {
        'width': int(v.get('width') or 0),
        'height': int(v.get('height') or 0),
        'codec': v.get('codec_name'),
        'pix_fmt': v.get('pix_fmt'),
        'fps': v.get('avg_frame_rate'),
        'r_fps': v.get('r_frame_rate'),
        'frames': count,
        'duration_s': dur,
        'audio': any(s.get('codec_type') == 'audio' for s in data.get('streams', [])),
    }
    return fmt

def validate(path, expected_duration):
    fmt = probe(path)
    run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads',str(FFMPEG_THREADS),
         '-i',str(path),'-map','0:v:0','-f','null','-'], timeout=1800)
    ok = (fmt['width'], fmt['height'], fmt['codec'], fmt['pix_fmt'], fmt['fps'], fmt['r_fps'], fmt['audio']) == \
         (1280, 720, 'h264', 'yuv420p', '16/1', '16/1', False)
    if not ok:
        raise RuntimeError(f'format mismatch: {fmt}')
    if abs(fmt['duration_s'] - expected_duration) > 0.15:
        raise RuntimeError(f'duration mismatch: {fmt["duration_s"]} vs {expected_duration}')
    if fmt['frames'] < 1:
        raise RuntimeError('zero decoded frames')
    return fmt

def process(item, sources):
    sid = item.get('source_id')
    row = sources.get(sid)
    if not row:
        return {'source_id': sid, 'event_index': item.get('_proposal_index'), 'state':'review_needed', 'error':'source_manifest_missing'}
    original_src = Path(row['source'])
    src = Path(item.get('processing_source') or row['source'])
    processing_row = dict(row)
    if item.get('processing_source'):
        processing_row.update({'width': 1280, 'height': 720, 'display_width': 1280, 'display_height': 720, 'landscape': True})
    start = max(0.0, float(item.get('start_s') or 0.0))
    end = min(float(row.get('duration') or item.get('end_s') or 0.0), float(item.get('end_s') or 0.0))
    duration = end - start
    idx = int(item.get('_proposal_index') or 0)
    base = {
        'source_id': sid, 'event_index': idx, 'source_path': row.get('local_source'),
        'remote_source': str(original_src), 'processing_source': str(src), 'source_sha256': row.get('sha256'),
        'source_start_s': start, 'source_end_s': end, 'source_duration_s': float(row.get('duration') or 0.0),
        'source_orientation': 'portrait' if not row.get('landscape', True) else 'landscape',
        'proposal_file': item.get('_proposal_file'), 'proposal_state': item.get('state','motion_proposal'),
        'proposal_reason': item.get('review_reason') or 'hand-presence motion proposal; semantic review required',
        'action': 'unreviewed_hand_activity', 'response': 'unreviewed_animal_response',
        'caption_verified': False, 'semantic_reviewed': False, 'privacy_reviewed': False,
        'accepted_for_training': False, 'keypoints_started': False, 'training_started': False,
        'state': 'motion_proposal', 'status': 'motion_proposal', 'training_ready': False,
        'temporal_envelope': 'motion_proposal', 'face_review_status': 'pending',
        'human_reviewed': False, 'created_at': now(),
    }
    if duration <= 0.05 or not src.is_file():
        base.update(error='invalid_interval_or_source_missing', state='review_needed', status='review_needed')
        return base
    crop = fixed_crop(processing_row)
    vf, crop = make_filter(processing_row)
    out_dir = OUT / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / output_name(sid, idx, start, end)
    base['output_path'] = str(out)
    base['crop'] = list(crop) if crop else None
    base['portrait_reframe_review'] = bool(crop) and not item.get('processing_source')
    base['source_pts_normalized'] = bool(item.get('source_pts_normalized'))
    base['long_motion_proposal'] = duration > 15.0
    base['short_motion_proposal'] = duration < 3.0
    try:
        if not out.exists():
            cmd = ['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads',str(FFMPEG_THREADS),
                   '-filter_threads',str(FFMPEG_THREADS),'-ss',f'{start:.6f}','-i',str(src),'-t',f'{duration:.6f}',
                   '-map','0:v:0','-vf',vf,'-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                   '-threads',str(FFMPEG_THREADS),'-vsync','cfr','-metadata:s:v:0','rotate=0','-movflags','+faststart',str(out)]
            try:
                run(cmd, timeout=1800)
            except Exception:
                # A retry with generated PTS is for decoder quirks only; original media is untouched.
                cmd2 = [x for x in cmd]
                i = cmd2.index('-i')
                cmd2[i:i] = ['-fflags','+genpts']
                run(cmd2, timeout=1800)
        fmt = validate(out, duration)
        base.update(output_sha256=sha256(out), encoded_format_verified=True,
                    decoded_frame_count_verified=True, normalized_frame_count=fmt['frames'],
                    encoded_width=fmt['width'], encoded_height=fmt['height'], encoded_fps=16,
                    encoded_duration_s=fmt['duration_s'], format=fmt, error=None)
    except Exception as e:
        base.update(error=str(e), encoded_format_verified=False, decoded_frame_count_verified=False,
                    state='review_needed', status='review_needed')
    return base

def write_progress(d):
    tmp = PROGRESS.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, PROGRESS)

def main():
    OUT.mkdir(parents=True, exist_ok=True); (R/'logs').mkdir(exist_ok=True)
    sources = source_rows(); proposals = load_proposals()
    # Stable unique key; a re-run of the same proposal must not duplicate manifest rows.
    unique=[]; seen=set()
    for item in proposals:
        key=(item.get('source_id'), round(float(item.get('start_s') or 0),6), round(float(item.get('end_s') or 0),6))
        if key in seen: continue
        seen.add(key); unique.append(item)
    existing={}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            try:
                r=json.loads(line); existing[(r.get('source_id'),r.get('event_index'))]=r
            except: pass
    total=len(unique); done=0; failed=0; reused=0; rows=[]; started=now()
    write_progress({'state':'running','started_at':started,'source_count':len(sources),'proposal_files':len(list((R/'proposals').glob('*/motion_proposals.json'))),
                    'proposal_intervals':total,'encoded':0,'failed':0,'reused':0,'accepted_for_training':0,'keypoints_started':False})
    def emit(r):
        nonlocal done,failed,reused
        done += 1
        if r.get('output_sha256') is None and r.get('error'): failed += 1
        if r.get('reused'): reused += 1
        with write_lock:
            with MANIFEST.open('a',encoding='utf-8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n')
        if done % 10 == 0 or done == total:
            write_progress({'state':'running','started_at':started,'updated_at':now(),'source_count':len(sources),
                            'proposal_files':len(list((R/'proposals').glob('*/motion_proposals.json'))),'proposal_intervals':total,
                            'encoded':done-failed,'failed':failed,'reused':reused,'accepted_for_training':0,
                            'keypoints_started':False,'training_started':False})
            print(json.dumps({'done':done,'total':total,'encoded':done-failed,'failed':failed,'reused':reused},ensure_ascii=False), flush=True)
    pending=[]
    for item in unique:
        key=(item.get('source_id'),int(item.get('_proposal_index') or 0))
        old=existing.get(key)
        if old and old.get('output_path') and Path(old['output_path']).is_file() and old.get('output_sha256'):
            try:
                if sha256(Path(old['output_path'])) == old['output_sha256']:
                    old=dict(old); old['reused']=True; pending.append(('reuse',old)); continue
            except: pass
        pending.append(('new',item))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures={ex.submit(process,item,sources):item for kind,item in pending if kind=='new'}
        for fut in concurrent.futures.as_completed(futures):
            emit(fut.result())
    for kind,item in pending:
        if kind=='reuse': emit(item)
    # Keep a compact review queue separate from the manifest.
    all_rows=[]
    for line in MANIFEST.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try: all_rows.append(json.loads(line))
            except: pass
    # Latest row for each source/event key.
    latest={}
    for r in all_rows: latest[(r.get('source_id'),r.get('event_index'))]=r
    rows=list(latest.values())
    rows.sort(key=lambda r:(str(r.get('source_id')),int(r.get('event_index') or 0)))
    MANIFEST.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary={'state':'motion_proposals_exported_review_needed','finished_at':now(),'source_count':len(sources),
             'proposal_files':len(list((R/'proposals').glob('*/motion_proposals.json'))),'proposal_intervals':total,
             'encoded_success':sum(bool(r.get('encoded_format_verified')) for r in rows),
             'encoded_failed':sum(not bool(r.get('encoded_format_verified')) for r in rows),
             'total_output_duration_s':round(sum(float(r.get('encoded_duration_s') or 0) for r in rows),3),
             'portrait_candidates':sum(bool(r.get('portrait_reframe_review')) for r in rows),
             'long_candidates_gt15s':sum(bool(r.get('long_motion_proposal')) for r in rows),
             'short_candidates_lt3s':sum(bool(r.get('short_motion_proposal')) for r in rows),
             'semantic_reviewed':0,'privacy_reviewed':0,'accepted_for_training':0,
             'keypoints_started':False,'training_started':False,'output_root':str(OUT),'manifest':str(MANIFEST)}
    write_progress(summary)
    (R/'reports'/'motion_proposal_export_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=['# 全量机械候选切片（仅 motion_proposal）','',f'生成时间：{summary["finished_at"]}','',
        f'- 源视频：{len(sources)}','- 候选区间：{total}','- 编码成功：{summary["encoded_success"]}','- 编码失败：{summary["encoded_failed"]}',
        f'- 候选总时长：{summary["total_output_duration_s"]} 秒','- 竖屏固定裁剪待审核：{summary["portrait_candidates"]}',
        f'- 超过 15 秒、需要人工重新拆分：{summary["long_candidates_gt15s"]}','',
        '这些文件是手部存在检测生成的机械候选，全部保持 `motion_proposal/review_needed`。它们不代表动作语义已确认，不代表没有人脸，也不进入关键点或训练。','']
    for r in rows:
        if r.get('error') or r.get('long_motion_proposal') or r.get('portrait_reframe_review'):
            md.append(f'- `{r.get("source_id")}` / {r.get("event_index")}: {r.get("error") or "long_or_portrait_review_required"}; `{r.get("output_path")}`')
    (R/'uncertain'/'motion_proposal_review_needed.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
