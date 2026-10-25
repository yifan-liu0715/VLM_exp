import argparse, importlib.metadata, json, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import cv2
except Exception:
    cv2 = None
import numpy as np

from video_face_detection import privacy_fields, scan_video
from causal_pipeline_contract import add_uncovered_second_pass_intervals

ROOT=Path(r"/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/clips")
WORK=Path(r"/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011")
INV=WORK/'full_inventory.json'
PIPELINE_VERSION='autonomous_batch_20261011-hand-presence-motion-proposals-v4'


def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if p.returncode: raise RuntimeError(p.stderr[-2000:])
    return p.stdout

def probe(path):
    d=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    v=next(s for s in d['streams'] if s['codec_type']=='video')
    count_data=json.loads(run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                               '-show_entries','stream=nb_read_frames','-of','json',str(path)]))
    counted=(count_data.get('streams') or [{}])[0].get('nb_read_frames')
    decoded_frames=int(counted) if str(counted).isdigit() else None
    return {'width':int(v['width']),'height':int(v['height']),'fps':v.get('r_frame_rate',''),
            'codec':v.get('codec_name',''),'pix_fmt':v.get('pix_fmt',''),
            'duration':float(d['format']['duration']), 'decoded_frames':decoded_frames,
            'normalized_frame_count':decoded_frames,
            'audio':any(s['codec_type']=='audio' for s in d['streams'])}


def validate_output_file(path):
    """Validate format and complete decode before treating an existing file as reusable."""
    try:
        metadata=probe(path)
        run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'])
    except Exception:
        return False
    return (metadata['width'], metadata['height'], metadata['fps'], metadata['codec'], metadata['pix_fmt'], metadata['audio']) == (1280, 720, '16/1', 'h264', 'yuv420p', False)

def build_hand_presence_intervals(times, hand_presence, duration,
                                  bridge_gap_s=0.75, lead_s=0.25,
                                  tail_s=1.0):
    """Create broad proposals only; the result is never a semantic decision."""
    if len(times) != len(hand_presence):
        raise ValueError('times and hand_presence must have equal length')
    detected=[i for i,value in enumerate(hand_presence) if value]
    if not detected:
        return []
    sample_step=float(np.median(np.diff(times))) if len(times)>1 else 0.0
    groups=[[detected[0]]]
    for index in detected[1:]:
        previous=groups[-1][-1]
        missing_gap=max(0.0,float(times[index])-float(times[previous])-sample_step)
        if missing_gap<=bridge_gap_s:
            groups[-1].append(index)
        else:
            groups.append([index])
    intervals=[]
    for group in groups:
        start=max(0.0,float(times[group[0]])-lead_s)
        end=min(float(duration),float(times[group[-1]])+tail_s)
        if end>start:
            intervals.append((round(start,3),round(end,3)))
    return intervals


def load_hand_detector():
    from hand_presence_detection import HandPresenceDetector
    return HandPresenceDetector


def detect_events(path, meta, *, detector_cleanup=None):
    """Decode all sampled frames; runtime/inference failures remain explicit."""
    import tempfile
    if cv2 is None:
        return [], {'detector_status':'detector_failed', 'detector_error':'opencv-python unavailable',
                    'hand_frames':0, 'frames':0}
    try:
        hands=load_hand_detector()()
    except Exception as error:
        return [], {'detector_status':'detector_failed', 'detector_error':str(error),
                    'hand_frames':0, 'frames':0}
    duration=float(meta['duration'])
    sample_fps=6.0 if duration<=30 else 3.0
    w,h=640,360
    frame_bytes=w*h*3
    times=[]; hand_ok=[]; hand_counts=[]
    analysis={'sample_fps':sample_fps, 'detector_backend':hands.backend,
              'detector_version':hands.version, 'model_sha256':hands.model_sha256}
    pipe=None
    try:
        with tempfile.TemporaryFile() as error_log:
            pipe=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-threads','2',
                '-i',str(path), '-vf',f'fps={sample_fps},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2',
                '-threads','2','-f','rawvideo','-pix_fmt','bgr24','-'],
                stdout=subprocess.PIPE,stderr=error_log)
            while True:
                raw=pipe.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw)!=frame_bytes:
                    raise RuntimeError('partial raw frame from decoder')
                frame=np.frombuffer(raw,dtype=np.uint8).reshape((h,w,3))
                t=len(times)/sample_fps
                result=hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),t)
                times.append(t); hand_counts.append(result.hand_count)
                hand_ok.append(int(result.hand_count>0))
            pipe.stdout.close()
            pipe.wait(timeout=30)
            error_log.seek(0); error=error_log.read().decode(errors='replace')
            if pipe.returncode or error.strip():
                raise RuntimeError(error[-2000:] or f'ffmpeg return code {pipe.returncode}')
        expected=duration*sample_fps
        if not times or abs(len(times)-expected)>max(3.0, expected*0.01):
            raise RuntimeError(f'sampled decode incomplete: {len(times)} frames versus {expected:.2f} expected')
        events=build_hand_presence_intervals(times,hand_ok,duration)
        analysis.update(detector_status='ready' if events else 'no_hands_detected',
            hand_frames=int(sum(hand_ok)),frames=len(times),proposal_count=len(events),
            sample_times_s=times,hand_counts=hand_counts,all_sampled_frames_decoded=True)
        return events,analysis
    except Exception as error:
        if pipe is not None and pipe.poll() is None:
            pipe.kill(); pipe.wait(timeout=10)
        analysis.update(detector_status='decode_or_inference_failed', detector_error=str(error),
                        hand_frames=int(sum(hand_ok)),frames=len(times),all_sampled_frames_decoded=False)
        return [],analysis
    finally:
        if detector_cleanup is None:
            hands.close()
        else:
            detector_cleanup(hands)


def build_second_pass_queue(source, duration, records, proposed_intervals):
    """Represent hand proposals and uncovered source gaps separately.

    This is still mechanical bookkeeping: no item is semantically accepted.
    """
    by_interval = {(round(float(r.get("start_s", 0.0)), 6), round(float(r.get("end_s", 0.0)), 6)): r
                   for r in records if r.get("file")}
    queue = []
    for start, end, kind in add_uncovered_second_pass_intervals(duration, proposed_intervals):
        key = (round(start, 6), round(end, 6))
        record = by_interval.get(key)
        queue.append({
            "source": str(source), "file": record.get("file") if record else None,
            "start_s": start, "end_s": end,
            "second_pass_required": True, "second_pass_completed": False,
            "interval_kind": kind,
            "normalized_frame_count": int(record.get("normalized_frame_count") or 0) if record else 0,
            "reason": "mechanical_hand_proposal" if kind == "hand_proposal" else "uncovered_source_interval",
        })
    return queue

def make_motion_proposal_record(source, output, start, end, analysis, media,
                                orientation='landscape', second_pass_reason=None):
    return {
        'source':str(source), 'file':str(output) if output else None,
        'start_s':start, 'end_s':end,
        'state':'motion_proposal', 'status':'motion_proposal',
        'next_state':'second_pass_required', 'second_pass_required':True,
        'second_pass_completed':False,
        'action':'unreviewed_hand_activity', 'response':'unreviewed_animal_response',
        'interaction_mode':None, 'animal_type':None,
        'caption_text':None, 'caption_verified':False,
        'semantic_valid':False, 'training_ready':False,
        'semantic_reviewed':False, 'privacy_reviewed':False,
        'accepted_for_training':False, 'keypoint_ready':False,
        'pipeline_version':PIPELINE_VERSION, 'analysis':analysis,
        'source_orientation':orientation, 'source_coverage_status':'proposal_only',
        'face_count_detected':None, 'face_removal_method':None,
        'faces_remaining_after_crop':None, 'subject_clarity_after_crop':None,
        'review_reason':second_pass_reason or 'motion proposal requires source-context semantic review',
        'boundary_evidence':{
            'hand_transition_checked':False,
            'split_or_merge_reason':'broad hand-presence proposal; semantic review required',
            'causal_start_s':None, 'action_or_condition_onset_s':None,
            'response_onset_s':None, 'response_evidence_s':None,
            'tail_checked_until_s':None, 'response_complete':False,
            'source_end_truncated':None, 'temporal_envelope':'motion_proposal',
            'encoded_format_verified':bool(media), 'encoded_semantics_verified':False,
        },
        **media,
    }


def encode_clip(source,out,start,end,index,analysis):
    action='unreviewed_hand_activity'; response='unreviewed_animal_response'
    name=f'{source.stem}__{index:03d}__{round(start*1000):09d}-{round(end*1000):09d}__{action}__{response}.mp4'
    outpath=out/name; tmp=out/(name+'.encoding.mp4')
    # Mechanical proposal render only. A semantic reviewer must replace the
    # center crop when a portrait or privacy-sensitive reframe needs another crop.
    vf='scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=16'
    run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-i',str(source),
         '-t',str(end-start),'-map','0:v:0','-vf',vf,'-c:v','libx264','-threads','2',
         '-preset','fast','-crf','18','-pix_fmt','yuv420p','-an','-movflags','+faststart','-y',str(tmp)])
    run(['ffmpeg','-v','error','-i',str(tmp),'-f','null','-'])
    m=probe(tmp)
    if not (m['width']==1280 and m['height']==720 and m['fps']=='16/1' and
            m['codec']=='h264' and m['pix_fmt']=='yuv420p' and not m['audio']):
        raise RuntimeError(f'output spec failed: {m}')
    tmp.replace(outpath)
    record=make_motion_proposal_record(source,outpath,start,end,analysis,m)
    face_report=scan_video(outpath,fps=16.0)
    record.update(privacy_fields(face_report))
    if face_report.get('status') in {'detector_failed','decode_failed','decode_or_inference_failed'}:
        record['review_reason']=face_report.get('error') or face_report.get('status')
    elif face_report.get('face_count_detected',0):
        record['review_reason']='recognizable_face_remaining_or_fixed_crop_required'
    else:
        record['review_reason']='no_detector_hits; full-timeline visual privacy review still required'
    return record


def process(row, force=False):
    sid=row['source_id']; source=Path(row['source']); out=ROOT/sid; out.mkdir(parents=True,exist_ok=True)
    manifest=out/'clips_manifest.json'
    if manifest.exists() and not force:
        records=json.loads(manifest.read_text(encoding='utf-8'))
        valid=all(r.get('pipeline_version')==PIPELINE_VERSION and
                  r.get('source_edge_sha256')==row.get('edge_sha256') and
                  (not r.get('file') or (Path(r['file']).is_file() and validate_output_file(Path(r['file'])))) for r in records)
        intervals=[(float(r.get('start_s', 0.0)), float(r.get('end_s', 0.0))) for r in records if r.get('file')]
        queue=build_second_pass_queue(source, float(row.get('duration', 0.0)), records, intervals)
        return {'source_id':sid,'status':'exists_valid' if valid else 'existing_requires_validation',
                'clips':len(records),'second_pass_required':len(queue), 'second_pass_items':queue,
                'coverage_status':'proposal_and_gap_coverage'}

    orientation='landscape' if row.get('landscape') else 'portrait'
    events,analysis=detect_events(source,row)
    if analysis.get('detector_status') in {'detector_failed','decode_failed','decode_or_inference_failed'}:
        record=make_motion_proposal_record(source,None,0.0,float(row.get('duration',0.0)),analysis,
                                           {},orientation,'detector_failed_or_decode_failed')
        manifest.write_text(json.dumps([record],ensure_ascii=False,indent=2),encoding='utf-8')
        queue=[{'source':str(source),'file':None,'start_s':0.0,'end_s':float(row.get('duration',0.0)),
                 'second_pass_required':True,'second_pass_completed':False,
                 'interval_kind':'uncovered_source_interval','reason':'detector_failed_or_decode_failed'}]
        return {'source_id':sid,'source':str(source),'status':analysis['detector_status'],
                'clips':0,'second_pass_required':len(queue),'analysis':analysis,
                'manifest':str(manifest),'second_pass_items':queue,
                'coverage_status':'detector_failed_full_source_review'}
    if not events:
        reason='portrait_reframe_and_semantic_review_required' if orientation=='portrait' else \
               'no_hand_proposal; review stable-hand animal-led events and detector misses'
        record=make_motion_proposal_record(source,None,0.0,float(row.get('duration',0.0)),analysis,
                                           {},orientation,reason)
        manifest.write_text(json.dumps([record],ensure_ascii=False,indent=2),encoding='utf-8')
        queue=[{'source':str(source),'file':None,'start_s':0.0,'end_s':float(row.get('duration',0.0)),
                 'second_pass_required':True,'second_pass_completed':False,
                 'interval_kind':'uncovered_source_interval','reason':reason}]
        return {'source_id':sid,'source':str(source),'status':'second_pass_required',
                'clips':0,'second_pass_required':len(queue),'analysis':analysis,'reason':reason,
                'manifest':str(manifest),'second_pass_items':queue,
                'coverage_status':'no_hand_proposal_full_source_review'}

    records=[]
    for idx,(a,b) in enumerate(events,1):
        try:
            record=encode_clip(source,out,a,b,idx,analysis)
            record.update(source_edge_sha256=row.get('edge_sha256'), source_bytes=row.get('bytes'))
            if orientation=='portrait':
                record['source_orientation']='portrait'
                record['review_reason']='portrait fixed crop requires semantic subject-preservation review'
            records.append(record)
        except Exception as e:
            records.append(make_motion_proposal_record(source,None,a,b,analysis,{},orientation,f'encode_failed: {e}'))
    manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    queue=build_second_pass_queue(source, float(row.get('duration', 0.0)), records, events)
    return {'source_id':sid,'source':str(source),'status':'motion_proposals_created','clips':len(records),
            'second_pass_required':len(queue),'events':events,'analysis':analysis,'duration':row.get('duration'),
            'manifest':str(manifest),'coverage_status':'proposal_and_gap_coverage',
            'second_pass_items':queue}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=3); ap.add_argument('--limit',type=int); ap.add_argument('--offset',type=int,default=0); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    rows=json.loads(INV.read_text(encoding='utf-8'))
    pending=[r for r in rows if not r.get('reference_available') and r.get('status')!='probe_failed']
    pending=pending[args.offset:args.offset+args.limit if args.limit else None]
    print(json.dumps({'pending':len(pending),'workers':args.workers},ensure_ascii=False),flush=True)
    results=[]; second_pass_queue=[]; started=time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(process,r,args.force) for r in pending]
        for j,f in enumerate(as_completed(futures),1):
            try:
                x=f.result(); results.append(x)
                second_pass_queue.extend(x.get('second_pass_items', []))
                if not x.get('second_pass_items'):
                    second_pass_queue.append({k:x.get(k) for k in ('source_id','source','status','reason','second_pass_required','clips','analysis')})
                print(json.dumps({'done':j,'total':len(pending),**x},ensure_ascii=False),flush=True)
            except Exception as e:
                second_pass_queue.append({'status':'second_pass_required','reason':f'worker_failed: {e}'})
                print(json.dumps({'error':str(e)},ensure_ascii=False),flush=True)
    run_id=time.strftime('%Y%m%d_%H%M%S'); run_dir=WORK/'runs'/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/'auto_batch_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (run_dir/'second_pass_queue.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in second_pass_queue)+'\n',encoding='utf-8')
    lines=['# Second-pass review queue','',
           'These are mechanical proposals or uncovered/uncertain source intervals. No item is semantically accepted.', '']
    for x in sorted(results,key=lambda z:z.get('source_id','')):
        lines.append(f"- `{x.get('source_id')}`: status={x.get('status')}; clips={x.get('clips',0)}; second_pass_required={x.get('second_pass_required',0)}; reason={x.get('reason') or x.get('analysis',{}).get('detector_error','semantic review required')}.")
    lines.extend(['', '## Intervals requiring second pass', ''])
    for item in sorted(second_pass_queue, key=lambda z: (str(z.get('source','')), float(z.get('start_s') or 0.0))):
        lines.append(f"- `{item.get('source')}` [{item.get('start_s')}–{item.get('end_s')}] `{item.get('interval_kind','unknown')}`: {item.get('reason','review required')}")
    (run_dir/'uncertain_interactions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'run_dir':str(run_dir),'elapsed_s':round(time.time()-started,1),
                      'proposal_sources':sum(x.get('status')=='motion_proposals_created' for x in results),
                      'clips':sum(x.get('clips',0) for x in results),
                      'second_pass_items':sum(x.get('second_pass_required',0) for x in results),
                      'detector_failures':sum(x.get('status')=='detector_failed' for x in results),
                      'portrait_sources':sum(x.get('status')=='second_pass_required' and 'portrait' in str(x.get('reason','')) for x in results)},ensure_ascii=False),flush=True)
if __name__=='__main__': main()



