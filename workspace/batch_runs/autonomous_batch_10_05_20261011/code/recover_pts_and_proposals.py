#!/usr/bin/env python3
"""Recover non-monotonic-P​​TS iPhone sources into derived 1280x720 CFR16 proxies.
The original files remain untouched. This only creates motion_proposal intervals."""
from __future__ import annotations
import datetime, hashlib, json, os, subprocess
from pathlib import Path
import sys

R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
CODE=R/'code'
sys.path.insert(0,str(CODE))
from proposal_detector import detect_events

REC=R/'recovery_normalized_pts'; PROP=R/'proposals'; REPORT=R/'reports'/'recovery_pts'

def now(): return datetime.datetime.now().astimezone().isoformat()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(cmd, timeout=7200):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',timeout=timeout)
 if p.returncode: raise RuntimeError((p.stderr or p.stdout)[-4000:])
 return p.stdout
def probe(path):
 d=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
 v=next(s for s in d['streams'] if s.get('codec_type')=='video')
 return {'duration':float(d['format']['duration']),'width':int(v['width']),'height':int(v['height']),'fps':v.get('avg_frame_rate'),'frames':int(v.get('nb_read_frames') or 0)}
def main():
 inv={r['source_id']:r for r in json.loads((R/'full_inventory.json').read_text(encoding='utf-8'))}
 bad=[]
 for p in (R/'reports'/'source_results').glob('*.json'):
  try:d=json.loads(p.read_text(encoding='utf-8'))
  except:continue
  if d.get('error') and 'decoder reported errors' in str(d.get('error')): bad.append(d['source_id'])
 for p in REPORT.glob('*.json'):
  try:d=json.loads(p.read_text(encoding='utf-8'))
  except:continue
  if d.get('error') or d.get('analysis',{}).get('detector_status') != 'ready': bad.append(d.get('source_id'))
 bad=sorted(set(x for x in bad if x))
 REC.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
 all_summary=[]
 for sid in bad:
  row=inv[sid]; src=Path(row['source']); proxy=REC/(sid+'.mp4'); report=REPORT/(sid+'.json'); propdir=PROP/sid; propdir.mkdir(parents=True,exist_ok=True)
  try:
   if proxy.exists():
    try: probe(proxy)
    except Exception:
     try: proxy.unlink()
     except FileNotFoundError: pass
   if not proxy.exists():
    # setpts removes non-monotonic DTS effects in a derived file; use 60 Hz source clock as diagnosed.
    vf='setpts=N/(60*TB),scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=16,setsar=1'
    run(['ffmpeg','-y','-nostdin','-hide_banner','-loglevel','error','-fflags','+genpts','-i',str(src),'-map','0:v:0','-vf',vf,'-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-r','16','-vsync','cfr','-movflags','+faststart',str(proxy)],timeout=10800)
   meta=probe(proxy); events,analysis=detect_events(proxy,meta)
   records=[]
   for i,(a,b) in enumerate(events,1):
    records.append({'source':str(src),'processing_source':str(proxy),'file':None,'start_s':round(float(a),6),'end_s':round(float(b),6),
      'state':'motion_proposal','status':'motion_proposal','next_state':'second_pass_required','second_pass_required':True,'second_pass_completed':False,
      'action':'unreviewed_hand_activity','response':'unreviewed_animal_response','interaction_mode':None,'animal_type':None,'caption_text':None,
      'caption_verified':False,'semantic_valid':False,'training_ready':False,'semantic_reviewed':False,'privacy_reviewed':False,
      'accepted_for_training':False,'keypoint_ready':False,'pipeline_version':'hand-presence-motion-proposals-v4-second-pass-queue-pts-normalized',
      'analysis':analysis,'source_orientation':'portrait' if not row.get('landscape',True) else 'landscape','source_id':sid,
      'source_sha256':row['sha256'],'source_size_bytes':row['size_bytes'],'processing_source_sha256':sha(proxy),
      'source_pts_normalized':True,'processing_resolution':[meta['width'],meta['height']],
      'source_coverage_status':'proposal_only','review_reason':'PTS-normalized derived proxy; source-context semantic review required',
      'boundary_evidence':{'hand_transition_checked':False,'response_complete':False,'temporal_envelope':'motion_proposal'}})
   (propdir/'motion_proposals.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
   result={'source_id':sid,'source':str(src),'processing_source':str(proxy),'source_sha256':row['sha256'],'state':'source_context_ready','status':'source_context_ready','source_duration_s':row['duration'],'processing_duration_s':meta['duration'],'analysis':analysis,'mechanical_proposal_count':len(records),'second_pass_count':len(records),'second_pass_items':[],'source_pts_normalized':True,'finished_at':now(),'error':None}
   (R/'reports'/'source_results'/(sid+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
   report.write_text(json.dumps({'source_id':sid,'source':str(src),'processing_source':str(proxy),'original_sha256':row['sha256'],'proxy_sha256':sha(proxy),'probe':meta,'analysis':analysis,'proposal_count':len(records),'source_pts_normalized':True,'finished_at':now()},ensure_ascii=False,indent=2),encoding='utf-8')
   all_summary.append({'source_id':sid,'state':'source_context_ready','proposals':len(records),'proxy':str(proxy)})
   print(json.dumps(all_summary[-1],ensure_ascii=False),flush=True)
  except Exception as e:
   report.write_text(json.dumps({'source_id':sid,'error':repr(e),'finished_at':now()},ensure_ascii=False,indent=2),encoding='utf-8')
   all_summary.append({'source_id':sid,'state':'review_needed','error':repr(e)}); print(json.dumps(all_summary[-1],ensure_ascii=False),flush=True)
 summary={'finished_at':now(),'sources':len(bad),'results':all_summary,'state':'pts_normalized_motion_proposals_ready'}
 (R/'reports'/'recovery_pts_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
