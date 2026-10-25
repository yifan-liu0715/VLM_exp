#!/usr/bin/env python3
from __future__ import annotations
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011'); sys.path.insert(0,str(R/'code'))
from proposal_detector import detect_events
REC=R/'recovery_normalized_pts'; PROP=R/'proposals'; REPORT=R/'reports'/'recovery_pts'
def now(): return datetime.datetime.now().astimezone().isoformat()
def run(cmd,timeout=10800):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',timeout=timeout)
 if p.returncode: raise RuntimeError((p.stderr or p.stdout)[-4000:])
 return p.stdout
def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def probe(path):
 d=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)])); v=next(s for s in d['streams'] if s.get('codec_type')=='video')
 return {'duration':float(d['format']['duration']),'width':int(v['width']),'height':int(v['height']),'fps':v.get('avg_frame_rate'),'frames':int(v.get('nb_read_frames') or 0)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--sid',required=True); args=ap.parse_args(); sid=args.sid
 inv={r['source_id']:r for r in json.loads((R/'full_inventory.json').read_text(encoding='utf-8'))}; row=inv[sid]; src=Path(row['source']); proxy=REC/(sid+'.mp4'); report=REPORT/(sid+'.json'); propdir=PROP/sid; REC.mkdir(exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True); propdir.mkdir(parents=True,exist_ok=True)
 try:
  try: meta=probe(proxy) if proxy.exists() else None
  except Exception:
   proxy.unlink(missing_ok=True); meta=None
  if meta is None:
   vf='setpts=N/(60*TB),scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=16,setsar=1'
   run(['ffmpeg','-y','-nostdin','-hide_banner','-loglevel','error','-fflags','+genpts','-i',str(src),'-map','0:v:0','-vf',vf,'-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-r','16','-vsync','cfr','-movflags','+faststart',str(proxy)])
   meta=probe(proxy)
  events,analysis=detect_events(proxy,meta)
  records=[]
  for i,(a,b) in enumerate(events,1):
   records.append({'source':str(src),'processing_source':str(proxy),'file':None,'start_s':round(float(a),6),'end_s':round(float(b),6),'state':'motion_proposal','status':'motion_proposal','next_state':'second_pass_required','second_pass_required':True,'second_pass_completed':False,'action':'unreviewed_hand_activity','response':'unreviewed_animal_response','interaction_mode':None,'animal_type':None,'caption_text':None,'caption_verified':False,'semantic_valid':False,'training_ready':False,'semantic_reviewed':False,'privacy_reviewed':False,'accepted_for_training':False,'keypoint_ready':False,'pipeline_version':'hand-presence-motion-proposals-v4-second-pass-queue-pts-normalized','analysis':analysis,'source_orientation':'portrait' if not row.get('landscape',True) else 'landscape','source_id':sid,'source_sha256':row['sha256'],'source_size_bytes':row['size_bytes'],'processing_source_sha256':sha(proxy),'source_pts_normalized':True,'processing_resolution':[meta['width'],meta['height']],'source_coverage_status':'proposal_only','review_reason':'PTS-normalized derived proxy; source-context semantic review required','boundary_evidence':{'hand_transition_checked':False,'response_complete':False,'temporal_envelope':'motion_proposal'}})
  (propdir/'motion_proposals.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
  result={'source_id':sid,'source':str(src),'processing_source':str(proxy),'source_sha256':row['sha256'],'state':'source_context_ready' if analysis.get('detector_status')=='ready' else 'review_needed','status':'source_context_ready' if analysis.get('detector_status')=='ready' else 'review_needed','source_duration_s':row['duration'],'processing_duration_s':meta['duration'],'analysis':analysis,'mechanical_proposal_count':len(records),'second_pass_count':len(records),'source_pts_normalized':True,'finished_at':now(),'error':None if analysis.get('detector_status')=='ready' else analysis.get('detector_error')}
  (R/'reports'/'source_results'/(sid+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
  report.write_text(json.dumps({'source_id':sid,'source':str(src),'processing_source':str(proxy),'original_sha256':row['sha256'],'proxy_sha256':sha(proxy),'probe':meta,'analysis':analysis,'proposal_count':len(records),'source_pts_normalized':True,'finished_at':now()},ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'source_id':sid,'state':result['state'],'proposals':len(records),'detector_status':analysis.get('detector_status'),'hand_frames':analysis.get('hand_frames')},ensure_ascii=False),flush=True)
 except Exception as e:
  report.write_text(json.dumps({'source_id':sid,'error':repr(e),'finished_at':now()},ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'source_id':sid,'state':'review_needed','error':repr(e)},ensure_ascii=False),flush=True); return 1
 return 0
if __name__=='__main__': raise SystemExit(main())
