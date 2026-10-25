"""Encode explicitly reviewed source-event plans; mechanical validation is never semantic approval."""
import argparse,datetime,fcntl,hashlib,json,math,os,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
from video_face_detection import scan_video
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=1800)
 if p.returncode or p.stderr.strip():raise RuntimeError(p.stderr[-3000:] or str(p.returncode))
 return p.stdout
def pages(out,dest,start):
 dest.mkdir(parents=True,exist_ok=True)
 run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-filter_threads','2','-i',str(out),'-q:v','3',str(dest/'frame_%06d.jpg')])
 frames=sorted(dest.glob('frame_*.jpg'));ps=[];tw,th=320,202
 for off in range(0,len(frames),30):
  sel=frames[off:off+30];c=Image.new('RGB',(tw*5,math.ceil(len(sel)/5)*th),(15,15,15));draw=ImageDraw.Draw(c)
  for j,f in enumerate(sel):
   with Image.open(f) as im:
    im.thumbnail((tw,th-22));x=(j%5)*tw;y=(j//5)*th;c.paste(im,(x,y));draw.text((x+3,y+th-19),f'src={start+(off+j)/16:.3f}s  #{off+j}',fill='white')
  p=dest/f'page_{off//30+1:03d}.jpg';c.save(p,quality=90);ps.append(str(p))
 return {'frames':len(frames),'pages':ps}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args();plan=json.loads(args.plan.read_text(encoding='utf-8-sig'));sid=plan['source_id'];lock=(R/'locks'/f'export_{sid}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic(R/'locks'/f'export_{sid}_owner.json',{'pid':os.getpid(),'started_at':datetime.datetime.now().astimezone().isoformat()})
 src=next(r for r in json.loads((R/'full_inventory.json').read_text()) if r['source_id']==sid);path=Path(src['source']);digest=sha(path);assert digest==src['sha256']==plan['source_sha256'];assert path.stat().st_size==src['size_bytes'];receipt=json.loads((R/'receipts'/(sid+'.json')).read_text());assert receipt['verified'] and receipt['sha256']==digest
 all_rows=[]
 for e in plan['events']:
  start,end=e['start_s'],e['end_s'];assert 0<=start<end<=src['duration']+.001
  crop=e.get('crop');vf=''
  if crop:
   w,h,x,y=map(int,crop);assert w%2==0 and h%2==0 and abs(w/h-16/9)<.001 and 0<=x and 0<=y and x+w<=src['display_width'] and y+h<=src['display_height'];vf=f'crop={w}:{h}:{x}:{y},'
  else:assert abs(src['display_width']/src['display_height']-16/9)<.001
  name=f'{sid}__{e["event_id"]}__{round(start*1000):09d}-{round(end*1000):09d}__{e["hand_action"]}.mp4';out=R/'candidate_clips'/sid/name;out.parent.mkdir(parents=True,exist_ok=True);ev=R/'encoded_review'/sid/e['event_id'];ev.mkdir(parents=True,exist_ok=True);record=ev/'validation.json'
  if record.exists():
   old=json.loads(record.read_text())
   if old['event_plan']==e and out.exists() and sha(out)==old['output_sha256']:all_rows.append(old);continue
  if out.exists():raise RuntimeError('Output exists without matching validation; use a new plan version')
  run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-filter_threads','2','-ss',str(start),'-i',str(path),'-t',str(end-start),'-map','0:v:0','-vf',vf+'fps=16,scale=1280:720:flags=lanczos,setsar=1','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2','-vsync','cfr','-metadata:s:v:0','rotate=0','-movflags','+faststart',str(out)])
  probe=json.loads(run(['ffprobe','-v','error','-threads','2','-count_frames','-show_streams','-show_format','-of','json',str(out)]));v=next(s for s in probe['streams'] if s['codec_type']=='video');count=int(v['nb_read_frames']);dur=float(probe['format']['duration']);run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-i',str(out),'-map','0:v:0','-f','null','-'])
  lines=run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(out)]);pts=[float(x.split(',')[0]) for x in lines.splitlines() if x.split(',')[0].strip()];cfr=len(pts)==count and all(abs(b-a-1/16)<1e-5 for a,b in zip(pts,pts[1:]));fmt=v['width']==1280 and v['height']==720 and v['pix_fmt']=='yuv420p' and v['codec_name']=='h264' and v['avg_frame_rate']=='16/1' and v['r_frame_rate']=='16/1' and not any(s['codec_type']=='audio' for s in probe['streams']);assert fmt and cfr and abs(dur-(end-start))<=.063
  face=scan_video(out,fps=16);atomic(ev/'face_detection.json',face);frame_pages=pages(out,ev/'frames',start);assert frame_pages['frames']==count
  d={'source_id':sid,'event_id':e['event_id'],'source_path':src['local_source'],'remote_source':str(path),'source_sha256':digest,'source_sha256_verified':True,'source_start_s':start,'source_end_s':end,'event_plan':e,'plan_sha256':sha(args.plan),'remote_output_path':str(out),'output_sha256':sha(out),'encoded_width':v['width'],'encoded_height':v['height'],'encoded_fps':16,'encoded_duration_s':dur,'normalized_frame_count':count,'full_decode_verified':True,'decoded_frame_count_verified':True,'encoded_format_verified':fmt,'cfr_timestamps_verified':cfr,'frame_ge_49':count>=49,'frame_ge_85':count>=85,'stage1_window_ok':None,'stage2_window_ok':None,'loader_review':False,'face_detection_report':str(ev/'face_detection.json'),'face_detection_status':face.get('status'),'face_detection_all_frames':face.get('all_frames_checked'),'face_count_detected':face.get('face_count_detected'),'faces_remaining_after_crop':None,'subject_clarity_after_crop':None,'privacy_reviewed':False,'encoded_semantics_verified':False,'state':'review_needed','status':'review_needed','accepted_for_training':False,'human_reviewed':False,'keypoints_started':False,'training_started':False,'vlm_used':False,'final_review_pages':frame_pages['pages'],'source_review_pages':plan['source_review_pages'],'updated_at':datetime.datetime.now().astimezone().isoformat()};atomic(record,d);all_rows.append(d);print(json.dumps({k:d[k] for k in ['source_id','event_id','normalized_frame_count','face_detection_status','face_count_detected','state']}),flush=True)
 (R/'plans'/f'{sid}_encoded.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows))
if __name__=='__main__':main()
