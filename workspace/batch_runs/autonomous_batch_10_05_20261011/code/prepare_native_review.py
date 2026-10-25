import json,pathlib,subprocess,math,datetime,concurrent.futures,fcntl,hashlib
from PIL import Image,ImageDraw
R=pathlib.Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011');inv={x['source_id']:x for x in json.loads((R/'full_inventory.json').read_text())};jobs=json.loads((R/'plans/pilot_native_jobs.json').read_text(encoding='utf-8-sig'))
lock=(R/'locks/native_review_preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def one(j):
 r=inv[j['source_id']];fps=j['fps'];out=R/'source_context'/r['source_id']/f'native_{fps}fps';out.mkdir(parents=True,exist_ok=True);record=out/'native_review_report.json'
 if record.exists():
  d=json.loads(record.read_text())
  if d.get('source_sha256')==r['sha256'] and all(pathlib.Path(p).is_file() for p in d.get('pages',[])):return d
 assert sha(r['source'])==r['sha256'];p=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-filter_threads','2','-i',r['source'],'-vf',f'fps={fps},setsar=1','-q:v','3','-y',str(out/'frame_%06d.jpg')],capture_output=True,text=True,timeout=1800)
 if p.returncode:raise RuntimeError(p.stderr[-2000:])
 files=sorted(out.glob('frame_*.jpg'));pages=[];portrait=r['display_width']<r['display_height'];tw,th,cols,per=(384,702,3,12) if portrait else (384,240,4,24)
 for offset in range(0,len(files),per):
  sel=files[offset:offset+per];canvas=Image.new('RGB',(tw*cols,math.ceil(len(sel)/cols)*th),(15,15,15));d=ImageDraw.Draw(canvas)
  for n,f in enumerate(sel):
   with Image.open(f) as im:
    im.thumbnail((tw,th-24));x=(n%cols)*tw;y=(n//cols)*th;canvas.paste(im,(x+(tw-im.width)//2,y));d.text((x+4,y+th-21),f'{(offset+n)/fps:.3f}s',fill='white')
  dst=out/f'review_page_{offset//per+1:03d}.jpg';canvas.save(dst,quality=90);pages.append(str(dst))
 report={'source_id':r['source_id'],'source_sha256':r['sha256'],'native_display_dimensions':[r['display_width'],r['display_height']],'metadata_rotation_deg':r['rotation_deg'],'orientation_correction':'ffmpeg metadata autorotation once','sample_fps':fps,'time_error_bound_s':.5/fps,'frames':len(files),'pages':pages,'source_time_s':[i/fps for i in range(len(files))],'decode_exit':p.returncode,'decode_stderr':p.stderr,'semantic_review_complete':False};record.write_text(json.dumps(report,indent=2));return report
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for report in ex.map(one,jobs):print(json.dumps({k:report[k] for k in ['source_id','frames','sample_fps','pages']}),flush=True)
