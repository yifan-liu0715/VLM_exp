import concurrent.futures,datetime,hashlib,json,math,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011');inv={r['source_id']:r for r in json.loads((R/'full_inventory.json').read_text())};jobs=json.loads((R/'plans/pilot_fixed_crop_jobs.json').read_text(encoding='utf-8-sig'))
def one(j):
 r=inv[j['source_id']];s,e,fps=j['start_s'],j['end_s'],j['fps'];crop=j['crop'];w,h,x,y=crop;assert 0<=x<=r['display_width']-w and 0<=y<=r['display_height']-h and abs(w/h-16/9)<.001
 out=R/'source_context'/r['source_id']/'crop_trials'/j['job_id'];out.mkdir(parents=True,exist_ok=True);report=out/'context.json'
 if report.exists():
  old=json.loads(report.read_text())
  if old.get('job')==j and all(Path(p).exists() for p in old['pages']):return old
 receipt=json.loads((R/'receipts'/(r['source_id']+'.json')).read_text());assert receipt['verified'] and receipt['sha256']==r['sha256']
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-filter_threads','2','-hwaccel','cuda','-hwaccel_device','0','-ss',str(s),'-i',r['source'],'-t',str(e-s),'-vf',f'crop={w}:{h}:{x}:{y},fps={fps},setsar=1','-q:v','3',str(out/'frame_%06d.jpg')];p=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
 if p.returncode or p.stderr.strip():raise RuntimeError(p.stderr[-1000:])
 frames=sorted(out.glob('frame_*.jpg'));assert abs(len(frames)-(e-s)*fps)<=1;pages=[]
 for off in range(0,len(frames),24):
  sel=frames[off:off+24];im=Image.new('RGB',(1536,math.ceil(len(sel)/4)*240),(20,20,20));d=ImageDraw.Draw(im)
  for i,f in enumerate(sel):
   with Image.open(f) as tile:
    tile.thumbnail((384,216));xx=i%4*384;yy=i//4*240;im.paste(tile,(xx,yy));d.text((xx+4,yy+220),f'src={s+(off+i)/fps:.4f}s',fill='white')
  dest=out/f'page_{off//24+1:03d}.jpg';im.save(dest,quality=91);pages.append(str(dest))
 d={'source_id':r['source_id'],'source_sha256':r['sha256'],'job':j,'frames':len(frames),'pages':pages,'orientation_correction':'metadata autorotation exactly once, then crop in display coordinates','privacy_reviewed':False,'semantic_reviewed':False,'updated_at':datetime.datetime.now().astimezone().isoformat()};report.write_text(json.dumps(d,indent=2));return d
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
 for d in pool.map(one,jobs):print(json.dumps({'job_id':d['job']['job_id'],'frames':d['frames'],'pages':len(d['pages'])}),flush=True)
