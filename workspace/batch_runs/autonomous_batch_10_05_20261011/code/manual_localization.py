import pathlib,json,subprocess,concurrent.futures,math,datetime
from PIL import Image,ImageDraw
R=pathlib.Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011');rows=json.loads((R/'pilot_inventory.json').read_text());jobs=[r for r in rows if (R/'receipts'/(r['source_id']+'.json')).exists() and r['source_id'] not in ['dji__DJI_20261008190537_0009_D','iphone__video_20261005_155721']]
def one(r):
 out=R/'source_context'/r['source_id']/'localization_manual';out.mkdir(parents=True,exist_ok=True);w,h=r['display_width'],r['display_height'];sw=2*round(w*960/max(w,h)/2);sh=2*round(h*960/max(w,h)/2)
 prior=out/'localization_report.json'
 if prior.exists():
  d=json.loads(prior.read_text())
  if d.get('source_sha256')==r['sha256'] and all(pathlib.Path(p).is_file() for p in d.get('pages',[])):return d
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-filter_threads','2','-i',r['source'],'-vf',f'fps=1,scale={sw}:{sh},setsar=1','-q:v','3','-y',str(out/'frame_%06d.jpg')]
 proc=subprocess.run(cmd,capture_output=True,text=True,timeout=1800)
 if proc.returncode:return {'sid':r['source_id'],'error':proc.stderr[-1000:]}
 files=sorted(out.glob('frame_*.jpg'));pages=[];tw,th=384,242;cols=4
 for offset in range(0,len(files),24):
  selected=files[offset:offset+24];im=Image.new('RGB',(cols*tw,math.ceil(len(selected)/cols)*th),(20,20,20));d=ImageDraw.Draw(im)
  for j,f in enumerate(selected):
   with Image.open(f) as tile:
    tile.thumbnail((tw,th-25));x=(j%cols)*tw;y=(j//cols)*th;im.paste(tile,(x+(tw-tile.width)//2,y));d.text((x+5,y+th-22),f'{offset+j:.3f}s',fill='white')
  p=out/f'page_{offset//24+1:03d}.jpg';im.save(p,quality=90);pages.append(str(p))
 data={'source_id':r['source_id'],'source_sha256':r['sha256'],'duration_s':r['duration'],'sample_fps':1,'mechanical_localization_only':True,'native_resolution_review_pending':True,'frames':len(files),'pages':pages,'source_decode_error':proc.stderr,'time':datetime.datetime.now().astimezone().isoformat()};prior.write_text(json.dumps(data,indent=2));return data
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for result in ex.map(one,jobs):print(json.dumps(result),flush=True)
