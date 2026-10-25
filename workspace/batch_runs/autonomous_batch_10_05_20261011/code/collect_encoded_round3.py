import json,math,zipfile,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
sids=['iphone__video_20261005_160632','iphone__video_20261005_163316','dji__DJI_20261005220906_0169_D']
files=[]
for sid in sids:
 for validation in sorted((R/'encoded_review'/sid).glob('*/validation.json')):
  ev=validation.parent; row=json.loads(validation.read_text()); files.extend([validation,ev/'face_detection.json',Path(row['remote_output_path'])]); files.extend((ev/'frames').glob('page_*.jpg'))
  face=json.loads((ev/'face_detection.json').read_text()); by={}
  for hit in face.get('hits',[]):by.setdefault(hit['frame_index'],[]).append(hit)
  items=sorted(by.items()); dest=ev/'face_hit_review_round3';dest.mkdir(exist_ok=True)
  for off in range(0,len(items),8):
   sel=items[off:off+8];canvas=Image.new('RGB',(1280,math.ceil(len(sel)/2)*314),(20,20,20));d=ImageDraw.Draw(canvas)
   for j,(index,hits) in enumerate(sel):
    f=ev/'frames'/f'frame_{index+1:06d}.jpg'
    with Image.open(f) as src:
     im=src.convert('RGB');draw=ImageDraw.Draw(im)
     for h in hits:
      x,y,w,hh=h['bbox_xywh'];draw.rectangle((x,y,x+w,y+hh),outline='red',width=4)
     im.thumbnail((474,267));xx=(j%2)*640;yy=(j//2)*314;canvas.paste(im,(xx,yy))
     h=hits[0];x,y,w,hh=h['bbox_xywh'];pad=.25*max(w,hh);box=(max(0,int(x-pad)),max(0,int(y-pad)),min(src.width,int(x+w+pad)),min(src.height,int(y+hh+pad)))
     if box[2]>box[0] and box[3]>box[1]:
      zoom=src.crop(box);zoom.thumbnail((162,267));canvas.paste(zoom,(xx+476,yy))
     d.text((xx+3,yy+276),f'frame={index} src={row["source_start_s"]+index/16:.4f}s hits={len(hits)}',fill='white')
   p=dest/f'page_{off//8+1:03d}.jpg';canvas.save(p,quality=92);files.append(p)
archive=R/'reports/pilot_encoded_round3.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
 for p in sorted(set(files)):z.write(p,str(p.relative_to(R)))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(set(files))}))

