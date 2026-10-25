#!/usr/bin/env python3
import argparse, concurrent.futures, hashlib, json, os, subprocess, time
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return p.returncode, p.stdout, p.stderr

def sha256(path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def one(row):
    path=Path(row['source'])
    r={'source_id':row['source_id'],'source':str(path),'expected_size_bytes':row.get('size_bytes'),
       'expected_sha256':row.get('sha256'),'expected_duration_s':row.get('duration'),
       'exists':path.is_file()}
    if not r['exists']:
        r.update(status='missing',reason='file_not_found'); return r
    try:
        r['actual_size_bytes']=path.stat().st_size
        r['size_match']=r['actual_size_bytes']==r['expected_size_bytes']
        r['actual_sha256']=sha256(path)
        r['sha256_match']=r['actual_sha256']==r['expected_sha256']
        rc,out,err=run(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,codec_name,pix_fmt,r_frame_rate','-of','json',str(path)])
        r['ffprobe_returncode']=rc
        if rc==0:
            d=json.loads(out); r['ffprobe']=d
            streams=d.get('streams') or []; v=next((s for s in streams if s.get('codec_type')=='video'),{})
            r['actual_duration_s']=float((d.get('format') or {}).get('duration',0.0) or 0.0)
            r['duration_delta_s']=abs(r['actual_duration_s']-float(row.get('duration') or 0.0))
            r['duration_match']=r['duration_delta_s'] <= 0.08
            r['width']=v.get('width'); r['height']=v.get('height'); r['codec']=v.get('codec_name'); r['pix_fmt']=v.get('pix_fmt'); r['fps']=v.get('r_frame_rate')
        else:
            r['ffprobe_stderr']=err[-2000:]
            r['duration_match']=False
        rc2,out2,err2=run(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-f','null','-'])
        r['decode_returncode']=rc2; r['decode_ok']=rc2==0
        if rc2: r['decode_stderr']=err2[-2000:]
        r['status']='verified' if (r['size_match'] and r['sha256_match'] and r.get('ffprobe_returncode')==0 and r.get('duration_match') and r['decode_ok']) else 'failed'
    except Exception as e:
        r['status']='failed'; r['error']=repr(e)
    return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inventory',required=True); ap.add_argument('--output',required=True); ap.add_argument('--workers',type=int,default=2); args=ap.parse_args()
    rows=json.loads(Path(args.inventory).read_text(encoding='utf-8'))
    started=time.time(); results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(one,r) for r in rows]
        for n,f in enumerate(concurrent.futures.as_completed(futs),1):
            r=f.result(); results.append(r); print(json.dumps({'done':n,'total':len(rows),'source_id':r['source_id'],'status':r['status']},ensure_ascii=False),flush=True)
    results.sort(key=lambda x:x['source_id'])
    report={'run_id':'autonomous_batch_20261011','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'source_count':len(rows),'verified_count':sum(r['status']=='verified' for r in results),'failed_count':sum(r['status']=='failed' for r in results),'elapsed_s':round(time.time()-started,1),'workers':args.workers,'results':results}
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
    print(json.dumps({k:report[k] for k in ('source_count','verified_count','failed_count','elapsed_s')},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
