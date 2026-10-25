"""Prepare source-context review packages; never export hand intervals as approved clips."""
import argparse, datetime, fcntl, hashlib, json, math, os, shutil, subprocess, time
from pathlib import Path
from PIL import Image, ImageDraw
import proposal_detector as mechanical

def now():return datetime.datetime.now().astimezone().isoformat()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run(args,timeout=7200):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[-3000:] or str(p.returncode))
    return p

def verify(row,root):
    path=Path(row['source']);receipt=root/'receipts'/(row['source_id']+'.json');expected=row['sha256']
    r={'source_id':row['source_id'],'source':str(path),'expected_sha256':expected,'expected_bytes':row['size_bytes'],'expected_duration_s':row['duration'],'exists':path.is_file(),'started_at':now()}
    if not r['exists'] or not receipt.exists():raise RuntimeError('source or transfer receipt missing')
    recorded=json.loads(receipt.read_text());assert recorded.get('verified') and recorded.get('sha256')==expected,'receipt not verified'
    r.update(actual_bytes=path.stat().st_size,actual_sha256=sha(path));r['size_match']=r['actual_bytes']==r['expected_bytes'];r['sha256_match']=r['actual_sha256']==expected
    assert r['size_match'] and r['sha256_match'],'remote identity mismatch'
    d=json.loads(run(['ffprobe','-v','error','-threads','2','-show_streams','-show_format','-of','json',str(path)],120).stdout)
    v=next(s for s in d['streams'] if s.get('codec_type')=='video');rotation=next((float(s['rotation']) for s in v.get('side_data_list',[]) if 'rotation' in s),float(v.get('tags',{}).get('rotate') or 0));r.update(actual_duration_s=float(d['format']['duration']),width=v['width'],height=v['height'],codec=v.get('codec_name'),rotation_deg=rotation,source_avg_fps=v.get('avg_frame_rate'))
    r['duration_match']=abs(r['actual_duration_s']-float(row['duration']))<=0.08
    assert r['duration_match'],'duration differs after transfer'
    decoded=run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-i',str(path),'-map','0:v:0','-an','-f','null','-'])
    r.update(full_decode_verified=True,decode_returncode=decoded.returncode,decode_stderr=decoded.stderr[-3000:],verified_at=now())
    assert not decoded.stderr.strip(),'decoder reported errors'
    atomic(root/'reports/source_validation'/(row['source_id']+'.json'),r)
    return r

def context(row,root):
    p=Path(row['source']);out=root/'source_context'/row['source_id'];out.mkdir(parents=True,exist_ok=True)
    # Overview only; full-resolution boundary frames are produced after visual event discovery.
    width,height=int(row['display_width']),int(row['display_height'])
    sw=max(2,2*round(width*640/max(width,height)/2));sh=max(2,2*round(height*640/max(width,height)/2))
    sample_fps=8 if float(row['duration'])<3 else 1
    filter=f'fps={sample_fps},scale={sw}:{sh}:flags=bicubic,setsar=1'
    run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-threads','2','-filter_threads','2','-i',str(p),'-vf',filter,'-q:v','3','-y',str(out/'overview_%06d.jpg')])
    files=sorted(out.glob('overview_*.jpg'))
    if not files:raise RuntimeError('no source-context frames decoded')
    if abs(len(files)-float(row['duration'])*sample_fps)>max(3,float(row['duration'])*sample_fps*0.01):raise RuntimeError('source-context decode incomplete')
    cols=5;tile_w=256;tile_h=170;per_page=30;pages=[]
    for offset in range(0,len(files),per_page):
        selected=files[offset:offset+per_page];nrows=math.ceil(len(selected)/cols)
        canvas=Image.new('RGB',(cols*tile_w,nrows*tile_h),(20,20,20));draw=ImageDraw.Draw(canvas)
        for j,f in enumerate(selected):
            with Image.open(f) as im:
                im=im.convert('RGB');im.thumbnail((tile_w,tile_h-22));x=(j%cols)*tile_w+(tile_w-im.width)//2;y=(j//cols)*tile_h;canvas.paste(im,(x,y));draw.text(((j%cols)*tile_w+5,y+tile_h-20),f'{(offset+j)/sample_fps:.3f}s',(255,255,255))
        page=out/f'overview_page_{offset//per_page+1:03d}.jpg';canvas.save(page,quality=88);pages.append(str(page))
    data={'source_id':row['source_id'],'source':str(p),'source_sha256':row['sha256'],'duration_s':row['duration'],'overview_sample_fps':sample_fps,'timestamps_are_coarse_localization_only':True,'display_width':width,'display_height':height,'metadata_rotation_deg':row.get('rotation_deg',0),'orientation_correction':'ffmpeg metadata autorotation exactly once','all_overview_frames':len(files),'pages':pages,'state':'source_context_ready','overview_is_semantic_approval':False,'normal_resolution_second_pass_required':True,'output_crop_review_required':not row.get('landscape',True),'privacy_reviewed':False,'keypoints_started':False,'training_started':False}
    atomic(out/'source_context.json',data);return data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--stage',default='pilot');args=ap.parse_args()
    root=args.root;root.joinpath('locks').mkdir(parents=True,exist_ok=True)
    lock=(root/'locks'/f'{args.stage}_prepare.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('preparer already live')
    atomic(root/'locks'/f'{args.stage}_prepare_owner.json',{'pid':os.getpid(),'started_at':now(),'stage':args.stage,'workers':1,'backend':'MediaPipe CPU; FFmpeg CPU, threads=2','keypoints_started':False,'training_started':False})
    rows=json.loads(args.inventory.read_text());results=[];queue=[]
    for row in rows:
        sid=row['source_id'];result_path=root/'reports/source_results'/(sid+'.json')
        previous=json.loads(result_path.read_text()) if result_path.exists() else None
        if previous and previous.get('source_sha256')==row['sha256'] and previous.get('state')=='source_context_ready':
            results.append(previous);queue.extend(previous.get('second_pass_items',[]));continue
        atomic(root/'progress_prepare.json',{'stage':args.stage,'state':'preparing_source','pid':os.getpid(),'source_id':sid,'done':len(results),'total':len(rows),'updated_at':now(),'worker_count':1,'keypoints_started':False,'training_started':False})
        r={'source_id':sid,'source':row['source'],'source_sha256':row['sha256'],'source_duration_s':row['duration'],'collection':row['collection'],'started_at':now(),'state':'review_needed','status':'review_needed'}
        try:
            r['validation']=verify(row,root)
            # Broad hand presence is evidence only. Missing detections do not drop a source.
            intervals,analysis=mechanical.detect_events(Path(row['source']),row)
            records=[]
            for start,end in intervals:
                rec=mechanical.make_motion_proposal_record(Path(row['source']),None,start,end,analysis,{},'landscape' if row.get('landscape') else 'portrait')
                rec.update(source_id=sid,source_sha256=row['sha256'],source_size_bytes=row['size_bytes']);records.append(rec)
            pdir=root/'proposals'/sid;pdir.mkdir(parents=True,exist_ok=True);atomic(pdir/'motion_proposals.json',records)
            second=mechanical.build_second_pass_queue(Path(row['source']),row['duration'],records,intervals)
            for q in second:q.update(source_id=sid,source_sha256=row['sha256'],collection=row['collection'],state='second_pass_required',status='second_pass_required')
            with (pdir/'second_pass_queue.jsonl').open('w',encoding='utf-8') as f:
                for q in second:f.write(json.dumps(q,ensure_ascii=False)+'\n')
            r.update(analysis=analysis,mechanical_proposal_count=len(records),second_pass_items=second,second_pass_count=len(second),source_context=context(row,root),state='source_context_ready',status='source_context_ready',semantic_reviewed=False,privacy_reviewed=False,accepted_for_training=False)
        except Exception as e:r.update(state='review_needed',status='review_needed',error=repr(e),accepted_for_training=False)
        r['finished_at']=now();atomic(result_path,r);results.append(r);queue.extend(r.get('second_pass_items',[]))
        with (root/'logs'/f'{args.stage}_worker_001.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps({'done':len(results),'total':len(rows),'source_id':sid,'state':r['state'],'proposals':r.get('mechanical_proposal_count',0),'error':r.get('error')},ensure_ascii=False),flush=True)
    with (root/'proposals/full_context_latest_wave_queue.jsonl').open('w',encoding='utf-8') as f:
        for q in queue:f.write(json.dumps(q,ensure_ascii=False)+'\n')
    summary={'stage':args.stage,'sources':len(rows),'source_context_ready':sum(r.get('state')=='source_context_ready' for r in results),'failed_sources':[r['source_id'] for r in results if r.get('error')],'mechanical_proposal_count':sum(r.get('mechanical_proposal_count',0) for r in results),'second_pass_count':len(queue),'accepted_for_training':0,'semantic_reviewed_sources':0,'finished_at':now(),'needs_visual_semantic_boundary_and_crop_review':True,'keypoints_started':False,'training_started':False,'vlm_used':False}
    atomic(root/'reports'/f'{args.stage}_preparation_report.json',summary);atomic(root/'progress_prepare.json',dict(summary,state='pilot_ready_for_semantic_boundary_review',pid=os.getpid()));print(json.dumps(summary,ensure_ascii=False),flush=True)
    fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()


