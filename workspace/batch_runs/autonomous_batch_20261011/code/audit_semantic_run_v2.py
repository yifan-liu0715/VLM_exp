#!/usr/bin/env python3
import argparse, concurrent.futures, json, subprocess, time
from pathlib import Path

def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    return p.returncode,p.stdout,p.stderr

def check_file(rec):
    f=Path(rec.get('file') or '')
    out={'file':str(f),'state':rec.get('state'),'exists':f.is_file(),
         'encoded_format_verified':rec.get('boundary_evidence',{}).get('encoded_format_verified'),
         'encoded_semantics_verified':rec.get('boundary_evidence',{}).get('encoded_semantics_verified'),
         'face_count_detected':rec.get('face_count_detected'),
         'faces_remaining_after_crop':rec.get('faces_remaining_after_crop'),
         'privacy_reviewed':rec.get('privacy_reviewed'),
         'status':rec.get('status')}
    if not f.is_file(): out['validation']='missing'; return out
    rc,stdout,stderr=run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height,codec_name,pix_fmt,r_frame_rate','-of','json',str(f)])
    out['ffprobe_ok']=rc==0
    if rc==0:
        try:
            d=json.loads(stdout); out['metadata']=d
            v=next((s for s in d.get('streams',[]) if s.get('codec_type')=='video'),{})
            out['spec_ok']=(v.get('width')==1280 and v.get('height')==720 and v.get('codec_name')=='h264' and v.get('pix_fmt')=='yuv420p' and v.get('r_frame_rate')=='16/1')
        except Exception as e: out['metadata_error']=repr(e); out['spec_ok']=False
    rc2,_,err2=run(['ffmpeg','-v','error','-threads','2','-i',str(f),'-map','0:v:0','-f','null','-'])
    out['decode_ok']=rc2==0
    if rc2: out['decode_error']=err2[-1000:]
    out['validation']='verified' if out['ffprobe_ok'] and out.get('spec_ok') and out['decode_ok'] else 'failed'
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--output',required=True); ap.add_argument('--workers',type=int,default=2); args=ap.parse_args()
    rd=Path(args.run_dir); results=json.loads((rd/'auto_batch_results.json').read_text(encoding='utf-8'))
    manifests=[]; source_status=[]
    for r in results:
        source_status.append({'source_id':r.get('source_id'),'status':r.get('status'),'clips':r.get('clips',0),'second_pass_required':r.get('second_pass_required',0),'analysis':r.get('analysis',{})})
        m=rd.parent.parent/'clips'/str(r.get('source_id'))/'clips_manifest.json'
        if m.is_file():
            try: manifests.extend(json.loads(m.read_text(encoding='utf-8')))
            except Exception as e: source_status[-1]['manifest_error']=repr(e)
        else: source_status[-1]['manifest_missing']=str(m)
    files=[r for r in manifests if r.get('file')]
    checked=[]; started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for x in ex.map(check_file,files): checked.append(x)
    report={'run_dir':str(rd),'source_results':source_status,'source_count':len(results),'candidate_manifest_records':len(manifests),'candidate_files':len(files),'file_validation_verified':sum(x['validation']=='verified' for x in checked),'file_validation_failed':sum(x['validation']=='failed' for x in checked),'missing_files':sum(not x['exists'] for x in checked),'states':{},'face_hits':sum(bool(x.get('face_count_detected')) for x in checked),'face_scan_unresolved':sum(x.get('faces_remaining_after_crop') is None for x in checked),'accepted_for_training':sum(bool(x.get('accepted_for_training')) for x in manifests),'privacy_reviewed_true':sum(bool(x.get('privacy_reviewed')) for x in manifests),'elapsed_s':round(time.time()-started,1),'workers':args.workers,'files':checked}
    for r in manifests: report['states'][r.get('state')]=report['states'].get(r.get('state'),0)+1
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('source_count','candidate_files','file_validation_verified','file_validation_failed','missing_files','face_hits','accepted_for_training','elapsed_s')},ensure_ascii=False),flush=True)
if __name__=='__main__': main()

