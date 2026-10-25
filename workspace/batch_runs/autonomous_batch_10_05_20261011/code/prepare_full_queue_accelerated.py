"""Resume the existing preparation workflow on verified non-pilot sources, with measured CUDA decode. Never promotes clips."""
import datetime,fcntl,json,os,sys,time,subprocess
from pathlib import Path
import prepare_remote_wave as prepare
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def main():
 lock=(R/'locks/full_queue_preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 owner={'pid':os.getpid(),'started_at':now(),'source_workers':1,'ffmpeg_threads':2,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'decode_backend':'FFmpeg CUDA input decode; CPU output filters','semantic_approval':False,'keypoints_started':False,'training_started':False};atomic(R/'locks/full_queue_preparation_owner.json',owner)
 pilot={x['source_id'] for x in json.loads((R/'pilot_inventory.json').read_text())};inventory=json.loads((R/'full_inventory.json').read_text());rows=sorted([x for x in inventory if x['source_id'] not in pilot],key=lambda x:(x['size_bytes'],x['source_id']));attempts={};wave=max([int(p.stem.rsplit('_',1)[-1]) for p in (R/'plans').glob('full_context_wave_*.json')]+[0])
 base_atomic=prepare.atomic
 def isolated_atomic(p,d):
  if p.name=='progress_prepare.json':p=p.with_name('progress_prepare_full_wave.json')
  if p.name.endswith('_prepare_owner.json'):d.update(decode_backend=owner['decode_backend'],cuda_visible_devices=owner['cuda_visible_devices'])
  return base_atomic(p,d)
 prepare.atomic=isolated_atomic
 old_popen=subprocess.Popen
 def accelerated_popen(args,*a,**kw):
  if isinstance(args,(tuple,list)) and args and Path(str(args[0])).name=='ffmpeg' and '-i' in args and '-hwaccel' not in args:
   cmd=list(args);i=cmd.index('-i')
   if '/staging/raw/' in str(cmd[i+1]):
    cmd[i:i]=['-hwaccel','cuda','-hwaccel_device','0'];args=cmd
    with (R/'logs/cuda_prepare_commands.jsonl').open('a') as f:f.write(json.dumps({'at':now(),'cmd':args})+'\n')
  return old_popen(args,*a,**kw)
 subprocess.Popen=accelerated_popen
 while True:
  done=[];failed=[];ready=[];waiting=[]
  for row in rows:
   sid=row['source_id'];p=R/'reports/source_results'/f'{sid}.json';result=json.loads(p.read_text()) if p.exists() else None
   if result and result.get('source_sha256')==row['sha256'] and result.get('state')=='source_context_ready':done.append(sid);continue
   if result and result.get('source_sha256')==row['sha256'] and result.get('error') and attempts.get(sid,0)>=3:failed.append(sid);continue
   receipt=R/'receipts'/f'{sid}.json'
   if receipt.exists():
    d=json.loads(receipt.read_text());source=Path(row['source'])
    if d.get('verified') and d.get('sha256')==row['sha256'] and source.is_file() and source.stat().st_size==row['size_bytes']:ready.append(row);continue
   waiting.append(sid)
  state='preparing_verified_sources' if ready else ('waiting_for_upload' if waiting else 'finished')
  progress={**owner,'state':state,'total_non_pilot':len(rows),'pilot_sources_reserved':len(pilot),'context_prepared':len(done),'failed':len(failed),'waiting_for_upload':len(waiting),'ready':len(ready),'updated_at':now(),'semantically_accepted':0,'accepted_for_training':0};atomic(R/'progress_prepare_full.json',progress)
  if not ready and not waiting:
   atomic(R/'reports/full_context_preparation_finished.json',{**progress,'prepared_source_ids':done,'failed_source_ids':failed});print(json.dumps(progress),flush=True);return
  if not ready:time.sleep(30);continue
  free=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
  if free<80*1024**3:
   atomic(R/'progress_prepare_full.json',{**progress,'state':'disk_reserve_hold','free_bytes':free});time.sleep(30);continue
  batch=ready[:8];wave+=1;p=R/'plans'/f'full_context_wave_{wave:04d}.json';atomic(p,batch)
  for x in batch:attempts[x['source_id']]=attempts.get(x['source_id'],0)+1
  sys.argv=['prepare_remote_pilot.py','--root',str(R),'--inventory',str(p),'--stage',f'full_context_wave_{wave:04d}'];prepare.main();print(json.dumps({'wave':wave,'source_ids':[x['source_id'] for x in batch],'updated_at':now()}),flush=True)
if __name__=='__main__':main()

