"""Resumable server-side Release publisher. Credential arrives on stdin and stays in memory."""
from __future__ import annotations
import concurrent.futures,datetime,hashlib,json,os,pathlib,subprocess,sys,threading,time,urllib.parse
import requests
import payload_safety as safety
import upload_archive_remote as archive
OUT=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025')
ROOT=pathlib.Path('/nvmedata/workspace2/users/nina')
REPO='yifan-liu0715/VLM_exp';LOCK=threading.RLock();PRINT_LOCK=threading.Lock()

def atomic(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def emit(value):
 with PRINT_LOCK:print(json.dumps(value,ensure_ascii=False),flush=True)

def now():return time.strftime('%Y-%m-%dT%H:%M:%S%z')

def main():
 config=json.load(sys.stdin);token=config.pop('token');workers=max(1,min(int(config.get('workers',2)),2))
 plan=json.loads((OUT/'release_transfer_plan.json').read_text());known=safety.load_known(OUT)
 for k in known:
  if k and k in token.encode():raise RuntimeError('unexpected credential source')
 status_path=OUT/'release_upload_status.json';receipts=OUT/'receipts';receipts.mkdir(mode=0o700,exist_ok=True)
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'Nina-resumable-backup','X-GitHub-Api-Version':'2022-11-28'}
 def api(path,method='GET',body=None):
  for attempt in range(3):
   try:
    r=requests.request(method,'https://api.github.com/'+path,headers=headers,json=body,timeout=(20,90),allow_redirects=False)
    if r.status_code>=500 or r.status_code==429:
     if attempt<2:time.sleep(3*(attempt+1));continue
    if r.status_code==404:return None
    if not 200<=r.status_code<300:raise RuntimeError('GitHub API HTTP '+str(r.status_code))
    return r.json() if r.content else {}
   except (requests.Timeout,requests.ConnectionError):
    if attempt==2:raise RuntimeError('GitHub API temporary network failure')
    time.sleep(3*(attempt+1))
 user=api('user')
 if not user or user.get('login','').lower()!='yifan-liu0715':raise RuntimeError('authenticated GitHub account differs from destination owner')
 repo=api('repos/'+REPO)
 if not repo or not repo.get('permissions',{}).get('push'):raise RuntimeError('destination write permission missing')
 release=api('repos/'+REPO+'/releases/tags/'+plan['release_tag'])
 if not release:raise RuntimeError('release must be created and pilot-verified first')
 url=release['upload_url'].split('{',1)[0]
 expected='https://uploads.github.com/repos/'+REPO+'/releases/'+str(release['id'])+'/assets'
 if url!=expected:raise RuntimeError('unapproved upload URL')
 existing={}
 for page in range(1,12):
  batch=api('repos/'+REPO+'/releases/'+str(release['id'])+'/assets?per_page=100&page='+str(page))
  if batch is None:raise RuntimeError('cannot list release assets')
  existing.update({a['name']:a for a in batch})
  if len(batch)<100:break
 results={};held=[];failures=[];active={};started=time.monotonic();started_at=now();base_logical=0
 for job in plan['jobs']:
  saved=receipts/(job['id']+'.json');asset=existing.get(job['asset_name'])
  if saved.exists() and asset:
   record=json.loads(saved.read_text())
   if asset.get('state')=='uploaded' and asset['size']==record['bytes'] and asset.get('digest')=='sha256:'+record['sha256']:
    results[job['id']]=record;base_logical+=record['logical_bytes']
 def status(state='uploading'):
  with LOCK:
   total=sum(r['logical_bytes'] for r in results.values());uploaded=sum(r['bytes'] for r in results.values())
   elapsed=max(time.monotonic()-started,0.001);rate=max(total-base_logical,0)/elapsed
   record={'state':state,'updated_at':now(),'started_at':started_at,'pid':os.getpid(),'repository':REPO,'release_url':release['html_url'],'planned_jobs':len(plan['jobs']),'planned_bytes':plan['source_bytes'],'uploaded_jobs':len(results),'uploaded_archive_bytes':uploaded,'uploaded_logical_bytes':total,'remaining_logical_bytes':plan['source_bytes']-total,'aggregate_logical_MB_per_second_since_start':round(rate/1e6,3),'estimated_remaining_transfer_hours':round((plan['source_bytes']-total)/rate/3600,2) if rate else None,'active_jobs':dict(active),'held':list(held),'failures':list(failures),'token_written_to_disk':False,'original_workspace_modified':False,'all_requested_content_uploaded':len(results)==len(plan['jobs'])}
   atomic(status_path,record)
 def monitor():
  while not stop.wait(15):status()
 stop=threading.Event();threading.Thread(target=monitor,daemon=True).start()
 def upload(job):
  with LOCK:active[job['id']]={'stage':'content_check','logical_bytes':job['logical_bytes'],'started_at':now()}
  status()
  try:
   safety.inspect_job(job,ROOT,OUT,known)
   with LOCK:active[job['id']]['stage']='preparing_archive'
   p,record=archive.produce(job)
   asset=existing.get(job['asset_name'])
   if asset and asset.get('state')=='uploaded':
    if asset.get('size')!=record['bytes'] or asset.get('digest')!='sha256:'+record['sha256']:raise safety.Held('existing asset differs from reconstructed source')
   else:
    for attempt in range(3):
     if attempt or asset:
      listed=api('repos/'+REPO+'/releases/'+str(release['id'])+'/assets?per_page=100') or []
      # Full asset list only on retry; avoid duplicate names left after interrupted uploads.
      for page in range(2,12):
       if len(listed)<(page-1)*100:break
       more=api('repos/'+REPO+'/releases/'+str(release['id'])+'/assets?per_page=100&page='+str(page)) or [];listed.extend(more)
       if len(more)<100:break
      found=next((a for a in listed if a['name']==job['asset_name']),None)
      if found and found.get('state')=='uploaded' and found.get('digest')=='sha256:'+record['sha256'] and found['size']==record['bytes']:asset=found;break
      if found:
       if found.get('state')!='starter':raise safety.Held('conflicting existing release asset')
       api('repos/'+REPO+'/releases/assets/'+str(found['id']),'DELETE')
     with LOCK:active[job['id']]['stage']='uploading';active[job['id']]['attempt']=attempt+1
     upload_headers={**headers,'Content-Type':'application/octet-stream','Content-Length':str(record['bytes'])}
     transfer_start=time.monotonic()
     try:
      with p.open('rb') as stream:response=requests.post(url,params={'name':job['asset_name']},headers=upload_headers,data=stream,timeout=(25,1800),allow_redirects=False)
      if response.status_code==201:asset=response.json();break
      if response.status_code not in (408,429,500,502,503,504,422):raise RuntimeError('GitHub upload HTTP '+str(response.status_code))
     except (requests.Timeout,requests.ConnectionError):pass
     if attempt==2:raise RuntimeError('release upload failed after three attempts')
     time.sleep(5*(attempt+1))
   if asset.get('state')!='uploaded' or asset['size']!=record['bytes']:raise RuntimeError('GitHub uploaded size or state mismatch')
   if asset.get('digest')!='sha256:'+record['sha256']:
    h=hashlib.sha256();size=0
    with requests.get(asset['browser_download_url'],stream=True,timeout=(25,1800)) as check:
     if check.status_code!=200:raise RuntimeError('asset download verification HTTP '+str(check.status_code))
     for b in check.iter_content(8*1024**2):h.update(b);size+=len(b)
    if h.hexdigest()!=record['sha256'] or size!=record['bytes']:raise RuntimeError('uploaded asset checksum mismatch')
   record.update(state='uploaded_verified',asset_id=asset['id'],download_url=asset['browser_download_url'],github_digest=asset.get('digest'),uploaded_at=now(),payload_safety_receipt='payload_safety/'+job['id']+'.json')
   atomic(receipts/(job['id']+'.json'),record)
   if p.resolve().parent!=(OUT/'chunks').resolve():raise RuntimeError('invalid generated chunk path')
   p.unlink()
   with LOCK:results[job['id']]=record
   emit({'id':job['id'],'state':'uploaded_verified','bytes':record['bytes'],'logical_bytes':record['logical_bytes']})
  except safety.Held as e:
   message=str(e)
   with LOCK:held.append({'id':job['id'],'logical_bytes':job['logical_bytes'],'reason':message})
   emit({'id':job['id'],'state':'held','reason':message})
  except Exception as e:
   # Never emit raw requests, credentials, model contents or response bodies.
   with LOCK:failures.append({'id':job['id'],'error_type':type(e).__name__})
   emit({'id':job['id'],'state':'failed','error_type':type(e).__name__})
  finally:
   with LOCK:active.pop(job['id'],None)
   status()
 # Numeric safetensor headers and media are independent of environment/history reviews.
 def priority(job):
  rows=job.get('files') or [job['file']];exts={pathlib.PurePosixPath(r['path']).suffix.lower() for r in rows}
  safe_media=exts.issubset(safety.MEDIA)
  weights=exts.issubset({'.safetensors'})
  return (0 if safe_media else 1 if weights else 2 if not job.get('needs_security_review') else 3,job['logical_bytes'])
 jobs=sorted([j for j in plan['jobs'] if j['id'] not in results],key=priority)
 status();emit({'state':'started','pid':os.getpid(),'workers':workers,'remaining_jobs':len(jobs),'resumed_verified_jobs':len(results)})
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(upload,j) for j in jobs]
  for future in concurrent.futures.as_completed(futures):future.result()
 stop.set();status('complete' if len(results)==len(plan['jobs']) else 'partial_requires_attention')
 emit({'state':'queue_finished','verified_jobs':len(results),'held':len(held),'failures':len(failures)})
 token=None;headers.clear()
if __name__=='__main__':
 try:main()
 except Exception as e:
  emit({'state':'fatal','error_type':type(e).__name__});raise SystemExit(1)
