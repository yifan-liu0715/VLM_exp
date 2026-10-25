"""Publish verified receipts/status to Git while the independent upload process runs."""
from __future__ import annotations
import json,os,pathlib,sys,time
import requests
import payload_safety
OUT=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025')
REPO='yifan-liu0715/VLM_exp'
def main():
 request=json.load(sys.stdin);token=request.pop('token');known=payload_safety.load_known(OUT)
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'Nina-backup-receipts'}
 plan=json.loads((OUT/'release_transfer_plan.json').read_text());ids={j['id'] for j in plan['jobs']}
 def api(path,method='GET',body=None):
  r=requests.request(method,'https://api.github.com/repos/'+REPO+'/'+path,headers=headers,json=body,timeout=(20,90),allow_redirects=False)
  if not 200<=r.status_code<300:raise RuntimeError('GitHub metadata API HTTP '+str(r.status_code))
  return r.json()
 last_count=-1;last_sync=0;errors=0
 while True:
  try:
   status=json.loads((OUT/'release_upload_status.json').read_text());pid=int((OUT/'bulk_upload.pid').read_text());cmd=pathlib.Path('/proc')/str(pid)/'cmdline'
   running=cmd.exists() and b'remote_bulk_release.py' in cmd.read_bytes()
   receipts={}
   for p in (OUT/'receipts').glob('*.json'):
    if p.stem in ids:
     data=json.loads(p.read_text())
     if data.get('state')=='uploaded_verified':receipts[p.name]=data
   count=len(receipts);elapsed=time.monotonic()-last_sync
   if count-last_count>=10 or elapsed>=180 or not running or status.get('state')!='uploading':
    files={'metadata/release_upload_status.json':json.dumps(status,ensure_ascii=False,indent=2)}
    for name,data in receipts.items():files['metadata/release_receipts/'+name]=json.dumps(data,ensure_ascii=False,indent=2)
    for p in (OUT/'payload_safety').glob('*.json'):
     if p.stem in ids:
      data=json.loads(p.read_text())
      if data.get('state')=='verified':files['metadata/payload_safety/'+p.name]=json.dumps(data,indent=2)
    for name in ('payload_safety.py','remote_bulk_release.py','sync_release_metadata_remote.py','test_payload_safety.py'):
     files['tools/'+name]=(OUT/name).read_text()
    sync={'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'uploader_running':running,'verified_receipts':count,'note':'Progress is a snapshot; only uploaded_verified receipts confirm preserved payloads. The original full plan is not marked verified globally.'}
    files['metadata/release_metadata_sync.json']=json.dumps(sync,indent=2)
    for text in files.values():payload_safety.check_bytes(text.encode(),known)
    head=api('commits/main')
    tree=api('git/trees','POST',{'base_tree':head['commit']['tree']['sha'],'tree':[{'path':name,'mode':'100644','type':'blob','content':data} for name,data in files.items()]})
    commit=api('git/commits','POST',{'message':'Save verified Release receipts: '+str(count)+'/'+str(len(ids)),'tree':tree['sha'],'parents':[head['sha']]})
    api('git/refs/heads/main','PATCH',{'sha':commit['sha'],'force':False})
    (OUT/'release_metadata_sync.json').write_text(json.dumps({**sync,'commit':commit['sha']},indent=2))
    print(json.dumps({'state':'metadata_published','verified_receipts':count,'commit':commit['sha']}),flush=True)
    last_count=count;last_sync=time.monotonic();errors=0
   if not running or status.get('state')!='uploading':break
  except Exception as e:
   errors+=1;print(json.dumps({'state':'metadata_sync_retry','error_type':type(e).__name__,'attempt':errors}),flush=True)
   if errors>=12:raise RuntimeError('metadata synchronization repeatedly failed')
  time.sleep(15)
 headers.clear();token=None
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'state':'metadata_sync_failed','error_type':type(e).__name__}),flush=True);raise SystemExit(1)
