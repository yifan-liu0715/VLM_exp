"""Publish the planned full snapshot to Releases. Requires one GitHub CLI login."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,pathlib,subprocess,sys,threading,time,urllib.request,urllib.error
BASE=pathlib.Path(__file__).resolve().parents[1]
META=BASE/'metadata';REPO='yifan-liu0715/VLM_exp';LOCK=threading.Lock()

def write_atomic(path,record):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def api(token,path,method='GET',data=None):
    request=urllib.request.Request('https://api.github.com/'+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'VLM-exp-workspace-export','X-GitHub-Api-Version':'2022-11-28'},method=method,data=json.dumps(data).encode() if data is not None else None)
    with urllib.request.urlopen(request,timeout=60) as r:return json.load(r)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--max-jobs',type=int);args=parser.parse_args()
    if not 1<=args.workers<=4:raise SystemExit('workers must be between 1 and 4')
    plan=json.loads((META/'release_transfer_plan.json').read_text(encoding='utf-8'));status_path=META/'release_upload_status.json'
    login=subprocess.run(['gh','auth','token','--hostname','github.com'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    token=login.stdout.strip()
    if login.returncode or not token:
        write_atomic(status_path,{'state':'requires_github_api_login','planned_jobs':len(plan['jobs']),'planned_bytes':plan['source_bytes'],'uploaded_jobs':0,'uploaded_bytes':0,'command':'gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key','note':'No archive upload was attempted. SSH Git authentication alone cannot call the Releases API.'})
        print('GitHub CLI login is required. No archive upload attempted.');raise SystemExit(2)
    me=api(token,'user')
    if me.get('login','').lower()!='yifan-liu0715':raise SystemExit('GitHub CLI must be logged in as yifan-liu0715')
    repo=api(token,'repos/'+REPO)
    if not repo.get('permissions',{}).get('push'):raise SystemExit('Repository write permission is required')
    try:release=api(token,'repos/'+REPO+'/releases/tags/'+plan['release_tag'])
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        release=api(token,'repos/'+REPO+'/releases','POST',{'tag_name':plan['release_tag'],'target_commitish':'main','name':'Nina workspace snapshot 2026-10-25','body':'Full workspace snapshot. See repository metadata for per-file coverage, exclusions, SHA-256 receipts, source categories and restore instructions. Assets are added in resumable batches.','draft':False,'prerelease':False})
    upload_url=release['upload_url'].split('{',1)[0]
    receipts=META/'release_receipts';receipts.mkdir(exist_ok=True)
    jobs=[j for j in plan['jobs'] if not j.get('needs_security_review')]
    held=[j['id'] for j in plan['jobs'] if j.get('needs_security_review')]
    if args.max_jobs:jobs=jobs[:args.max_jobs]
    # Repository and API URLs contain no token; only stdin carries the token to the server.
    existing={}
    for page in range(1,20):
        entries=api(token,f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100&page={page}')
        existing.update({entry['name']:entry for entry in entries})
        if len(entries)<100:break
    results={};failures=[]
    def status(state='uploading'):
        write_atomic(status_path,{'state':state,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'release_url':release['html_url'],'planned_jobs':len(plan['jobs']),'selected_jobs':len(jobs),'uploaded_jobs':len(results),'uploaded_archive_bytes':sum(r.get('bytes',0) for r in results.values()),'uploaded_logical_bytes':sum(r.get('logical_bytes',0) for r in results.values()),'held_for_security_review':held,'failures':failures})
    def worker(job):
        receipt=receipts/(job['id']+'.json');asset=existing.get(job['asset_name'])
        if receipt.exists() and asset:
            record=json.loads(receipt.read_text())
            if asset.get('state')=='uploaded' and asset.get('size')==record['bytes'] and asset.get('digest')=='sha256:'+record['sha256']:return record
        if asset:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','h3-server','python3 '+plan['remote_export_root']+'/upload_archive_remote.py'],input=json.dumps({'action':'receipt','job_id':job['id']}),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if p.returncode:raise RuntimeError('Existing release asset has no matching verified receipt: '+job['asset_name'])
            record=json.loads(p.stdout)
            if asset.get('size')!=record['bytes'] or asset.get('digest')!='sha256:'+record['sha256']:raise RuntimeError('Existing asset checksum differs from receipt')
            write_atomic(receipt,record);return record
        request={'action':'upload','job_id':job['id'],'upload_url':upload_url,'token':token}
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=20','h3-server','python3 '+plan['remote_export_root']+'/upload_archive_remote.py']
        p=subprocess.run(command,input=json.dumps(request),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        lines=p.stdout.splitlines()
        record=json.loads(lines[-1]) if lines else {'state':'failed','error':'worker returned no status'}
        if p.returncode or record.get('state')!='uploaded_verified':raise RuntimeError(job['id']+': '+str(record.get('error','upload failed')))
        write_atomic(receipt,record);return record
    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,j):j for j in jobs}
        for future in concurrent.futures.as_completed(pending):
            job=pending[future]
            try:
                record=future.result();results[job['id']]=record;print(json.dumps({'id':job['id'],'state':'uploaded_verified','archive_bytes':record['bytes']}),flush=True)
            except Exception as e:
                failures.append({'id':job['id'],'error':str(e)})
                # Stop queued work on the first hard failure; in-flight uploads finish.
                for f in pending:f.cancel()
            status()
    state='complete' if not failures and not held and len(results)==len(plan['jobs']) else 'partial_requires_attention'
    status(state);print(json.dumps({'state':state,'verified_jobs':len(results),'failures':len(failures),'held':len(held)}))
    if state!='complete':raise SystemExit(1)
if __name__=='__main__':main()