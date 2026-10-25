"""Remote archive worker. Receives API credentials only through stdin, never files/arguments."""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,sys,tarfile,time
ROOT=pathlib.Path('/nvmedata/workspace2/users/nina')
OUT=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025')
MAX_ASSET=2*1024**3

def safe_source(row,root=ROOT):
    relative=pathlib.PurePosixPath(row['path'])
    if relative.is_absolute() or any(p in ('..','') for p in relative.parts):raise ValueError('unsafe source path')
    p=root.joinpath(*relative.parts)
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('source escapes workspace')
    st=p.stat()
    if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('source changed since inventory: '+row['path'])
    return p

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def produce(job,root=ROOT,out=OUT):
    if job.get('needs_security_review'):raise RuntimeError('nested Git history requires an independent secret scan before export')
    name=job['asset_name']
    if name!=pathlib.PurePath(name).name or not re.fullmatch('[A-Za-z0-9_.-]+',name):raise ValueError('unsafe asset name')
    chunks=out/'chunks';chunks.mkdir(mode=0o700,exist_ok=True,parents=True)
    p=chunks/name;receipt=chunks/(name+'.prepared.json')
    sources=job['files'] if job['kind']=='tar_zstd' else [job['file']]
    for row in sources:safe_source(row,root)
    if p.exists() and receipt.exists():
        record=json.loads(receipt.read_text())
        if record['bytes']==p.stat().st_size and record['sha256']==sha(p):return p,record
    part=p.with_suffix(p.suffix+'.partial')
    if job['kind']=='file_slice':
        row=job['file'];source=safe_source(row,root)
        offset=job['offset'];length=job['length']
        if offset<0 or length<0 or offset+length>row['size']:raise ValueError('invalid file range')
        with source.open('rb') as src,part.open('wb') as dst:
            src.seek(offset);remaining=length
            while remaining:
                b=src.read(min(8*1024*1024,remaining))
                if not b:raise EOFError('source ended inside file range')
                dst.write(b);remaining-=len(b)
    elif job['kind']=='tar_zstd':
        with part.open('wb') as dst:
            compressor=subprocess.Popen(['zstd','-q','-3','-T2','-c'],stdin=subprocess.PIPE,stdout=dst)
            try:
                with tarfile.open(fileobj=compressor.stdin,mode='w|',format=tarfile.PAX_FORMAT,dereference=True) as tf:
                    for row in job['files']:
                        source=safe_source(row,root);info=tf.gettarinfo(str(source),arcname='workspace/'+row['path'])
                        if not info.isfile():raise ValueError('only regular files are allowed in an archive')
                        with source.open('rb') as src:tf.addfile(info,src)
                        safe_source(row,root)
            finally:compressor.stdin.close()
            if compressor.wait()!=0:raise RuntimeError('zstd compression failed')
    else:raise ValueError('unknown archive kind')
    for row in sources:safe_source(row,root)
    if part.stat().st_size>=MAX_ASSET:raise RuntimeError('asset exceeds GitHub release limit')
    digest=sha(part);os.replace(part,p)
    record={'id':job['id'],'asset_name':name,'kind':job['kind'],'bytes':p.stat().st_size,'sha256':digest,'logical_bytes':job['logical_bytes'],'source_stat_verified':True}
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
    return p,record

def main():
    # Never echo this request: it may include a GitHub API token.
    request=json.load(sys.stdin)
    plan=json.loads((OUT/'release_transfer_plan.json').read_text(encoding='utf-8'))
    matches=[j for j in plan['jobs'] if j['id']==request['job_id']]
    if len(matches)!=1:raise ValueError('unknown archive id')
    job=matches[0]
    if request.get('action')=='receipt':
        saved=OUT/'receipts'/(job['id']+'.json')
        if not saved.exists():raise RuntimeError('verified remote receipt not found')
        print(saved.read_text(encoding='utf-8'));return
    p,record=produce(job)
    if request.get('action')=='prepare':print(json.dumps(record));return
    upload_url=request['upload_url']
    if not re.fullmatch(r'https://uploads\.github\.com/repos/yifan-liu0715/VLM_exp/releases/[0-9]+/assets',upload_url):raise ValueError('unapproved upload destination')
    import requests
    token=request.pop('token');headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/octet-stream','Content-Length':str(p.stat().st_size),'User-Agent':'VLM-exp-workspace-export'}
    with p.open('rb') as f:
        response=requests.post(upload_url,params={'name':job['asset_name']},headers=headers,data=f,timeout=(30,3600),allow_redirects=False)
    del token;headers.clear()
    if response.status_code!=201:raise RuntimeError('GitHub release upload HTTP '+str(response.status_code))
    asset=response.json()
    if asset.get('size')!=record['bytes'] or asset.get('state')!='uploaded':raise RuntimeError('GitHub uploaded size/state mismatch')
    server_digest=asset.get('digest')
    if server_digest!='sha256:'+record['sha256']:
        # Current GitHub returns a server SHA-256. Without it, verify by downloading.
        h=hashlib.sha256();size=0
        with requests.get(asset['browser_download_url'],stream=True,timeout=(30,3600)) as check:
            check.raise_for_status()
            for b in check.iter_content(8*1024*1024):h.update(b);size+=len(b)
        if size!=record['bytes'] or h.hexdigest()!=record['sha256']:raise RuntimeError('download verification mismatch')
    record.update(state='uploaded_verified',asset_id=asset['id'],download_url=asset['browser_download_url'],github_digest=server_digest,uploaded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    receipts=OUT/'receipts';receipts.mkdir(mode=0o700,exist_ok=True)
    (receipts/(job['id']+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    # Only generated copies in OUT/chunks can be removed, after remote verification.
    if p.resolve().parent!=(OUT/'chunks').resolve():raise RuntimeError('invalid temporary archive cleanup path')
    p.unlink()
    print(json.dumps(record))
if __name__=='__main__':
    try:main()
    except Exception as e:
        # Do not print requests, authorization headers, response bodies or credentials.
        print(json.dumps({'state':'failed','error_type':type(e).__name__,'error':str(e)}));raise SystemExit(1)