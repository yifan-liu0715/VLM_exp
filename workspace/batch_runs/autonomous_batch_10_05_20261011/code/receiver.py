"""Resumable, SHA-verified binary file receiver over authenticated SSH stdin."""
import sys,json,hashlib,os,shutil,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'manifest.json').read_text());sources={x['source_id']:x for x in M['sources']}
for sub in ['staging/raw','receipts','logs','source_context','jobs','encoded_review','trial_clips','clips']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
inp=sys.stdin.buffer;out=sys.stdout.buffer
def reply(x):out.write((json.dumps(x)+'\n').encode());out.flush()
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def receive(q):
    s=sources[q['source_id']];p=Path(s['remote_source_path']);expected=q['sha256'];size=int(s['size_bytes'])
    assert re.fullmatch('[a-f0-9]{64}',expected)
    receipt=ROOT/'receipts'/(s['source_id']+'.json')
    def finish():
        save(receipt,dict(source_id=s['source_id'],path=str(p),sha256=expected,size_bytes=size,verified=True,verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if p.exists():
        if p.stat().st_size!=size or digest(p)!=expected:raise RuntimeError('Existing source identity mismatch')
        finish();reply(dict(status='verified_existing'));return
    if s['reuse_pilot']:raise RuntimeError('Verified pilot source is missing')
    assert p.resolve().is_relative_to((ROOT/'staging/raw').resolve())
    if shutil.disk_usage(ROOT).free<size+M['min_free_bytes']:raise RuntimeError('Low disk space; retain reserve')
    part=p.with_name(p.name+'.part');meta=p.with_name(p.name+'.part.json')
    identity=dict(size=size,sha256=expected)
    if meta.exists() and json.loads(meta.read_text())!=identity:raise RuntimeError('Partial identity differs; retain for investigation')
    save(meta,identity)
    offset=part.stat().st_size if part.exists() else 0
    assert offset<=size
    reply(dict(status='ready',offset=offset,prefix_sha256=digest(part) if offset else hashlib.sha256(b'').hexdigest()))
    ack=json.loads(inp.readline());assert ack['status']=='prefix_verified'
    with part.open('ab') as f:
        remaining=size-offset
        while remaining:
            chunk=inp.read(min(1024*1024,remaining))
            if not chunk:raise EOFError('Transfer interrupted; partial retained')
            f.write(chunk);remaining-=len(chunk)
        f.flush();os.fsync(f.fileno())
    if digest(part)!=expected:raise RuntimeError('Transferred checksum mismatch; partial retained')
    part.replace(p);finish();reply(dict(status='verified',bytes_written=size-offset))
while True:
    line=inp.readline()
    if not line:break
    try:receive(json.loads(line))
    except Exception as e:
        reply(dict(status='error',error=repr(e)))
        break
