"""Resumable SHA256-verified pilot uploader for autonomous_batch_20261011."""
import json,hashlib,subprocess,threading,queue,time,os,msvcrt
from pathlib import Path
D=Path(__file__).resolve().parent
M=json.loads((D/'full_manifest.json').read_text(encoding='utf-8'))
ROOT=M['remote_root']; HOST=M['remote_host']; PYTHON=M.get('remote_python','python3')
lockfile=(D/'upload.lock').open('a+b')
if (D/'upload.lock').stat().st_size==0: lockfile.write(b'0'); lockfile.flush()
lockfile.seek(0)
try: msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
except OSError: raise SystemExit('uploader already active')
(D/'upload.pid').write_text(str(os.getpid()))
mutex=threading.Lock(); states={}; started=time.time(); q=queue.Queue()
for s in M['sources']: q.put(s)
def emit(s,status,**more):
    with mutex:
        row=dict(source_id=s['source_id'],status=status,at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**more); states[s['source_id']]=row
        with (D/'upload_events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        snap=dict(pid=os.getpid(),updated_at=row['at'],total=M['source_count'],verified=sum(x['status']=='verified' for x in states.values()),failed=sum(x['status']=='failed' for x in states.values()),active=[x for x in states.values() if x['status']=='uploading'],verified_bytes=sum(x.get('size_bytes',0) for x in states.values() if x['status']=='verified'),elapsed_seconds=time.time()-started,remote_root=ROOT)
        t=D/'upload_status.json.tmp'; t.write_text(json.dumps(snap,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); t.replace(D/'upload_status.json')
        if status!='uploading': print(json.dumps(row,ensure_ascii=False),flush=True)
def start(worker):
    err=(D/f'ssh_worker_{worker}.log').open('ab')
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',HOST,f'{PYTHON} -u {ROOT}/receiver.py']
    return subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=1024*1024),err
def read(p):
    line=p.stdout.readline()
    if not line: raise ConnectionError('SSH receiver disconnected')
    a=json.loads(line)
    if a.get('status')=='error': raise RuntimeError(a['error'])
    return a
def stop(p,err):
    if p:
        try: p.stdin.close(); p.wait(timeout=10)
        except Exception: p.kill(); p.wait()
        p.stdout.close()
    if err: err.close()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''): h.update(c)
    return h.hexdigest()
def worker(w):
    p=None; err=None
    try:
        while True:
            try: s=q.get_nowait()
            except queue.Empty: break
            path=Path(s['source_path'])
            try:
                st=path.stat(); assert st.st_size==s['size_bytes'] and st.st_mtime_ns==s['mtime_ns'], 'local source changed'
                d=digest(path); assert d==s['expected_sha256'], 'local sha256 mismatch'
                for attempt in range(3):
                    try:
                        if p is None: p,err=start(w)
                        emit(s,'uploading',size_bytes=s['size_bytes'],attempt=attempt+1)
                        p.stdin.write((json.dumps({'source_id':s['source_id'],'sha256':d})+'\n').encode()); p.stdin.flush(); answer=read(p)
                        if answer['status']=='ready':
                            offset=answer['offset']; hh=hashlib.sha256()
                            with path.open('rb') as f:
                                left=offset
                                while left:
                                    c=f.read(min(4*1024*1024,left)); assert c; hh.update(c); left-=len(c)
                                assert hh.hexdigest()==answer['prefix_sha256'], 'partial prefix differs'
                                p.stdin.write(b'{"status":"prefix_verified"}\n'); p.stdin.flush()
                                for c in iter(lambda:f.read(1024*1024),b''): p.stdin.write(c)
                                p.stdin.flush()
                            answer=read(p)
                        assert answer['status'] in ['verified','verified_existing'], answer
                        emit(s,'verified',sha256=d,size_bytes=s['size_bytes'],remote_path=s['remote_source_path'],transfer_result=answer['status']); break
                    except Exception as e:
                        stop(p,err); p=None; err=None
                        emit(s,'retry' if attempt<2 else 'failed',error=repr(e),attempt=attempt+1)
                        if attempt<2: time.sleep(2)
            except Exception as e: emit(s,'failed',error=repr(e))
            finally: q.task_done()
    finally: stop(p,err)
threads=[threading.Thread(target=worker,args=(i,)) for i in range(M['workers'])]
for t in threads:t.start()
for t in threads:t.join()
final=dict(completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),all_verified=len(states)==M['source_count'] and all(s['status']=='verified' for s in states.values()),states=states)
(D/'upload_finished.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
raise SystemExit(0 if final['all_verified'] else 1)

