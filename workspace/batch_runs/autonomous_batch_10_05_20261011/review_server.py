from __future__ import annotations
import json, threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
BATCH=Path(__file__).resolve().parent
REVIEW=BATCH/'review'
DECISIONS=REVIEW/'review_decisions.jsonl'
PORT=18880
lock=threading.Lock()
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(BATCH),**kwargs)
    def log_message(self,fmt,*args): print(fmt%args,flush=True)
    def send_json(self,obj,code=200):
        b=json.dumps(obj,ensure_ascii=False).encode('utf-8'); self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.split('?',1)[0]=='/api/decisions':
            latest={}
            if DECISIONS.exists():
                for line in DECISIONS.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        try:
                            x=json.loads(line); latest[x['clip_id']]=x
                        except Exception: pass
            return self.send_json({'latest':latest,'count':len(latest)})
        return super().do_GET()
    def do_POST(self):
        if self.path.split('?',1)[0]!='/api/decision': self.send_error(404); return
        try:
            n=int(self.headers.get('Content-Length','0')); x=json.loads(self.rfile.read(n).decode('utf-8'))
            if not x.get('clip_id') or x.get('decision') not in {'keep','recut','delete','unsure'}: raise ValueError('invalid decision')
            with lock:
                with DECISIONS.open('a',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n')
            self.send_json({'ok':True})
        except Exception as e: self.send_json({'ok':False,'error':str(e)},400)
if __name__=='__main__':
    print(f'Review server: http://127.0.0.1:{PORT}/review/review.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
