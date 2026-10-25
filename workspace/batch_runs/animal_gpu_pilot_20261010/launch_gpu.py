"""Use consistent bundled CUDA libraries, without changing the shared environment."""
import os,sys,subprocess,json,time
from pathlib import Path
r=Path(__file__).resolve().parent
env=os.environ.copy()
env['CUDA_VISIBLE_DEVICES']='GPU-1672e605-3a89-298b-9bec-f6315bc324d8'
env['PYTHONPATH']=str(r/'deps')
env['OMP_NUM_THREADS']='2'
libs=sorted((Path(sys.prefix)/'lib/python3.11/site-packages/nvidia').glob('*/lib'))
env['LD_LIBRARY_PATH']=':'.join(map(str,libs))+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
before=subprocess.run(['nvidia-smi','--id=3','--query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version','--format=csv'],capture_output=True,text=True,check=True).stdout
(r/'reports/launch_metadata.json').write_text(json.dumps(dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],library_paths=list(map(str,libs)),gpu_before=before),indent=2))
suffix='_unprofiled' if '--no-profile' in sys.argv else ''
if '--decoder' in sys.argv:suffix+='_'+sys.argv[sys.argv.index('--decoder')+1]
with (r/('reports/remote_gpu_run'+suffix+'.log')).open('w') as log:
    p=subprocess.Popen([sys.executable,str(r/'benchmark_faces.py'),'--mode','remote_gpu']+sys.argv[1:],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in p.stdout:
        log.write(line);log.flush();print(line,end='',flush=True)
    code=p.wait()
raise SystemExit(code)
