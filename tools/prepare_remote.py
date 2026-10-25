from __future__ import annotations
import os,sys,json,stat,hashlib,tarfile,subprocess,collections,time,pathlib
ROOT=pathlib.Path('/nvmedata/workspace2/users/nina')
OUT=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025')
OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
TEXT={'.py','.pyi','.sh','.bash','.zsh','.ps1','.bat','.cmd','.yaml','.yml','.toml','.ini','.cfg','.conf','.json','.jsonl','.csv','.tsv','.txt','.md','.rst','.html','.htm','.js','.jsx','.ts','.tsx','.css','.xml','.sql','.log','.out','.err','.tex','.ipynb','.c','.cc','.cpp','.h','.hpp','.cu','.cuh','.java','.go','.rs','.r','.lock','.map','.gitignore','.gitattributes','.gitmodules'}
SECRET_NAMES={'.netrc','.git-credentials','.credentials.json','credentials.json','auth.json','id_rsa','id_dsa','id_ecdsa','id_ed25519','id_ed25519_server','id_ed25519_teacher','credentials','token','tokens.json','access_tokens','stored_tokens'}
SECRET_DIRS={'.ssh','.aws','.azure','.gnupg','.kube'}
ARCHIVE_PARTS={'.git','__pycache__','.cache','node_modules','.pytest_cache','.mypy_cache','.venv','.venvs'}
PUBLIC_PREFIXES=('dataset_h3_public_v1/manifests/','dataset_h3_public_v1/reports/')
rows=[]; text=[]; payload=[]; excluded=[]; errors=[]; dirs=[]; links=[]
def log(obj):print(json.dumps(obj,ensure_ascii=True),flush=True)
def dumpjson(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def secret_path(rel):
 parts=pathlib.PurePosixPath(rel).parts;name=parts[-1].lower()
 return bool(set(parts)&SECRET_DIRS) or name in SECRET_NAMES or name=='.env' or name.startswith('.env.') or (name.endswith(('.p12','.pfx')))
def public_allowed(rel):
 ext=pathlib.PurePosixPath(rel).suffix.lower();name=pathlib.PurePosixPath(rel).name.lower()
 if ext not in {'.json','.jsonl','.csv','.tsv','.txt','.md','.yaml','.yml','.html'}: return False
 return rel.startswith(PUBLIC_PREFIXES) or any(s in name for s in ('manifest','screen','filter','select','review','audit','report','summary','status','readme'))
for dp,dn,fn in os.walk(ROOT,topdown=True,followlinks=False):
 dn.sort();fn.sort()
 for n in list(dn):
  p=pathlib.Path(dp)/n;rel=p.relative_to(ROOT).as_posix()
  try:st=p.lstat()
  except OSError as e:errors.append({'path':rel,'error':str(e)});dn.remove(n);continue
  if p.is_symlink():
   links.append({'path':rel,'type':'symlink','target':os.readlink(p),'size':st.st_size,'mode':st.st_mode,'mtime_ns':st.st_mtime_ns});dn.remove(n)
  else:dirs.append({'path':rel,'type':'directory','mode':st.st_mode,'mtime_ns':st.st_mtime_ns})
 for n in fn:
  p=pathlib.Path(dp)/n;rel=p.relative_to(ROOT).as_posix()
  try:st=p.lstat()
  except OSError as e:errors.append({'path':rel,'error':str(e)});continue
  row={'path':rel,'size':st.st_size,'mode':st.st_mode,'mtime_ns':st.st_mtime_ns,'type':'file'}
  if stat.S_ISLNK(st.st_mode):
   row.update(type='symlink',target=os.readlink(p));links.append(row);continue
  if not stat.S_ISREG(st.st_mode):
   row.update(type='special',status='excluded_runtime_socket_or_pipe');excluded.append(row);continue
  rows.append(row)
  if secret_path(rel):row['status']='excluded_credential_file';excluded.append(row);continue
  if rel.startswith('dataset_h3_public_v1/') and not public_allowed(rel):
   row['status']='excluded_public_dataset_payload';excluded.append(row);continue
  parts=pathlib.PurePosixPath(rel).parts;ext=p.suffix.lower()
  if '.git' in parts:
   row['status']='pending_nested_git_metadata_safety_scan';payload.append(row);continue
  if any(s in parts for s in ARCHIVE_PARTS) or parts[0] in {'conda_envs'}:
   row['status']='queued_environment_or_cache_archive';payload.append(row);continue
  istext=ext in TEXT or n in {'LICENSE','COPYING','NOTICE','Makefile','Dockerfile','requirements.txt','.gitignore','.gitattributes','.gitmodules'}
  if not istext and st.st_size<=2*1024*1024:
   try:
    b=p.open('rb').read(8192)
    istext=b'\0' not in b and bool(b.decode('utf-8',errors='ignore').strip()) and len(b.decode('utf-8',errors='ignore').encode())>len(b)*.95
   except OSError:pass
  if istext and st.st_size<90*1024*1024:
   row['status']='selected_text_pending_security_scan';text.append(row)
  else:row['status']='queued_payload_archive';payload.append(row)
  if len(rows)%20000==0:log({'inventory_files':len(rows),'text':len(text),'payload':len(payload),'excluded':len(excluded)})
for name,seq in [('inventory.jsonl',rows),('text_selection.jsonl',text),('payload_selection.jsonl',payload),('exclusions.jsonl',excluded),('directories.jsonl',dirs),('symlinks.jsonl',links),('inventory_errors.jsonl',errors)]:
 with (OUT/name).open('w',encoding='utf-8') as f:
  for r in seq:f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary={'root':str(ROOT),'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'files':len(rows),'bytes':sum(r['size'] for r in rows),'text_files':len(text),'text_bytes':sum(r['size'] for r in text),'payload_files':len(payload),'payload_bytes':sum(r['size'] for r in payload),'excluded_files':len(excluded),'excluded_bytes':sum(r['size'] for r in excluded),'symlinks':len(links),'directories':len(dirs),'errors':len(errors)}
dumpjson(OUT/'inventory_summary.json',summary);log(summary)
# The working-file source copy is staged separately; original files are never edited.
staging=OUT/'text_staging'/'workspace';staging.mkdir(parents=True,exist_ok=True)
manifest=[]
for i,row in enumerate(text):
 p=ROOT/row['path'];dest=staging/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
 try:
  st=p.stat();data=p.read_bytes();after=p.stat()
  if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns'] or st.st_size!=after.st_size or st.st_mtime_ns!=after.st_mtime_ns:raise RuntimeError('source changed after inventory')
  dest.write_bytes(data);dest.chmod(0o755 if st.st_mode&0o111 else 0o644)
  manifest.append({**row,'sha256':hashlib.sha256(data).hexdigest(),'repository_path':'workspace/'+row['path']})
 except Exception as e:errors.append({'path':row['path'],'phase':'copy_text','error':str(e)})
 if i%5000==0:log({'staged_text_files':i+1})
with (OUT/'text_export_manifest.jsonl').open('w',encoding='utf-8') as f:
 for r in manifest:f.write(json.dumps(r,ensure_ascii=False)+'\n')
dumpjson(OUT/'text_stage_summary.json',{'files':len(manifest),'bytes':sum(r['size'] for r in manifest),'errors':errors,'source_unchanged_checked':True})
log({'staged_text_complete':len(manifest),'errors':len(errors)})