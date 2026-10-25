"""Static payload checks. Never loads model checkpoints or executes source content."""
from __future__ import annotations
import collections,gzip,hashlib,io,json,os,pathlib,re,struct,subprocess,tarfile,tempfile,time,zipfile
CHUNK=4*1024**2
class Held(RuntimeError):pass
PATTERN=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255}|AKIA[0-9A-Z]{16}|-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,255})')
CREDENTIAL_NAMES={'.credentials.json','.netrc','.git-credentials','auth.json','id_rsa','id_ed25519','id_ecdsa','.env'}
ARCHIVES={'.zip','.whl','.egg','.tar','.gz','.tgz','.bz2','.xz','.zst'}
MEDIA={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.3gp','.jpg','.jpeg','.png','.gif','.bmp','.tiff','.tif','.webp','.wav','.mp3','.flac','.m4a','.pdf'}
TEXT={'.py','.pyi','.sh','.bash','.zsh','.ps1','.cmd','.bat','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.conf','.txt','.md','.html','.js','.jsx','.ts','.tsx','.c','.cc','.cpp','.h','.hpp','.cu','.cuh','.xml','.csv','.tsv','.log','.out','.err','.map'}

def check_bytes(data,known):
 if any(k in data for k in known if k) or PATTERN.search(data):raise Held('credential value or high-confidence credential pattern detected')

def scan_chunks(stream,known):
 h=hashlib.sha256();tail=b'';total=0
 overlap=max(4096,max((len(k) for k in known),default=0))
 while True:
  b=stream.read(CHUNK)
  if not b:break
  total+=len(b);h.update(b);check_bytes(tail+b,known);tail=(tail+b)[-overlap:]
 return {'bytes_scanned':total,'sha256':h.hexdigest()}

def check_source(row,root):
 rel=pathlib.PurePosixPath(row['path']);root=pathlib.Path(root).resolve()
 if rel.is_absolute() or '..' in rel.parts:raise Held('unsafe source path')
 p=root.joinpath(*rel.parts)
 if p.is_symlink() or not p.resolve().is_relative_to(root):raise Held('external or linked source')
 try:st=p.stat()
 except OSError:raise Held('source missing')
 if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:raise Held('source changed since inventory')
 return p

def archive_name(name):
 p=pathlib.PurePosixPath(name.replace('\\','/'))
 if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]):raise Held('unsafe archive member path')
 if p.name.lower() in CREDENTIAL_NAMES or p.name.lower().startswith('.env.'):raise Held('credential filename in archive')

def scan_archive(path,known,depth=0,view_dir=None):
 if depth>4:raise Held('nested archive depth limit reached')
 path=pathlib.Path(path)
 def content(name,stream,size):
  archive_name(name)
  ext=pathlib.PurePosixPath(name).suffix.lower()
  if ext in ARCHIVES:
   if size>16*1024**3:raise Held('large nested archive requires separate inspection')
   temp_path=None
   try:
    with tempfile.NamedTemporaryFile(suffix=ext,dir=view_dir,delete=False) as temp:
     temp_path=pathlib.Path(temp.name)
     while True:
      b=stream.read(CHUNK)
      if not b:break
      temp.write(b)
    scan_archive(temp_path,known,depth+1,view_dir)
   finally:
    if temp_path is not None:temp_path.unlink(missing_ok=True)
  else:
   # Torch tensor storage is numeric; inspect archive names plus all pickle/metadata members.
   if re.fullmatch(r'(?:[^/]+/)?(?:data|byteorder)/[0-9]+',name):return
   if ext in TEXT|{'.pkl','.pickle'} and view_dir is not None:
    target=pathlib.Path(view_dir)/(hashlib.sha256((str(path)+'!'+name).encode()).hexdigest()+'.txt')
    with target.open('wb') as dst:
     tail=b''
     while True:
      b=stream.read(CHUNK)
      if not b:break
      check_bytes(tail+b,known);tail=(tail+b)[-4096:];dst.write(b)
   else:scan_chunks(stream,known)
 if zipfile.is_zipfile(path):
  with zipfile.ZipFile(path) as z:
   members=z.infolist()
   if len(members)>500000 or sum(m.file_size for m in members)>200*1024**3:raise Held('archive expansion limit reached')
   for member in members:
    archive_name(member.filename)
    if member.flag_bits&1:raise Held('encrypted archive requires separate inspection')
    if member.is_dir():continue
    if (member.external_attr>>16)&0o170000==0o120000:raise Held('symlink member requires separate inspection')
    with z.open(member) as f:content(member.filename,f,member.file_size)
 else:
  try:
   with tarfile.open(path,mode='r|*') as archive:
    count=0;expanded=0
    for member in archive:
     count+=1;expanded+=member.size;archive_name(member.name)
     if count>500000 or expanded>200*1024**3:raise Held('archive expansion limit reached')
     if member.isdir():continue
     if not member.isfile():raise Held('nonregular archive member requires separate inspection')
     with archive.extractfile(member) as f:content(member.name,f,member.size)
  except tarfile.ReadError:
   if path.suffix.lower()=='.gz':
    with gzip.open(path,'rb') as f:scan_chunks(f,known)
   else:raise Held('unsupported compressed archive requires separate inspection')
 return {'archive_content_verified':True}

def scan_safetensors(path,known):
 p=pathlib.Path(path)
 with p.open('rb') as f:
  raw=f.read(8)
  if len(raw)!=8:raise Held('invalid safetensors prefix')
  length=struct.unpack('<Q',raw)[0]
  if not 2<=length<=100*1024**2 or length+8>p.stat().st_size:raise Held('invalid safetensors header length')
  header=f.read(length);check_bytes(header,known)
 try:data=json.loads(header)
 except (ValueError,UnicodeError):raise Held('invalid safetensors header')
 if not isinstance(data,dict):raise Held('invalid safetensors metadata')
 available=p.stat().st_size-8-length
 for name,info in data.items():
  if name=='__metadata__':continue
  if not isinstance(info,dict):raise Held('invalid safetensors tensor descriptor')
  offsets=info.get('data_offsets')
  if not isinstance(offsets,list) or len(offsets)!=2 or not all(isinstance(v,int) for v in offsets) or not 0<=offsets[0]<=offsets[1]<=available:raise Held('invalid safetensors tensor range')
 return header

def load_known(out):
 findings=json.loads((out/'security_findings.private.json').read_text())
 def sensitive(r):
  p=r.get('File','');s=r.get('Secret','');m=r.get('Match','');rule=r.get('RuleID')
  if rule=='sourcegraph-access-token' and re.fullmatch('[0-9a-fA-F]{40}',s) and p.endswith('/.claude-plugin/marketplace.json'):return False
  if rule=='generic-api-key':
   if 'key_sha256' in m and re.fullmatch('[0-9a-fA-F]{64}',s) and p.endswith('/numpy/random/tests/test_generator_mt19937.py'):return False
   if p.endswith(('/onnx/reference/ops/aionnxml/op_dict_vectorizer.py','/onnxruntime/tools/symbolic_shape_infer.py','/google/protobuf/descriptor_pb2.py','/mmpose/datasets/datasets/body3d/h36m_dataset.py','/sat/tokenization/__init__.py','/pretrain_weights/ast_indexer')):return False
  return bool(s)
 return list({r['Secret'].encode() for r in findings if sensitive(r)})

def inspect_job(job,root,out,known):
 safety=out/'payload_safety';safety.mkdir(mode=0o700,exist_ok=True)
 receipt=safety/(job['id']+'.json');rows=job.get('files') or [job['file']]
 for row in rows:check_source(row,root)
 if receipt.exists():
  saved=json.loads(receipt.read_text())
  if saved.get('state')=='verified' and saved.get('scanner_version')==1:return saved
 if job.get('needs_security_review'):raise Held('original nested Git history requires independent history scan')
 views=safety/(job['id']+'.views');views.mkdir(mode=0o700,exist_ok=True)
 checked=[]
 for row in rows:
  p=check_source(row,root);ext=p.suffix.lower();key=hashlib.sha256((row['path']+str(row['size'])+str(row['mtime_ns'])).encode()).hexdigest();cached=safety/(key+'.file.json')
  if cached.exists():continue
  view=views/(key+'.txt')
  if p.name.lower() in CREDENTIAL_NAMES or p.name.lower().startswith('.env.'):raise Held('credential filename requires exclusion')
  if ext=='.safetensors':
   view.write_bytes(scan_safetensors(p,known));method='safetensors structure and complete JSON header; tensor bytes excluded as numeric'
  elif ext in ARCHIVES or (ext in {'.pt','.pth','.ckpt','.bin'} and zipfile.is_zipfile(p)):
   scan_archive(p,known,view_dir=views);method='recursive archive member scan without executing pickle or checkpoints'
  else:
   with p.open('rb') as stream:result=scan_chunks(stream,known)
   method='full-file known-secret and high-confidence credential-pattern scan'
   if ext in TEXT or ext not in MEDIA and row['size']<16*1024**2:
    with p.open('rb') as stream:data=stream.read(8192)
    if b'\0' not in data:
     if not view.exists():os.link(p,view)
    elif ext in {'.pyc','.pkl','.pickle'}:
     with p.open('rb') as stream:binary=stream.read()
     view.write_bytes(b'\n'.join(re.findall(rb'[\x20-\x7e]{8,}',binary)))
  check_source(row,root)
  checked.append((cached,{'path':row['path'],'size':row['size'],'mtime_ns':row['mtime_ns'],'method':method}))
 # All views, including complete archive text members, are scanned with the same vetted Gitleaks binary.
 private=safety/(job['id']+'.findings.private.json');log=safety/(job['id']+'.gitleaks.private.log')
 command=[str(out/'gitleaks-linux'),'dir','--no-banner','--no-color','--ignore-gitleaks-allow','--max-decode-depth','3','--report-format','json','--report-path',str(private),str(views)]
 env=dict(os.environ,GOMAXPROCS='4')
 with log.open('wb') as sink:run=subprocess.run(command,stdout=sink,stderr=sink,env=env,timeout=600)
 candidates=json.loads(private.read_text()) if private.exists() else []
 # Keep unknown candidates held for explicit classification; never print secret values.
 unresolved=[];allowed=0
 for r in candidates:
  secret=r.get('Secret','');match=r.get('Match','')
  if r.get('RuleID')=='generic-api-key' and 'key_sha256' in match and re.fullmatch('[0-9a-fA-F]{64}',secret):allowed+=1
  elif r.get('RuleID')=='sourcegraph-access-token' and re.fullmatch('[0-9a-fA-F]{40}',secret) and secret in {v.get('Secret') for v in json.loads((out/'security_findings.private.json').read_text()) if v.get('RuleID')=='sourcegraph-access-token' and str(v.get('File','')).endswith('/.claude-plugin/marketplace.json')}:allowed+=1
  else:unresolved.append({'rule':r.get('RuleID'),'view':pathlib.Path(r.get('File','')).name,'line':r.get('StartLine')})
 if run.returncode not in (0,1):raise Held('payload secret scanner failed')
 if unresolved:
  receipt.write_text(json.dumps({'state':'held','scanner_version':1,'id':job['id'],'candidate_count':len(unresolved),'candidates':unresolved},indent=2));raise Held('payload has '+str(len(unresolved))+' unclassified secret candidates')
 for path,record in checked:path.write_text(json.dumps(record))
 record={'state':'verified','scanner_version':1,'id':job['id'],'source_files':len(rows),'logical_bytes':job['logical_bytes'],'known_sensitive_values_checked':len(known),'gitleaks_candidates':len(candidates),'classified_false_positives':allowed,'original_workspace_modified':False,'verified_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
 receipt.write_text(json.dumps(record,indent=2))
 # Remove only generated shadow links/files, never sources.
 import shutil
 if views.parent.resolve()!=safety.resolve():raise RuntimeError('invalid scan-shadow cleanup')
 shutil.rmtree(views)
 return record
