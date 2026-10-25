import pathlib,json,os,signal,hashlib,time
out=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025')
pid=3707124
try:
 cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
 if b'gitleaks-linux\x00dir\x00' in cmd and str(out).encode() in cmd:os.kill(pid,signal.SIGTERM)
except FileNotFoundError:pass
source=out/'text_staging';dest=out/'scan_shadow';dest.mkdir(mode=0o700,exist_ok=True)
counts={'files':0,'json_string_views':0,'unchanged_text_views':0,'original_bytes':0,'scanned_bytes':0}
def leaves(obj,label='',seen=None):
 if seen is None:seen=set()
 if isinstance(obj,dict):
  for k,v in obj.items():
   if isinstance(k,str):
    s='field_name = '+k
    if s not in seen:seen.add(s);yield s
   yield from leaves(v,str(k),seen)
 elif isinstance(obj,list):
  for v in obj:yield from leaves(v,label,seen)
 elif isinstance(obj,str):
  s=label+' = '+obj
  if s not in seen:seen.add(s);yield s
for row in map(json.loads,(out/'text_export_manifest.jsonl').read_text(encoding='utf-8').splitlines()):
 rel=row['repository_path'];p=source/rel;target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
 counts['files']+=1;counts['original_bytes']+=p.stat().st_size
 try:
  if p.suffix.lower()=='.json':
   obj=json.loads(p.read_text(encoding='utf-8-sig'));data='\n'.join(leaves(obj)).encode('utf-8');target.write_bytes(data);counts['json_string_views']+=1
  elif p.suffix.lower()=='.jsonl':
   seen=set();lines=[]
   with p.open(encoding='utf-8-sig') as f:
    for line in f:
     if line.strip():lines.extend(leaves(json.loads(line),seen=seen))
   target.write_text('\n'.join(lines),encoding='utf-8');counts['json_string_views']+=1
  else:
   if not target.exists():os.link(p,target)
   counts['unchanged_text_views']+=1
 except (ValueError,UnicodeError):
  if not target.exists():os.link(p,target)
  counts['unchanged_text_views']+=1
 counts['scanned_bytes']+=target.stat().st_size
counts['method']='For parsed JSON/JSONL, every distinct string value and field name is scanned; numeric/null/boolean leaves are excluded. Other files are scanned verbatim. Original source and exported JSON content are unchanged.'
(out/'security_scan_method.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts))