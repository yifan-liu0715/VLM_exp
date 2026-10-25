import pathlib,tempfile,importlib.util,json,hashlib,subprocess,tarfile,io,os
BASE=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025')
def load(name):
 spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
u=load('upload_archive_remote');r=load('restore_release');b=load('build_release_plan');checks=[]
with tempfile.TemporaryDirectory(prefix='verify_export_',dir=BASE) as d:
 home=pathlib.Path(d);root=home/'source';root.mkdir();out=home/'export';out.mkdir();dest=home/'restored';dest.mkdir()
 source=root/'example'/'clip.bin';source.parent.mkdir();data=bytes(range(256))*4096+os.urandom(32768);source.write_bytes(data)
 st=source.stat();row={'path':'example/clip.bin','size':st.st_size,'mode':st.st_mode,'mtime_ns':st.st_mtime_ns,'status':'test'}
 job={'id':'test-tar','kind':'tar_zstd','category':'test','asset_name':'test.tar.zst','files':[row],'logical_bytes':len(data),'needs_security_review':False}
 p,receipt=u.produce(job,root,out);assert u.sha(p)==receipt['sha256'];assert p.stat().st_size==receipt['bytes'];assert r.extract_tar(p,dest,job)==1;assert (dest/'workspace'/'example'/'clip.bin').read_bytes()==data;checks.append('compressed archive roundtrip and SHA-256')
 assert source.read_bytes()==data;checks.append('source unchanged')
 chunks=[]
 for offset in range(0,len(data),300000):
  j={'id':'slice-'+str(offset),'kind':'file_slice','asset_name':'slice-'+str(offset)+'.bin','file':row,'offset':offset,'length':min(300000,len(data)-offset),'logical_bytes':min(300000,len(data)-offset),'needs_security_review':False};p,rec=u.produce(j,root,out);assert p.stat().st_size==j['length'];chunks.append(p.read_bytes())
 assert b''.join(chunks)==data;checks.append('split-file range roundtrip')
 for bad in ['../escape','/etc/passwd']:
  try:u.safe_source({**row,'path':bad},root);raise AssertionError('unsafe path accepted')
  except ValueError:pass
 checks.append('traversal and absolute path rejection')
 (root/'external').symlink_to('/etc/passwd')
 try:u.safe_source({**row,'path':'external'},root);raise AssertionError('external symlink accepted')
 except ValueError:pass
 checks.append('external symlink rejection')
 temp=home/'replacement';temp.write_bytes(b'wrong');target=dest/'workspace'/'example'/'clip.bin'
 try:r.install_file(temp,target);raise AssertionError('existing different file overwritten')
 except RuntimeError:pass
 assert target.read_bytes()==data;checks.append('restore preserves existing different file')
 source.write_bytes(data+b'changed')
 try:u.safe_source(row,root);raise AssertionError('changed source accepted')
 except RuntimeError:pass
 checks.append('changed source rejection')
 rows=[{'path':'prj/model.bin','size':11*1024**3,'mode':33188,'mtime_ns':1},{'path':'conda_envs/pkg/a.py','size':1,'mode':33188,'mtime_ns':1}]
 jobs=b.build(rows);assert sum(j['logical_bytes'] for j in jobs)==sum(x['size'] for x in rows);assert all(j.get('length',0)<2*1024**3 for j in jobs);assert len({j['asset_name'] for j in jobs})==len(jobs);checks.append('archive plan conserves byte ranges and asset names')
report={'passed':len(checks),'checks':checks,'failures':0,'network_upload_performed':False}
(BASE/'export_tools_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))