"""Verify release assets and reconstruct the exported workspace without overwriting originals."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,pathlib,shutil,subprocess,tarfile,tempfile,urllib.request

def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def confined(root,name):
 rel=pathlib.PurePosixPath(name)
 if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('unsafe restore path')
 p=root.joinpath(*rel.parts)
 if not p.resolve().is_relative_to(root.resolve()):raise ValueError('restore path escapes destination')
 return p

def install_file(temp,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if target.is_symlink() or digest(target)!=digest(temp):raise RuntimeError('refusing to overwrite a different existing file: '+str(target))
  temp.unlink();return
 os.replace(temp,target)

def extract_tar(archive,destination,job):
 allowed={'workspace/'+r['path']:r for r in job['files']};seen=set()
 temp_root=destination/'.nina_restore_tmp';temp_root.mkdir(exist_ok=True,parents=True)
 process=subprocess.Popen(['zstd','-q','-dc',str(archive)],stdout=subprocess.PIPE)
 try:
  with tarfile.open(fileobj=process.stdout,mode='r|') as tf:
   for member in tf:
    if not member.isfile() or member.name not in allowed or member.name in seen:raise ValueError('unexpected archive entry')
    row=allowed[member.name]
    if member.size!=row['size']:raise ValueError('archive member size differs from plan')
    target=confined(destination,member.name)
    with tempfile.NamedTemporaryFile(dir=temp_root,delete=False) as out:
     temp=pathlib.Path(out.name);shutil.copyfileobj(tf.extractfile(member),out,8*1024*1024)
    install_file(temp,target);os.chmod(target,row['mode']&0o777);os.utime(target,ns=(row['mtime_ns'],row['mtime_ns']));seen.add(member.name)
 except BaseException:
  process.stdout.close()
  if process.poll() is None:process.terminate()
  process.wait();raise
 process.stdout.close()
 if process.wait()!=0 or seen!=set(allowed):raise RuntimeError('archive ended before the planned files were restored')
 return len(seen)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True,type=pathlib.Path);parser.add_argument('--metadata',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'metadata');parser.add_argument('--downloads',type=pathlib.Path);parser.add_argument('--download',action='store_true');args=parser.parse_args()
 destination=args.destination.resolve();destination.mkdir(exist_ok=True,parents=True)
 downloads=args.downloads or destination/'.release_assets';downloads.mkdir(exist_ok=True,parents=True)
 plan=json.loads((args.metadata/'release_transfer_plan.json').read_text(encoding='utf-8'));jobs={j['id']:j for j in plan['jobs']}
 receipt_dir=args.metadata/'release_receipts'
 if not receipt_dir.exists():raise SystemExit('No verified release receipts exist yet; payloads have not been uploaded.')
 verified={};parts=collections.defaultdict(list);restored=0
 for receipt_file in sorted(receipt_dir.glob('*.json')):
  receipt=json.loads(receipt_file.read_text(encoding='utf-8'));job=jobs[receipt['id']]
  if receipt.get('state')!='uploaded_verified':continue
  asset=confined(downloads,job['asset_name'])
  if not asset.exists() and args.download:
   partial=asset.with_suffix(asset.suffix+'.partial')
   with urllib.request.urlopen(receipt['download_url'],timeout=300) as src,partial.open('wb') as out:shutil.copyfileobj(src,out,8*1024*1024)
   if partial.stat().st_size!=receipt['bytes'] or digest(partial)!=receipt['sha256']:raise RuntimeError('download checksum mismatch')
   os.replace(partial,asset)
  if not asset.exists():print('MISSING_ASSET',job['asset_name']);continue
  if asset.stat().st_size!=receipt['bytes'] or digest(asset)!=receipt['sha256']:raise RuntimeError('asset checksum mismatch: '+job['asset_name'])
  if job['kind']=='tar_zstd':restored+=extract_tar(asset,destination,job)
  else:parts[job['file']['path']].append((job,asset))
  verified[job['id']]=receipt['sha256']
 for path,entries in parts.items():
  entries.sort(key=lambda e:e[0]['offset']);cursor=0
  for job,_ in entries:
   if job['offset']!=cursor:break
   cursor+=job['length']
  if cursor!=entries[0][0]['file']['size']:print('INCOMPLETE_SPLIT_FILE',path);continue
  temp_root=destination/'.nina_restore_tmp';temp_root.mkdir(exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=temp_root,delete=False) as out:
   temp=pathlib.Path(out.name)
   for job,asset in entries:
    if asset.stat().st_size!=job['length']:raise RuntimeError('split range size mismatch')
    with asset.open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
  install_file(temp,confined(destination,'workspace/'+path));restored+=1
 report={'verified_assets':len(verified),'planned_assets':len(jobs),'restored_files':restored,'complete':len(verified)==len(jobs),'note':'Text files are in Git under workspace/. Symlinks and empty directories are described in metadata; external symlink targets are never downloaded implicitly.'}
 (destination/'restore_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()