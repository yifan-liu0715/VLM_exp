import pathlib,json,re,hashlib,collections,tarfile,subprocess,time
OUT=pathlib.Path('/nvmedata/workspace2/users/nina_export_20261025');STAGE=OUT/'text_staging'
findings=json.loads((OUT/'security_findings.private.json').read_text())
def classify(r):
 p=str(pathlib.Path(r['File']).relative_to(OUT/'scan_shadow'));s=r.get('Secret','');m=r.get('Match','');rule=r['RuleID']
 if rule=='sourcegraph-access-token' and re.fullmatch('[0-9a-fA-F]{40}',s) and p.endswith('/.claude-plugin/marketplace.json'):return 'allow_commit_sha'
 if rule=='generic-api-key':
  if 'key_sha256' in m and re.fullmatch('[0-9a-fA-F]{64}',s) and p.endswith('/numpy/random/tests/test_generator_mt19937.py'):return 'allow_random_generator_test_hash'
  if p.endswith(('/onnx/reference/ops/aionnxml/op_dict_vectorizer.py','/onnxruntime/tools/symbolic_shape_infer.py','/google/protobuf/descriptor_pb2.py','/mmpose/datasets/datasets/body3d/h36m_dataset.py','/sat/tokenization/__init__.py','/pretrain_weights/ast_indexer')):return 'allow_code_identifier_false_positive'
 return 'redact_sensitive_value'
secrets=sorted({r.get('Secret','') for r in findings if classify(r)=='redact_sensitive_value' and r.get('Secret')},key=len,reverse=True)
changes=[];manifest=[];rows=[json.loads(l) for l in (OUT/'text_export_manifest.jsonl').read_text().splitlines() if l]
for r in rows:
 p=STAGE/r['repository_path'];b=p.read_bytes();new=b;occurrences=0
 for secret in secrets:
  raw=secret.encode();n=new.count(raw)
  if n:new=new.replace(raw,b'REDACTED_SENSITIVE_VALUE');occurrences+=n
 if new!=b:
  # Replace the file atomically: scan_shadow may hard-link the original staged copy.
  temp=p.with_name(p.name+'.redacting');temp.write_bytes(new);temp.chmod(p.stat().st_mode&0o777);temp.replace(p)
  changes.append({'repository_path':r['repository_path'],'redactions':occurrences,'original_sha256':r['sha256'],'exported_sha256':hashlib.sha256(new).hexdigest(),'original_bytes':len(b),'exported_bytes':len(new)})
 r['original_sha256']=r.pop('sha256');r['exported_sha256']=hashlib.sha256(new).hexdigest();r['exported_bytes']=len(new);r['status']='sanitized_text_export' if new!=b else 'text_export_unchanged';manifest.append(r)
with (OUT/'text_export_manifest.sanitized.jsonl').open('w') as f:
 for r in manifest:f.write(json.dumps(r,ensure_ascii=False)+'\n')
# No matched secret or source snippet is exported; only rule, file and disposition.
report={'scanner':'gitleaks 8.30.1','scanned_text_files':len(rows),'candidate_findings':len(findings),'dispositions':dict(collections.Counter(classify(r) for r in findings)),'distinct_sensitive_values_redacted':len(secrets),'changed_files':changes,'unhandled_findings':0,'original_workspace_modified':False,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'scan_method_file':'security_scan_method.json'}
(OUT/'security_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
# Verify every identified sensitive value is absent from every exported text file.
residual=[]
for r in manifest:
 b=(STAGE/r['repository_path']).read_bytes()
 if any(s.encode() in b for s in secrets):residual.append(r['repository_path'])
if residual:raise RuntimeError('redaction verification failed in '+str(len(residual))+' files')
report['identified_sensitive_values_residual_files']=0;(OUT/'security_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='changed_files'}));print(json.dumps({'changed_file_count':len(changes)}))
# Bundle the sanitized text snapshot and audit metadata for the Git staging checkout.
meta_names=['inventory.jsonl','inventory_summary.json','text_selection.jsonl','payload_selection.jsonl','exclusions.jsonl','symlinks.jsonl','directories.jsonl','inventory_errors.jsonl','text_export_manifest.sanitized.jsonl','security_report.json','security_scan_method.json']
package=OUT/'git_text_snapshot.tar.zst'
with package.open('wb') as dest:
 compressor=subprocess.Popen(['zstd','-q','-3','-T2','-c'],stdin=subprocess.PIPE,stdout=dest)
 try:
  with tarfile.open(fileobj=compressor.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tf:
   for r in manifest:tf.add(STAGE/r['repository_path'],arcname=r['repository_path'],recursive=False)
   for name in meta_names:tf.add(OUT/name,arcname='metadata/'+name,recursive=False)
 finally:compressor.stdin.close()
 if compressor.wait()!=0:raise RuntimeError('snapshot compression failed')
h=hashlib.sha256()
with package.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
receipt={'file':package.name,'bytes':package.stat().st_size,'sha256':h.hexdigest(),'text_files':len(manifest),'metadata_files':len(meta_names)}
(OUT/'git_text_snapshot.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))