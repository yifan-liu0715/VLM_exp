"""Build a complete bounded GitHub Release transfer plan; does not upload data."""
from __future__ import annotations
import collections, hashlib, json, pathlib, time
BASE=pathlib.Path(__file__).resolve().parents[1]
META=BASE/'metadata'
LIMIT=1536*1024*1024
GROUP_LIMIT=1024*1024*1024
MAX_FILES=4000

def category(path):
    root=path.split('/',1)[0]
    if '/.git/' in '/'+path:return 'git_history_needs_security_review'
    if root in ('conda_envs','envs'):return 'environments'
    if root=='prj':return 'projects_training_and_tests'
    if root=='batch_runs':return 'batch_raw_pending_and_results'
    if root.startswith('dataset_'):return root
    if root in ('claude-config','tmp'):return 'logs_and_tool_state'
    return 'other_workspace_files'

def build(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[category(row['path'])].append(row)
    jobs=[]
    def flush(cat,group):
        if not group:return
        key=hashlib.sha256(('\n'.join(r['path'] for r in group)).encode()).hexdigest()[:16]
        jobs.append({'id':f'{cat}-{key}','kind':'tar_zstd','category':cat,'asset_name':f'{cat}-{key}.tar.zst','files':group,'logical_bytes':sum(r['size'] for r in group),'needs_security_review':cat=='git_history_needs_security_review'})
    for cat,items in sorted(groups.items()):
        group=[];size=0
        for row in sorted(items,key=lambda r:r['path']):
            if row['size']>GROUP_LIMIT:
                flush(cat,group);group=[];size=0
                for offset in range(0,row['size'],LIMIT):
                    length=min(LIMIT,row['size']-offset)
                    key=hashlib.sha256((row['path']+':'+str(offset)).encode()).hexdigest()[:16]
                    jobs.append({'id':f'{cat}-{key}','kind':'file_slice','category':cat,'asset_name':f'{cat}-{key}.bin','file':row,'offset':offset,'length':length,'logical_bytes':length,'needs_security_review':cat=='git_history_needs_security_review'})
            else:
                if group and (size+row['size']>GROUP_LIMIT or len(group)>=MAX_FILES):flush(cat,group);group=[];size=0
                group.append(row);size+=row['size']
        flush(cat,group)
    # Put irreproducible capture data before downloaded base weights or tool caches.
    def priority(j):
        c=j['category']
        if c.startswith('dataset_'):return (0,c,j['id'])
        if c=='batch_raw_pending_and_results':return (1,c,j['id'])
        if c=='projects_training_and_tests':return (2,c,j['id'])
        if c=='git_history_needs_security_review':return (9,c,j['id'])
        return (3,c,j['id'])
    return sorted(jobs,key=priority)

def main():
    rows=[json.loads(line) for line in (META/'payload_selection.jsonl').read_text(encoding='utf-8').splitlines() if line]
    jobs=build(rows)
    assert len({j['asset_name'] for j in jobs})==len(jobs)
    assert sum(j['logical_bytes'] for j in jobs)==sum(r['size'] for r in rows)
    assert all(j.get('length',0)<2*1024**3 for j in jobs)
    plan={'version':1,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'remote_root':'/nvmedata/workspace2/users/nina','remote_export_root':'/nvmedata/workspace2/users/nina_export_20261025','repository':'yifan-liu0715/VLM_exp','release_tag':'nina-workspace-20261025','source_files':len(rows),'source_bytes':sum(r['size'] for r in rows),'job_count':len(jobs),'upload_state':'pending_github_api_authentication','payload_safety_status':'pending_archive_content_scan','credential_policy':'Never include tokens, private keys, credential files, or unreviewed Git history. No token is written to the plan.','jobs':jobs}
    (META/'release_transfer_plan.json').write_text(json.dumps(plan,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    counts=collections.Counter(j['category'] for j in jobs)
    print(json.dumps({'jobs':len(jobs),'files':len(rows),'bytes':plan['source_bytes'],'categories':counts},ensure_ascii=False))
if __name__=='__main__':main()