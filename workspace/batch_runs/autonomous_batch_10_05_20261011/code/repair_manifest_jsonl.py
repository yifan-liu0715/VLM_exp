import json,os
from pathlib import Path
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
p=R/'motion_proposal_manifest.jsonl'; text=p.read_text(encoding='utf-8')
parts=text.split('\\n')
rows=[json.loads(x) for x in parts if x.strip()]
latest={}
for r in rows: latest[(r.get('source_id'),r.get('event_index'))]=r
rows=list(latest.values()); rows.sort(key=lambda r:(str(r.get('source_id')),int(r.get('event_index') or 0)))
tmp=p.with_suffix('.tmp'); tmp.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8'); os.replace(tmp,p)
print('rows',len(rows))
for f in [R/'progress_motion_proposals.json',R/'reports'/'motion_proposal_export_summary.json']:
 try:
  d=json.loads(f.read_text(encoding='utf-8')); d['proposal_intervals']=len(rows); d['encoded_success']=sum(bool(r.get('encoded_format_verified')) for r in rows); d['encoded_failed']=sum(not bool(r.get('encoded_format_verified')) for r in rows); f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 except Exception as e: print('summary',e)
