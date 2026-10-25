import json,os
from pathlib import Path
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
for f in [R/'progress_motion_proposals.json',R/'reports'/'motion_proposal_export_summary.json']:
 t=f.read_text(encoding='utf-8').strip()
 if t.endswith('\\n'): t=t[:-2]
 d=json.loads(t); f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(f, 'ok')
