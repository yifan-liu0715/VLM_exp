from __future__ import annotations
import json,re
from pathlib import Path

def strict_parse_json(raw):
    if not isinstance(raw,str): return None
    raw=re.sub(r'<think>.*?</think>','',raw.strip(),flags=re.S).strip()
    candidates=[]
    fence=re.search(r'```(?:json)?\s*(\{.*?\})\s*```',raw,flags=re.S|re.I)
    if fence: candidates.append(fence.group(1))
    if raw.startswith('{'):
        try:
            _,end=json.JSONDecoder().raw_decode(raw); candidates.append(raw[:end])
        except Exception: pass
    for start in [m.start() for m in re.finditer(r'\{',raw)]:
        depth=0; ins=False; esc=False
        for pos in range(start,len(raw)):
            ch=raw[pos]
            if ins:
                if esc: esc=False
                elif ch=='\\': esc=True
                elif ch=='"': ins=False
                continue
            if ch=='"': ins=True
            elif ch=='{': depth+=1
            elif ch=='}':
                depth-=1
                if depth==0: candidates.append(raw[start:pos+1]); break
    seen=set()
    for text in candidates:
        if text in seen: continue
        seen.add(text)
        try: obj=json.loads(text)
        except Exception: continue
        if isinstance(obj,dict) and 'actions' in obj and ('action_count' in obj or 'semantic_state' in obj) and ('first_person_hand' in obj or 'animal_visible' in obj): return obj
    return None

def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
base=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/review_artifacts/qwen_semantic_merged_20261012')
retry=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/review_artifacts/qwen_semantic_retry_v2_20261012')
out=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/review_artifacts/qwen_semantic_final_20261012')
out.mkdir(parents=True,exist_ok=True)
canonical={int(x['queue_index']):x for x in rows(base/'semantic_review.jsonl')}
retry_rows=rows(retry/'reviews_worker_00.jsonl')
for x in retry_rows: canonical[int(x['queue_index'])]=x
final=[]; decisions=[]; retry_queue=[]
for idx in range(597):
    if idx not in canonical: continue
    x=dict(canonical[idx])
    parsed=strict_parse_json(x.get('raw_response',''))
    x['parser_version']='strict_top_level_v1'
    x['parsed_review']=parsed
    x['review_status']='completed' if parsed is not None else 'malformed'
    if parsed is None:
        state='semantic_backend_failed'; reason='malformed_or_backend_failed_top_level_review_requires_human_or_backend_retry'
        retry_queue.append({'queue_index':idx,'source':x.get('source'),'file':x.get('file'),'start_s':x.get('source_start_s'),'end_s':x.get('source_end_s'),'reason':reason})
    else:
        state='review_needed'; reason='sidecar_proposal_requires_deterministic_temporal_privacy_and_human_review'
    final.append(x)
    decisions.append({'queue_index':idx,'source':x.get('source'),'file':x.get('file'),'start_s':x.get('source_start_s'),'end_s':x.get('source_end_s'),'backend_name':x.get('backend_name'),'backend_version':x.get('backend_version'),'semantic_review_path':str(out/'semantic_review.jsonl'),'state':state,'status':state,'semantic_reviewed':False,'accepted_for_training':False,'training_ready':False,'promotion_status':'not_promoted','decision_reason':reason})
final.sort(key=lambda x:x['queue_index']); decisions.sort(key=lambda x:x['queue_index']); retry_queue.sort(key=lambda x:x['queue_index'])
(out/'semantic_review.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in final)+'\n',encoding='utf-8')
(out/'semantic_decisions.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in decisions)+'\n',encoding='utf-8')
(out/'semantic_retry_queue_final.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in retry_queue)+('\n' if retry_queue else ''),encoding='utf-8')
indices=[int(x['queue_index']) for x in final]
from collections import Counter
statuses=Counter(x['review_status'] for x in final)
report={'expected':597,'canonical_rows':len(final),'completed_valid':statuses.get('completed',0),'malformed_retry':statuses.get('malformed',0),'missing_indices':[i for i in range(597) if i not in indices],'duplicate_indices':[i for i,c in Counter(indices).items() if c>1],'retry_input_rows':len(retry_rows),'retry_replaced_indices':sorted(int(x['queue_index']) for x in retry_rows),'state_counts':{'review_needed':statuses.get('completed',0),'semantic_backend_failed':statuses.get('malformed',0)},'promotion_performed':False,'keypoints_started':False,'training_started':False}
(out/'final_merge_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
