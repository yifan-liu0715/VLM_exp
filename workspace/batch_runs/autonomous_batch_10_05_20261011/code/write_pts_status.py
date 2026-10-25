import json,os,datetime
from pathlib import Path
R=Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011')
ids=['iphone__video_20261005_154621','iphone__video_20261005_162646','iphone__video_20261005_160949','iphone__video_20261005_161507','iphone__video_20261005_163444','iphone__video_20261005_163059','iphone__video_20261005_155526','iphone__video_20261005_161300','iphone__video_20261005_164046','iphone__video_20261005_154858','iphone__video_20261005_164329']
rows=[]
for sid in ids:
 p=R/'reports'/'source_results'/(sid+'.json'); q=R/'reports'/'recovery_pts'/(sid+'.json'); prop=R/'proposals'/sid/'motion_proposals.json'
 d=json.loads(p.read_text()) if p.exists() else {}; qd=json.loads(q.read_text()) if q.exists() else {}; n=len(json.loads(prop.read_text())) if prop.exists() else 0
 if qd.get('analysis',{}).get('detector_status')=='ready' and n: status='recovered_pts_normalized_motion_proposals'
 elif qd.get('analysis',{}).get('detector_status')=='detector_failed': status='review_needed_detector_dependency_or_resource'
 elif qd.get('error') or d.get('error'): status='review_needed_normalized_decode_or_resource'
 else: status='review_needed'
 rows.append({'source_id':sid,'status':status,'proposal_count':n,'source_result_state':d.get('state'),'source_result_error':d.get('error'),'recovery_detector_status':qd.get('analysis',{}).get('detector_status'),'recovery_error':qd.get('error'),'original_source_unchanged':True,'keypoints_started':False,'training_started':False})
summary={'updated_at':datetime.datetime.now().astimezone().isoformat(),'total_anomaly_sources':len(rows),'recovered_sources':sum(r['status'].startswith('recovered') for r in rows),'review_needed_sources':sum(not r['status'].startswith('recovered') for r in rows),'proposal_count':sum(r['proposal_count'] for r in rows),'rows':rows}
(R/'reports'/'pts_recovery_status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md=['# PTS 异常源恢复状态','',f"更新时间：{summary['updated_at']}",'',f"已恢复并生成机械候选：{summary['recovered_sources']} / {summary['total_anomaly_sources']}",f"仍需人工或后续资源空闲时重跑：{summary['review_needed_sources']}",'','这些源文件原始内容未修改；未恢复的源不会被计入当前候选，也不会进入关键点或训练。','']
for r in rows: md.append(f"- `{r['source_id']}`：{r['status']}；候选 {r['proposal_count']} 条")
(R/'uncertain'/'pts_recovery_status.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
