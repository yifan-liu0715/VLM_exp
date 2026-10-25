from __future__ import annotations
import argparse, json, os, re, subprocess, time
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL='/mnt/highspeed/models/Qwen3.5-9B'

def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if p.returncode: raise RuntimeError(p.stderr[-2000:])
    return p.stdout

def source_duration(path):
    try:
        return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(path)]).strip())
    except Exception:
        return None

def parse_json(raw):
    """Return only a complete top-level review object; never an inner action object."""
    raw=raw.strip()
    raw=re.sub(r'<think>.*?</think>','',raw,flags=re.S).strip()

    def balanced_objects(text):
        found=[]
        for start in [m.start() for m in re.finditer(r'\{', text)]:
            depth=0; in_string=False; escaped=False
            for pos in range(start, len(text)):
                ch=text[pos]
                if in_string:
                    if escaped:
                        escaped=False
                    elif ch=='\\':
                        escaped=True
                    elif ch=='"':
                        in_string=False
                    continue
                if ch=='"':
                    in_string=True
                elif ch=='{':
                    depth += 1
                elif ch=='}':
                    depth -= 1
                    if depth==0:
                        found.append(text[start:pos+1])
                        break
        return found

    candidates=[]
    fence=re.search(r'```(?:json)?\s*(\{.*?\})\s*```',raw,flags=re.S|re.I)
    if fence:
        candidates.append(fence.group(1))
    if raw.startswith('{'):
        try:
            obj,end=json.JSONDecoder().raw_decode(raw)
            candidates.append(raw[:end])
        except Exception:
            pass
    candidates.extend(balanced_objects(raw))
    seen=set()
    for text in candidates:
        if text in seen:
            continue
        seen.add(text)
        try:
            obj=json.loads(text)
        except Exception:
            continue
        if not isinstance(obj,dict):
            continue
        # Require the review envelope. Inner action objects do not pass this gate.
        if 'actions' in obj and ('action_count' in obj or 'semantic_state' in obj) and ('first_person_hand' in obj or 'animal_visible' in obj):
            return obj
    return None

def make_context(source, start, end, out):
    src_d=source_duration(source) or end
    cstart=max(0.0,start-2.0)
    cend=min(src_d,end+2.0)
    # Keep model input bounded; long proposals are marked as partial context.
    if cend-cstart>24.0:
        cend=min(src_d,cstart+24.0)
    out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        run(['ffmpeg','-hide_banner','-loglevel','error','-ss',f'{cstart:.3f}','-i',str(source),'-t',f'{max(0.1,cend-cstart):.3f}',
             '-vf','scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=8',
             '-an','-c:v','libx264','-preset','ultrafast','-crf','25','-pix_fmt','yuv420p','-y',str(out)])
    return cstart,cend,src_d

def build_prompt(context_start, context_end, proposal_start, proposal_end):
    system=(
      'You are a strict temporal reviewer for a first-person hand-animal interaction dataset. '
      'Return ONLY one compact JSON object, no prose and no chain of thought. Review the entire supplied sequence. '
      'The camera operator hand must be the causal actor. Identify every independent action in order. '
      'Split when a new hand execution, contact region, target, delivery method, or independent animal outcome begins. '
      'Keep one continuous stroke/offer and its complete response together. Include delayed response and response tail. '
      'This is a sidecar proposal only; when uncertain use review_needed and never claim training acceptance. '
      'JSON keys: first_person_hand, animal_visible, action_count, actions, response_tail_complete, '
      'needs_split, semantic_state, reasons. Each action has start_s,end_s,hand_action,animal_response,boundary_reason. '
      'Times are relative to the supplied video. semantic_state must be review_needed, semantic_valid_short, or reject.'
    )
    user=(f'The supplied context covers source time {context_start:.3f} to {context_end:.3f} seconds. '
          f'The mechanical proposal is source time {proposal_start:.3f} to {proposal_end:.3f}. '
          'Review hand visibility, animal response, action count, and whether the proposal merges multiple gestures.')
    return [{"role":"system","content":[{"type":"text","text":system}]},
            {"role":"user","content":[{"type":"video"},{"type":"text","text":user}]}]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queue',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--worker',type=int,default=0); ap.add_argument('--workers',type=int,default=1); ap.add_argument('--limit',type=int,default=None); ap.add_argument('--offset',type=int,default=0); ap.add_argument('--max-new-tokens',type=int,default=320); ap.add_argument('--auto-device',action='store_true'); ap.add_argument('--gpu-memory',default='28GiB'); args=ap.parse_args()
    all_items=[json.loads(x) for x in args.queue.read_text(encoding='utf-8').splitlines() if x.strip()]
    begin=max(0,args.offset)
    end=(begin+args.limit) if args.limit is not None else None
    items=all_items[begin:end]
    indexed=[(i,x) for i,x in enumerate(items,start=begin) if i % args.workers == args.worker]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    reviews_path=args.out_dir/f'reviews_worker_{args.worker:02d}.jsonl'; decisions_path=args.out_dir/f'decisions_worker_{args.worker:02d}.jsonl'; progress_path=args.out_dir/f'progress_worker_{args.worker:02d}.json'
    done={}
    if reviews_path.exists():
        for line in reviews_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try: done[int(json.loads(line)['queue_index'])]=True
                except Exception: pass
    print(json.dumps({'stage':'load_processor','worker':args.worker,'assigned':len(indexed),'already_done':len(done)},ensure_ascii=False),flush=True)
    proc=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True)
    load_kwargs={'dtype':torch.bfloat16,'trust_remote_code':True}
    if args.auto_device:
        load_kwargs['device_map']='auto'; load_kwargs['max_memory']={0:args.gpu_memory,'cpu':'220GiB'}
    else:
        load_kwargs['device_map']='cuda:0'
    model=AutoModelForImageTextToText.from_pretrained(MODEL,**load_kwargs).eval()
    print(json.dumps({'stage':'model_loaded','worker':args.worker,'device':str(model.device)},ensure_ascii=False),flush=True)
    reviews_handle=reviews_path.open('a',encoding='utf-8'); decisions_handle=decisions_path.open('a',encoding='utf-8')
    for n,(idx,item) in enumerate(indexed,1):
        if idx in done: continue
        start=float(item.get('start_s') or 0.0); end=float(item.get('end_s') or start)
        record={'queue_index':idx,'source':item.get('source'),'file':item.get('file'),'source_start_s':start,'source_end_s':end,'backend_name':'qwen3.5-9b-local','backend_version':'transformers-5.14.1','review_status':'backend_failed'}
        t0=time.time(); context=None
        try:
            if not item.get('file'):
                raise RuntimeError('candidate_file_missing')
            source=Path(item['source']);
            if not source.exists(): raise RuntimeError('source_missing')
            context_dir=args.out_dir/'context'; context_path=context_dir/f'{idx:06d}.mp4'
            cstart,cend,sd=make_context(source,start,end,context_path)
            context={'context_path':str(context_path),'context_start_s':cstart,'context_end_s':cend,'source_duration_s':sd}
            messages=build_prompt(cstart,cend,start,end)
            try: text=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
            except TypeError: text=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
            inputs=proc(text=[text],videos=[str(context_path)],return_tensors='pt',videos_kwargs={'fps':2,'do_sample_frames':True,'max_frames':32}).to(model.device)
            with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            gen=out[:,inputs['input_ids'].shape[1]:]
            raw=proc.batch_decode(gen,skip_special_tokens=True)[0]
            parsed=parse_json(raw)
            record.update({'review_status':'completed' if parsed is not None else 'malformed','raw_response':raw,'parsed_review':parsed,'context':context,'elapsed_s':round(time.time()-t0,2)})
        except Exception as e:
            record.update({'error':str(e),'context':context,'elapsed_s':round(time.time()-t0,2)})
        # Fail closed: sidecar never promotes automatically.
        decision={'queue_index':idx,'source':item.get('source'),'file':item.get('file'),'start_s':start,'end_s':end,'backend_name':record['backend_name'],'backend_version':record['backend_version'],'semantic_review_path':str(reviews_path),'state':'semantic_backend_failed' if record['review_status'] in {'backend_failed','malformed'} else 'review_needed','status':'semantic_backend_failed' if record['review_status'] in {'backend_failed','malformed'} else 'review_needed','semantic_reviewed':False,'accepted_for_training':False,'training_ready':False,'promotion_status':'not_promoted','decision_reason':'sidecar_proposal_requires_deterministic_temporal_and_privacy_review'}
        reviews_handle.write(json.dumps(record,ensure_ascii=False)+'\n'); reviews_handle.flush(); decisions_handle.write(json.dumps(decision,ensure_ascii=False)+'\n'); decisions_handle.flush()
        progress_path.write_text(json.dumps({'worker':args.worker,'assigned':len(indexed),'completed':n,'remaining':len(indexed)-n,'last_queue_index':idx,'last_elapsed_s':record.get('elapsed_s'),'updated_at':time.time()},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'worker':args.worker,'done':n,'assigned':len(indexed),'queue_index':idx,'status':record['review_status'],'elapsed_s':record.get('elapsed_s')},ensure_ascii=False),flush=True)
    reviews_handle.close(); decisions_handle.close(); print(json.dumps({'worker':args.worker,'stage':'complete','assigned':len(indexed)},ensure_ascii=False),flush=True)
if __name__=='__main__': main()




