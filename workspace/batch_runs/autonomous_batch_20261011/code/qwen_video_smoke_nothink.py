import json, sys, time, torch
from transformers import AutoProcessor, AutoModelForImageTextToText
MODEL="/mnt/highspeed/models/Qwen3.5-9B"; VIDEO=sys.argv[1]
proc=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True)
model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
sysmsg=("You are a strict temporal video reviewer. Return ONLY one compact JSON object, no explanation and no chain of thought. "
"Keys: first_person_hand (boolean), animal_visible (boolean), action_count (integer), actions (array of objects with start_s,end_s,hand_action,animal_response,boundary_reason), response_tail_complete (boolean), semantic_state (review_needed, semantic_valid_short, accepted_for_training, or reject), reasons (array). "
"A valid action must show a camera-operator hand causing an animal response or verified non-response. Split independent gestures; keep one continuous causal interaction together. If uncertain use review_needed.")
messages=[{"role":"system","content":[{"type":"text","text":sysmsg}]},{"role":"user","content":[{"type":"video"},{"type":"text","text":"Review this entire video timeline."}]}]
try:
 text=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
except TypeError:
 text=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
print('textlen',len(text),flush=True)
inputs=proc(text=[text],videos=[VIDEO],return_tensors='pt',videos_kwargs={'fps':2,'do_sample_frames':True,'max_frames':32}).to(model.device)
print('inputids',inputs['input_ids'].shape,'video',inputs['pixel_values_videos'].shape,flush=True)
t=time.time()
with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=128,do_sample=False)
gen=out[:,inputs['input_ids'].shape[1]:]
ans=proc.batch_decode(gen,skip_special_tokens=True)[0]
print(json.dumps({'elapsed':round(time.time()-t,2),'raw':ans},ensure_ascii=False),flush=True)
