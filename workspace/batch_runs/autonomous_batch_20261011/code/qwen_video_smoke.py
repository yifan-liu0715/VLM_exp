import json, os, sys, time
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "/mnt/highspeed/models/Qwen3.5-9B"
VIDEO = sys.argv[1]
FPS = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

print(json.dumps({"stage":"load_processor","model":MODEL}, ensure_ascii=False), flush=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print(json.dumps({"stage":"load_model"}, ensure_ascii=False), flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
)
model.eval()
print(json.dumps({"stage":"model_loaded","device":str(model.device)}, ensure_ascii=False), flush=True)

system = (
    "You are a strict temporal reviewer for a first-person hand-animal interaction dataset. "
    "Review the entire video sequence, not a single frame. The camera operator's hand must "
    "be the causal actor. A valid sample needs a readable hand action or stable hand condition, "
    "a visible target animal, and an observable response or verified non-response. Identify every "
    "independent hand action in order. Do not merge separate gestures such as rubbing the face, "
    "petting the head, and touching the nose. Do not split one continuous causal interaction. "
    "Return ONLY JSON with keys: first_person_hand, animal_visible, action_count, actions, "
    "response_tail_complete, semantic_state, reasons. Each action must have start_s, end_s, "
    "hand_action, animal_response, and boundary_reason. semantic_state must be one of "
    "review_needed, semantic_valid_short, accepted_for_training, reject. If uncertain, use review_needed."
)
text = (
    "Review this video candidate at the supplied temporal resolution. "
    "Pay special attention to action onset, hand-gesture changes, animal response onset, "
    "delayed response, and whether the candidate contains multiple actions."
)
messages = [
    {"role": "system", "content": [{"type": "text", "text": system}]},
    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]},
]
chat_text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(json.dumps({"stage":"prepare","chat_chars":len(chat_text),"video":VIDEO,"fps":FPS}, ensure_ascii=False), flush=True)

# The processor accepts a local path through its video input. Keep the frame budget bounded for the smoke test.
inputs = proc(
    text=[chat_text],
    videos=[VIDEO],
    return_tensors="pt",
    videos_kwargs={"fps": FPS, "do_sample_frames": True, "max_frames": 64},
)
inputs = inputs.to(model.device)
print(json.dumps({"stage":"inputs_ready","keys":list(inputs.keys()),"shapes":{k:(list(v.shape) if hasattr(v,"shape") else str(type(v))) for k,v in inputs.items()}}, ensure_ascii=False), flush=True)
started=time.time()
with torch.inference_mode():
    out=model.generate(**inputs, max_new_tokens=512, do_sample=False)
gen=out[:, inputs["input_ids"].shape[1]:]
answer=proc.batch_decode(gen, skip_special_tokens=True)[0]
print(json.dumps({"stage":"complete","elapsed_s":round(time.time()-started,2),"raw":answer}, ensure_ascii=False), flush=True)
