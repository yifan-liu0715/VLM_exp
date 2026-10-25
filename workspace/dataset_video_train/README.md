# Standard dataset (JPEG)

Each sample = input video + output video + reference image (JPEG) + caption.
Paths in `metadata.jsonl`/`metadata.csv` are relative to this folder.

## Stats
- Samples: 3916
- Buckets (WxH): {'1280x704': 1225, '1280x512': 1032, '704x1280': 1659}
- Reference: 3642 AI-generated white-bg character refs (1024x1024) +
  274 video-keyframe fallbacks (flagged `reference_is_fallback`).

## Fields
sample_id, bucket, width, height, num_frames,
input_video, output_video, reference_image (JPEG q92),
caption, caption_model, reference_is_fallback

## Layout
```
metadata.jsonl
metadata.csv
{1280x512,1280x704,704x1280}/
  input_video/{sample_id}.mp4
  output_video/{sample_id}.mp4
  reference/{sample_id}.jpg
```
