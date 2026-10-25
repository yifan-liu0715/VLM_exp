# Capture action-candidate batch

- Source videos: 282 DJI MP4 files from the connected G: drive.
- Source validation: 282/282 decoded by `ffprobe`; original files were not moved or modified.
- Candidate windows: 1,882 after IoU-based deduplication, from 264 source videos.
- Candidate duration: 4.0--6.0 seconds; 1,865 are exactly 5 seconds.
- Export format: H.264 MP4, 24 FPS, no audio; landscape 1280x720 and portrait 720x1280.
- `auto_candidate` means motion-based high-recall selection. It does not confirm the object, action semantics, response, causality, or hand visibility.
- `split=unassigned` is intentional. Assign train/validation/test only after source/session leakage checks and manual review.

The candidate windows can overlap in time. They are suitable for review and landmark extraction, but should not be counted as independent final actions without semantic annotation.
