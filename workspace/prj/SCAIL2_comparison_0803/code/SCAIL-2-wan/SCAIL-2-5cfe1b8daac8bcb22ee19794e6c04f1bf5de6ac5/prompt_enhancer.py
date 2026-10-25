#!/usr/bin/env python3
import argparse
import mimetypes
import tempfile
import os
from pathlib import Path


VIDEO_CAPTION_PROMPT = """You are captioning sampled frames from a source video for a character replacement video generation task.

Describe the source video in one detailed English paragraph. Focus on:
- the scene, location, lighting, camera framing, and background;
- the action, motion, timing, and camera movement across the sampled frames;
- the clothing, pose, body motion, and nearby objects touched or interacted with by the person/character being replaced.

If the user specifies who should be replaced, identify that source subject clearly in the caption. Pay special attention to the source subject's clothing and any objects they hold, touch, operate, sit on, stand near, or otherwise interact with, because those details help locate the replacement region.

Do not mention the replacement target image. Do not invent an identity for the replacement target.
Output only the source-video caption.
"""


REPLACEMENT_PROMPT_TEMPLATE = """You are a prompt enhancer for SCAIL-2 character replacement.

Your task is to write one detailed English description of the final replaced video. This is not an editing instruction. The output must describe the video after replacement has already happened: the replacement character from the reference image is performing the source subject's motion in the source scene.

Replacement instruction from user:
{instruction}

Source video caption:
{caption}

Few-shot examples of the desired prompt style:
{examples}

Rules:
1. Output a positive video-generation prompt describing the replaced video itself. Do not output wording like "replace X with Y", "swap", "edit", or "the task is".
2. Remove the original source subject's identity and appearance. Keep only the original subject's motion, pose, timing, spatial position, and interaction with the scene.
3. The final prompt for SCAIL-2 should describe the replacement character's visible clothing and appearance in enough detail, using the reference image as the source of identity and wardrobe details.
4. The final prompt should also describe important objects the character interacts with or stays close to in the source video, such as tools, instruments, furniture, vehicles, doors, tables, handheld items, or work surfaces.
5. Keep the original video environment, lighting, camera angle, shot scale, background objects, and motion trajectory.
6. If the source caption mentions the original subject's clothing only to locate body regions or interactions, translate those grounding details into the replacement character's final appearance instead of preserving the original identity.
7. Use natural video wording with concrete verbs. Avoid mentioning masks, segmentation, editing software, Gemini, or the prompt generation process.
8. Output only the final enhanced prompt, in one English paragraph, around 90-140 words.
"""


def _check_file(path: str, name: str) -> Path:
    if path is None:
        raise ValueError(f"Please specify {name}.")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{name} does not exist: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {file_path}")
    return file_path


def _read_examples(path: str | None, max_chars: int) -> str:
    if path is None:
        return ""
    file_path = _check_file(path, "prompt examples")
    text = file_path.read_text(encoding="utf-8").strip()
    return text[:max_chars]


def _guess_mime(path: Path, fallback: str) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or fallback


def extract_video_frames(video_path: Path, num_frames: int, image_format: str = "jpg") -> list[bytes]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine frame count for source video: {video_path}")

    num_frames = max(1, min(num_frames, frame_count))
    if num_frames == 1:
        indices = [frame_count // 2]
    else:
        indices = [round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]

    ext = ".jpg" if image_format.lower() in ("jpg", "jpeg") else ".png"
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92] if ext == ".jpg" else []
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, encoded = cv2.imencode(ext, frame, encode_params)
        if ok:
            frames.append(encoded.tobytes())
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to extract frames from source video: {video_path}")
    return frames


def _make_client(api_key: str | None):
    try:
        from google import genai
        import google.genai.types as gtypes
    except ImportError as exc:
        raise ImportError(
            "Missing Gemini SDK. Install it with: pip install google-genai"
        ) from exc

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or pass --api_key.")
    base_url = os.environ.get("GEMINI_BASE_URL")
    http_opts = gtypes.HttpOptions(baseUrl=base_url) if base_url else None
    return genai.Client(api_key=api_key, http_options=http_opts)


def _generate_text(client, model: str, contents, temperature: float) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError(f"Gemini returned an empty response: {response}")
    return text.strip()


def caption_video(
    client,
    model: str,
    video_path: Path,
    instruction: str,
    temperature: float,
    num_frames: int,
) -> str:
    import google.genai.types as gtypes

    frame_bytes = extract_video_frames(video_path, num_frames=num_frames)
    prompt = (
        f"{VIDEO_CAPTION_PROMPT}\n\n"
        f"User replacement instruction: {instruction}\n"
        f"The following images are {len(frame_bytes)} sampled frames in chronological order."
    )
    parts = [
        gtypes.Part(
            inline_data=gtypes.Blob(data=data, mime_type="image/jpeg")
        )
        for data in frame_bytes
    ]
    parts.append(gtypes.Part(text=prompt))
    contents = gtypes.Content(parts=parts)
    return _generate_text(client, model, contents, temperature)


def enhance_prompt(
    client,
    model: str,
    image_path: Path,
    instruction: str,
    caption: str,
    examples: str,
    temperature: float,
) -> str:
    import google.genai.types as gtypes

    image_bytes = image_path.read_bytes()
    prompt = REPLACEMENT_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        caption=caption.strip(),
        examples=examples.strip() or "(No examples provided.)",
    )
    contents = gtypes.Content(
        parts=[
            gtypes.Part(
                inline_data=gtypes.Blob(
                    data=image_bytes,
                    mime_type=_guess_mime(image_path, "image/jpeg"),
                )
            ),
            gtypes.Part(text=prompt),
        ]
    )
    return _generate_text(client, model, contents, temperature)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhance a character replacement prompt with Gemini."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Source/original video for Gemini to caption.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Reference image of the replacement character.",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help='User replacement instruction, e.g. "replace the man with the person in the image".',
    )
    parser.add_argument(
        "--examples",
        default="prompt_examples.txt",
        help="Few-shot prompt examples. Default: prompt_examples.txt",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name. Default: gemini-3-flash-preview",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Gemini sampling temperature.",
    )
    parser.add_argument(
        "--max_example_chars",
        type=int,
        default=4000,
        help="Maximum characters loaded from the few-shot examples file.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of source-video frames sampled for Gemini captioning.",
    )
    parser.add_argument(
        "--caption_out",
        default=None,
        help="Optional path to save the intermediate source video caption.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the enhanced prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = _check_file(args.video, "source video")
    image_path = _check_file(args.image, "replacement image")
    examples = _read_examples(args.examples, args.max_example_chars)

    client = _make_client(args.api_key)

    caption = caption_video(
        client=client,
        model=args.model,
        video_path=video_path,
        instruction=args.instruction,
        temperature=args.temperature,
        num_frames=args.num_frames,
    )
    if args.caption_out:
        Path(args.caption_out).write_text(caption + "\n", encoding="utf-8")

    enhanced_prompt = enhance_prompt(
        client=client,
        model=args.model,
        image_path=image_path,
        instruction=args.instruction,
        caption=caption,
        examples=examples,
        temperature=args.temperature,
    )
    if args.output:
        Path(args.output).write_text(enhanced_prompt + "\n", encoding="utf-8")
    print(enhanced_prompt)


if __name__ == "__main__":
    main()
