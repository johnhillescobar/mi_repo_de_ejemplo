import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# Load .env next to this script, then cwd (so IDE shells see updates without restarting the app).
_script_dir = Path(__file__).resolve().parent
load_dotenv(_script_dir / ".env")
load_dotenv()

# ---------- Config ----------
MODEL = "gpt-4o-mini-tts"
DEFAULT_INPUT_JSON = _script_dir / "autocomplete_society_podcast.json"
MAX_CHARS_PER_CHUNK = 3500
TEMP_DIR_NAME = "tts_temp_json_chunks"
IMAGE_DIR_NAME = "image"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MP3_DIR_NAME = "mp3_files"
MP4_DIR_NAME = "mp4_files"
# ---------------------------

client = OpenAI()


def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split a long speaker turn without mixing it with another speaker."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= max_chars:
                    current += (" " if current else "") + sentence
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
            continue

        if len(current) + len(para) + 2 <= max_chars:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return chunks


def generate_speech_chunk(text: str, voice: str, output_path: Path) -> None:
    """Generate one MP3 chunk using the voice assigned to this dialogue turn."""
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=voice,
        input=text,
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)


def _ffmpeg_executable() -> str | None:
    """Resolve ffmpeg: FFMPEG_PATH (.env ok), PATH, then a common Windows install location."""
    raw = os.environ.get("FFMPEG_PATH", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return str(p.resolve())
    found = shutil.which("ffmpeg")
    if found:
        return found
    if sys.platform == "win32":
        common = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
        if common.is_file():
            return str(common.resolve())
    return None


def combine_mp3_files(mp3_files: list[Path], output_file: Path) -> None:
    """Combine generated turn/chunk MP3 files in conversation order."""
    ffmpeg_bin = _ffmpeg_executable()
    if ffmpeg_bin:
        concat_file = output_file.parent / "concat_list.txt"
        try:
            with open(concat_file, "w", encoding="utf-8") as f:
                for mp3 in mp3_files:
                    f.write(f"file '{mp3.resolve().as_posix()}'\n")

            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c",
                    "copy",
                    str(output_file),
                ],
                check=True,
            )
        finally:
            concat_file.unlink(missing_ok=True)
    else:
        print(
            "ffmpeg not found (PATH may be stale in this terminal; restart Cursor, or set "
            "FFMPEG_PATH in .env). Joining chunk MP3s as raw bytes."
        )
        with open(output_file, "wb") as out:
            for mp3 in mp3_files:
                out.write(mp3.read_bytes())


def _first_image_in_dir(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths[0] if paths else None


def resolve_background_image(image_arg: Path | None, default_dir: Path) -> Path | None:
    """
    Explicit --image wins (file, or directory to pick first image).
    Otherwise the first sorted image under default_dir (project image/).
    """
    if image_arg is not None:
        cand = image_arg.expanduser()
        cand = cand.resolve() if cand.is_absolute() else (Path.cwd() / cand).resolve()
        if cand.is_file():
            return cand if cand.suffix.lower() in IMAGE_EXTENSIONS else None
        if cand.is_dir():
            return _first_image_in_dir(cand)
        return None
    return _first_image_in_dir(default_dir)


def mux_still_image_mp4(
    ffmpeg_bin: str, image: Path, audio_mp3: Path, output_mp4: Path
) -> None:
    """
    Still image + MP3 to MP4 (H.264/AAC when possible).
    Tries encoders in order: libx264, libopenh264, then mpeg4.
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    image_s = str(image.resolve())
    audio_s = str(audio_mp3.resolve())
    out_s = str(output_mp4.resolve())

    attempts: list[tuple[str, list[str]]] = [
        ("libx264", ["-c:v", "libx264", "-tune", "stillimage"]),
        ("libopenh264", ["-c:v", "libopenh264", "-b:v", "2500k"]),
        ("mpeg4", ["-c:v", "mpeg4", "-q:v", "5"]),
    ]
    head = [
        ffmpeg_bin,
        "-y",
        "-loop",
        "1",
        "-framerate",
        "1",
        "-i",
        image_s,
        "-i",
        audio_s,
    ]

    for i, (name, vflags) in enumerate(attempts):
        if i > 0:
            print(f"No `{attempts[i - 1][0]}` encoder; retrying MP4 with `{name}`...")
        tail = [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-movflags",
            "+faststart",
            out_s,
        ]
        proc = subprocess.run([*head, *vflags, *tail])
        if proc.returncode == 0:
            return

    raise RuntimeError(
        "ffmpeg could not mux MP4 after trying: "
        + ", ".join(n for n, _ in attempts)
        + ". Prefer a ffmpeg build that includes libx264."
    )


def _load_dialogue(json_path: Path) -> tuple[str, list[dict[str, str]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    title = str(data.get("title") or json_path.stem)
    raw_dialogue = data.get("dialogue")
    if not isinstance(raw_dialogue, list):
        raise ValueError("JSON must contain a top-level `dialogue` list.")

    dialogue: list[dict[str, str]] = []
    for index, item in enumerate(raw_dialogue, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dialogue item {index} must be an object.")
        text = _string_field(item, "text", index)
        voice = _string_field(item, "voice", index)
        speaker = str(item.get("speaker") or f"speaker_{index}")
        if not text.strip():
            continue
        dialogue.append({"speaker": speaker, "voice": voice, "text": text.strip()})

    if not dialogue:
        raise ValueError("No dialogue turns with text found in JSON.")

    return title, dialogue


def _string_field(item: dict[str, Any], field: str, index: int) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dialogue item {index} must have a non-empty `{field}` string.")
    return value.strip()


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return stem or "podcast"


def _remove_temp_dir(temp_dir: Path) -> None:
    """Best-effort cleanup for Windows, where ffmpeg/AV can briefly hold files."""
    if not temp_dir.exists():
        return

    for attempt in range(5):
        try:
            shutil.rmtree(temp_dir)
            return
        except PermissionError as exc:
            if attempt == 4:
                print(
                    f"Warning: could not remove temp folder {temp_dir}. "
                    f"It may still be locked by another process: {exc}"
                )
                return
            time.sleep(0.25 * (attempt + 1))


def main() -> None:
    default_image_dir = _script_dir / IMAGE_DIR_NAME
    mp3_dir = _script_dir / MP3_DIR_NAME
    mp4_dir = _script_dir / MP4_DIR_NAME

    parser = argparse.ArgumentParser(
        description="Convert multi-voice podcast JSON to MP3; optionally mux a still image to MP4.",
    )
    parser.add_argument(
        "input_json",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_JSON,
        help=f"Podcast JSON file (default: {DEFAULT_INPUT_JSON.name})",
    )
    parser.add_argument(
        "output_mp3",
        type=Path,
        nargs="?",
        default=None,
        help=f"Output MP3 path (default: {MP3_DIR_NAME}/<title>.mp3)",
    )
    parser.add_argument(
        "--mp4",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"MP4 output path (default: {MP4_DIR_NAME}/<mp3 stem>.mp4 if an image is found)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Still image file or folder (default: first image in project {IMAGE_DIR_NAME}/)",
    )
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="Do not produce MP4 even if a background image is available",
    )

    args = parser.parse_args()
    input_json = args.input_json
    if not input_json.exists():
        print(f"File not found: {input_json}")
        sys.exit(1)

    title, dialogue = _load_dialogue(input_json)
    output_mp3 = (
        args.output_mp3
        if args.output_mp3 is not None
        else mp3_dir / f"{_safe_stem(title)}.mp3"
    )
    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = input_json.parent / TEMP_DIR_NAME
    _remove_temp_dir(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    mp3_files: list[Path] = []
    planned_chunks = sum(len(split_text(turn["text"])) for turn in dialogue)
    print(f"Found {len(dialogue)} dialogue turn(s), {planned_chunks} audio chunk(s).")

    try:
        chunk_number = 1
        for turn_index, turn in enumerate(dialogue, start=1):
            chunks = split_text(turn["text"])
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_file = temp_dir / f"chunk_{chunk_number:04d}.mp3"
                print(
                    f"Generating turn {turn_index}/{len(dialogue)} "
                    f"({turn['speaker']}, voice={turn['voice']}), "
                    f"chunk {chunk_index}/{len(chunks)}..."
                )
                generate_speech_chunk(chunk, turn["voice"], chunk_file)
                mp3_files.append(chunk_file)
                chunk_number += 1

        print("Combining chunks into final MP3...")
        combine_mp3_files(mp3_files, output_mp3)
        print(f"Done: {output_mp3}")

        if not args.no_mp4:
            bg = resolve_background_image(args.image, default_image_dir)
            mp4_dest = args.mp4
            if bg is None and mp4_dest is not None:
                print("Cannot create MP4: no background image (use image/ folder or --image).")
                sys.exit(1)
            if bg is not None:
                if mp4_dest is None:
                    mp4_dest = mp4_dir / f"{output_mp3.stem}.mp4"
                ffmpeg_bin = _ffmpeg_executable()
                if not ffmpeg_bin:
                    print(
                        "MP4 skipped: ffmpeg not found. Set FFMPEG_PATH in .env or install ffmpeg "
                        "(still-image video requires ffmpeg)."
                    )
                else:
                    print(f"Creating MP4 with still image: {bg.name}")
                    mux_still_image_mp4(ffmpeg_bin, bg, output_mp3, mp4_dest)
                    print(f"Done: {mp4_dest}")

    finally:
        _remove_temp_dir(temp_dir)


if __name__ == "__main__":
    main()
