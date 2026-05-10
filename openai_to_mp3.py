import argparse
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env next to this script, then cwd (so IDE shells see updates without restarting the app).
_script_dir = Path(__file__).resolve().parent
load_dotenv(_script_dir / ".env")
load_dotenv()

# ---------- Config ----------
MODEL = "gpt-4o-mini-tts"
VOICE = "alloy"
MAX_CHARS_PER_CHUNK = 3500  # keep chunks reasonably sized
TEMP_DIR_NAME = "tts_temp_chunks"
IMAGE_DIR_NAME = "image"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MP3_DIR_NAME = "mp3_files"
MP4_DIR_NAME = "mp4_files"
# ---------------------------

client = OpenAI()


def markdown_to_plain_text(md: str) -> str:
    """Basic Markdown cleanup for better speech output."""
    text = md

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Convert links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)

    # Remove images ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", "", text)

    # Remove headings markup
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^\s*([-*_]\s*){3,}$", "", text, flags=re.MULTILINE)

    # Convert bold/italic markers
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # Convert list markers into pauses
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "• ", text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    """Split text into chunks, trying paragraph boundaries first."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            # Split oversized paragraph by sentences
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


def generate_speech_chunk(text: str, output_path: Path):
    """Generate one MP3 chunk using OpenAI TTS."""
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
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


def combine_mp3_files(mp3_files, output_file: Path):
    """Combine MP3 files. Uses ffmpeg when available; otherwise joins bytes (fine for same TTS chunks)."""
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
            "ffmpeg not found (PATH may be stale in this terminal—restart Cursor, or set "
            "FFMPEG_PATH in .env). Joining chunk MP3s as raw bytes."
        )
        with open(output_file, "wb") as out:
            for mp3 in mp3_files:
                out.write(Path(mp3).read_bytes())


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
    Still image + MP3 → MP4 (H.264/AAC when possible).
    Tries encoders in order: libx264 (common), libopenh264 (many “lite” Windows builds),
    then mpeg4 as a last resort.
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    image_s = str(image.resolve())
    audio_s = str(audio_mp3.resolve())
    out_s = str(output_mp4.resolve())

    attempts: list[tuple[str, list[str]]] = [
        (
            "libx264",
            [
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
            ],
        ),
        (
            "libopenh264",
            [
                "-c:v",
                "libopenh264",
                "-profile:v",
                "baseline",
                "-b:v",
                "2500k",
            ],
        ),
        (
            "mpeg4",
            [
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
            ],
        ),
    ]

    # One decoded frame per second for the looping still (default 25 fps explodes encode time).
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
        + ". Prefer a ffmpeg build that includes libx264 (e.g. gyan.dev FFmpeg “full”). "
        + "Inspect any ffmpeg messages printed above."
    )


def main():
    default_image_dir = _script_dir / IMAGE_DIR_NAME
    mp3_dir = _script_dir / MP3_DIR_NAME
    mp4_dir = _script_dir / MP4_DIR_NAME

    parser = argparse.ArgumentParser(
        description="Convert Markdown script to spoken MP3 (OpenAI TTS); optionally mux a still image to MP4.",
    )
    parser.add_argument(
        "input_md", type=Path, help="Markdown file with the narration script"
    )
    parser.add_argument(
        "output_mp3",
        type=Path,
        nargs="?",
        default=None,
        help=f"Output MP3 path (default: {MP3_DIR_NAME}/<markdown stem>.mp3 next to openai_to_mp3.py)",
    )
    parser.add_argument(
        "--mp4",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            f"MP4 output path (default when omitted: {MP4_DIR_NAME}/<MP3 stem>.mp4 "
            "if a background image is found)"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            f"Still image file, or folder of images "
            f"(default: first image in {IMAGE_DIR_NAME}/ next to openai_to_mp3.py, sorted by name)"
        ),
    )
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="Do not produce MP4 even if a background image is available",
    )

    args = parser.parse_args()
    input_md = args.input_md
    if not input_md.exists():
        print(f"File not found: {input_md}")
        sys.exit(1)

    output_mp3 = (
        args.output_mp3
        if args.output_mp3 is not None
        else mp3_dir / f"{input_md.stem}.mp3"
    )
    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    raw_md = input_md.read_text(encoding="utf-8")
    plain_text = markdown_to_plain_text(raw_md)

    if not plain_text.strip():
        print("No readable text found in the Markdown file.")
        sys.exit(1)

    chunks = split_text(plain_text, MAX_CHARS_PER_CHUNK)
    print(f"Found {len(chunks)} chunk(s) to convert.")

    temp_dir = input_md.parent / TEMP_DIR_NAME
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = []

    try:
        for i, chunk in enumerate(chunks, start=1):
            chunk_file = temp_dir / f"chunk_{i:03d}.mp3"
            print(f"Generating chunk {i}/{len(chunks)}...")
            generate_speech_chunk(chunk, chunk_file)
            mp3_files.append(chunk_file)

        print("Combining chunks into final MP3...")
        combine_mp3_files(mp3_files, output_mp3)

        print(f"Done: {output_mp3}")

        if not args.no_mp4:
            bg = resolve_background_image(args.image, default_image_dir)
            mp4_dest = args.mp4
            if bg is None and mp4_dest is not None:
                print(
                    "Cannot create MP4: no background image (use image/ folder or --image)."
                )
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
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
