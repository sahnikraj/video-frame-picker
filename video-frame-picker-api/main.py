import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "Lastframe Extractor API"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "180"))
FRAME_IMAGE_FORMAT = os.getenv("FRAME_IMAGE_FORMAT", "jpeg").strip().lower()
FRAME_IMAGE_QUALITY = int(os.getenv("FRAME_IMAGE_QUALITY", "4"))
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "1280"))

raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
if raw_origins == "*":
    allowed_origins: List[str] = ["*"]
else:
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_cmd(cmd: List[str], timeout: int = REQUEST_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed")
    return result


def _probe_duration(video_path: Path) -> float:
    result = _run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise RuntimeError("Invalid duration")
    return duration


def _build_times(duration: float, count: int = 5) -> List[float]:
    epsilon = 0.03
    total = max(duration - epsilon, 0)
    start = max(total - 2, 0)
    step = 0 if count == 1 else (total - start) / (count - 1)
    return [max(0, min(start + i * step, total)) for i in range(count)]


def _format_time(seconds: float) -> str:
    s = max(0, int(seconds))
    mins = s // 60
    rem = s % 60
    return f"{mins}:{rem:02d}"


def _get_frame_encoding() -> tuple[str, str, List[str]]:
    image_format = FRAME_IMAGE_FORMAT if FRAME_IMAGE_FORMAT in {"jpeg", "png", "webp"} else "jpeg"

    if image_format == "png":
        return ".png", "image/png", []

    if image_format == "webp":
        webp_quality = max(1, min(FRAME_IMAGE_QUALITY, 100))
        return ".webp", "image/webp", ["-c:v", "libwebp", "-q:v", str(webp_quality)]

    # JPEG: lower payload and faster response than PNG for preview thumbnails.
    jpeg_quality = max(2, min(FRAME_IMAGE_QUALITY, 31))
    return ".jpg", "image/jpeg", [
        "-c:v",
        "mjpeg",
        "-q:v",
        str(jpeg_quality),
        "-pix_fmt",
        "yuvj420p",
        "-strict",
        "-2",
    ]


def _extract_frame_at_time(video_path: Path, t: float, frame_path: Path, encoding_args: List[str]) -> None:
    ffmpeg_cmd: List[str] = [
        "ffmpeg",
        "-v",
        "error",
        "-hide_banner",
        "-y",
        "-ss",
        str(t),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
    ]
    if MAX_FRAME_WIDTH > 0:
        ffmpeg_cmd.extend(["-vf", f"scale={MAX_FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease"])
    ffmpeg_cmd.extend(encoding_args)
    ffmpeg_cmd.append(str(frame_path))
    _run_cmd(ffmpeg_cmd)


def _extract_frames(video_path: Path, times: List[float], work_dir: Path) -> List[dict]:
    ext, mime_type, encoding_args = _get_frame_encoding()
    output_pattern = work_dir / f"batch-frame-%02d{ext}"
    start_time = times[0] if times else 0.0
    end_time = times[-1] if times else 2.0
    window = max(end_time - start_time, 0.1)
    fps = max(2.0, min(12.0, (len(times) - 1) / window if len(times) > 1 else 2.0))
    ffmpeg_cmd: List[str] = [
        "ffmpeg",
        "-v",
        "error",
        "-hide_banner",
        "-y",
        "-sseof",
        "-2",
        "-i",
        str(video_path),
        "-frames:v",
        "5",
    ]
    vf_parts = [f"fps={fps:.3f}"]
    if MAX_FRAME_WIDTH > 0:
        vf_parts.append(f"scale={MAX_FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease")
    ffmpeg_cmd.extend(["-vf", ",".join(vf_parts)])
    ffmpeg_cmd.extend(encoding_args)
    ffmpeg_cmd.append(str(output_pattern))

    frames = []
    try:
        _run_cmd(ffmpeg_cmd)
    except Exception:
        # Fallback for files where `-sseof` path is unsupported.
        pass

    batch_files = sorted(work_dir.glob(f"batch-frame-*{ext}"))
    if batch_files:
        if len(batch_files) > 5:
            batch_files = batch_files[-5:]
        while len(batch_files) < 5:
            batch_files.append(batch_files[-1])

        for idx, (t, frame_path) in enumerate(zip(times, batch_files), start=1):
            data = frame_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            frames.append(
                {
                    "data_url": f"data:{mime_type};base64,{b64}",
                    "time_seconds": round(t, 3),
                    "time_label": _format_time(t),
                }
            )
        return frames

    for idx, t in enumerate(times, start=1):
        frame_path = work_dir / f"frame-{idx}{ext}"
        current_t = t
        success = False
        for _ in range(4):
            try:
                _extract_frame_at_time(video_path, current_t, frame_path, encoding_args)
            except Exception:
                pass
            if frame_path.exists():
                success = True
                break
            current_t = max(0.0, current_t - 0.15)

        if success:
            data = frame_path.read_bytes()
        elif frames:
            # Reuse prior frame if the final timestamp has no decodable frame.
            frames.append(
                {
                    "data_url": frames[-1]["data_url"],
                    "time_seconds": round(t, 3),
                    "time_label": _format_time(t),
                }
            )
            continue
        else:
            raise RuntimeError("Could not decode frames from video")

        b64 = base64.b64encode(data).decode("ascii")
        frames.append(
            {
                "data_url": f"data:{mime_type};base64,{b64}",
                "time_seconds": round(t, 3),
                "time_label": _format_time(t),
            }
        )
    return frames


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": APP_NAME}


@app.post("/api/extract-last-frames")
async def extract_last_frames(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "upload.bin").suffix or ".bin"

    with tempfile.TemporaryDirectory(prefix="lastframe-") as td:
        tmp_dir = Path(td)
        input_path = tmp_dir / f"input{suffix}"
        normalized_path = tmp_dir / "normalized.mp4"

        with input_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        size_mb = input_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB")

        try:
            duration = _probe_duration(input_path)
            times = _build_times(duration, count=5)
            try:
                frames = _extract_frames(input_path, times, tmp_dir)
                engine = "ffmpeg-direct"
            except Exception:
                _run_cmd(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(input_path),
                        "-map",
                        "0:v:0",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(normalized_path),
                    ]
                )

                duration = _probe_duration(normalized_path)
                times = _build_times(duration, count=5)
                frames = _extract_frames(normalized_path, times, tmp_dir)
                engine = "ffmpeg-transcode-fallback"

            return {
                "ok": True,
                "engine": engine,
                "duration_seconds": round(duration, 3),
                "frames": frames,
            }
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Video processing timed out")
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Video could not be processed. Try converting to MP4 (H.264/AAC) or use a smaller file. "
                    f"Details: {str(exc)[:300]}"
                ),
            )
