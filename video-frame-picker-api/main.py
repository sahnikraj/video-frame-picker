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


def _extract_frames(video_path: Path, times: List[float], work_dir: Path) -> List[dict]:
    frames = []
    for idx, t in enumerate(times, start=1):
        frame_path = work_dir / f"frame-{idx}.png"
        _run_cmd(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(t),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        data = frame_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        frames.append(
            {
                "data_url": f"data:image/png;base64,{b64}",
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
