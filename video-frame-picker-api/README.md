# Lastframe Extractor API (Render)

Backend API for robust video frame extraction using FFmpeg.

## Endpoints
- `GET /health`
- `POST /api/extract-last-frames` (multipart form field: `video`)

## Local run
```bash
cd video-frame-picker-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Render deploy
1. Push this folder to GitHub (already in repo).
2. In Render: New + -> Blueprint and select this repo.
3. Render will read `video-frame-picker-api/render.yaml`.
4. Set env vars if needed:
- `MAX_UPLOAD_MB`
- `REQUEST_TIMEOUT_SEC`
- `ALLOWED_ORIGINS`

## Frontend integration
Call:
`POST https://<your-render-service>/api/extract-last-frames`
with multipart file field `video`.

Response contains frame `data_url` strings ready to render/download.
