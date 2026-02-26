# Hidden Discoveries Cloud Setup (GitHub Actions)

Workflow file:
- `.github/workflows/hidden-discoveries-cloud.yml`

Schedule:
- Daily at 09:00 IST (`03:30 UTC`)
- Also supports manual trigger (`Run workflow`)

## Required GitHub Secrets
Set these in: `GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret`

1. `OPENAI_API_KEY`
2. `GOOGLE_DRIVE_FOLDER_ID`
3. `GOOGLE_OAUTH_CLIENT_SECRET_JSON`
- Paste full JSON content of:
  `/Users/sahilsharma/Downloads/client_secret_1068032788976-l0vdrs3dkq3qh0a7bgisn0tl31qskdqf.apps.googleusercontent.com.json`
4. `GOOGLE_OAUTH_TOKEN_JSON`
- Paste full JSON content of:
  `/Users/sahilsharma/Documents/New project/secrets/google_oauth_token.json`

## One-time enablement
1. Push this branch/workflow to GitHub.
2. Add the 4 secrets above.
3. Open `Actions -> Hidden Discoveries Cloud Run -> Run workflow` for first test.
4. Verify docs appear in your Drive folder.

## Notes
- This workflow uses OAuth token refresh (no browser interaction in cloud).
- If token is revoked/expired, regenerate locally via:
  `python ops/hidden_discoveries_daily.py --auth-only`
  then update `GOOGLE_OAUTH_TOKEN_JSON` secret.
