# Oh Hell — Card Game

A browser-playable Oh Hell card game with ISMCTS AI opponents.

## Files needed in this folder

- `oh_hell_server.py` — Flask API server (this file)
- `oh_hell_game.html` — Frontend
- `oh_hell_game.py` — Game logic (your existing file)
- `CardHelper.py` — Card utilities (your existing file)
- `requirements.txt` — Python dependencies
- `Procfile` — Railway/Heroku start command

## Deploy to Railway

1. Make sure all 6 files above are in a GitHub repo (can be private)
2. Go to https://railway.app and sign in with GitHub
3. Click **New Project → Deploy from GitHub repo**
4. Select your repo
5. Railway will auto-detect the Procfile and deploy
6. Once deployed, click the generated URL — that's your public game link!

No environment variables needed. Railway sets PORT automatically.

## Run locally

```bash
pip install flask flask-cors
python oh_hell_server.py
# then open http://localhost:5001
```

## Notes

- Each browser session gets its own isolated game (via session_id)
- Idle sessions are cleaned up after 2 hours
- The server uses 1 worker with 4 threads — fine for casual use
  (ISMCTS is CPU-heavy; for high traffic you'd want a task queue)
