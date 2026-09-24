# ComicCraft

An AI comic generator. Describe a story, pick a style, and ComicCraft writes
the outline, drafts each panel, renders the artwork, and exports a PDF — all
from a single self-contained web app.

The UI started life as a Canva export (`frontend/index.html`) and has been
rewired onto a real FastAPI backend. The whole thing runs **fully offline**
with zero external requests: fonts, icons, CSS and stock imagery are vendored
locally, and both AI providers have deterministic mock fallbacks.

---

## How it fits together

```
ComicCraft/
├── backend/                 FastAPI app + config + tests
│   ├── app/
│   │   ├── main.py          create_app(): routers + static mounts
│   │   ├── config.py        pydantic-settings → reads backend/.env
│   │   ├── dependencies.py  selects Gemini/mock + image provider
│   │   ├── ai/              Gemini client, image generators
│   │   ├── api/routes/      comics, images, exports, users, health
│   │   ├── services/        story, character, prompt, image, export
│   │   ├── database/        in-memory comic store
│   │   ├── models/  schemas/
│   │   └── utils/
│   ├── tests/               unittest suite + smoke_e2e.py
│   ├── generated/           runtime output: images/, pdfs/
│   ├── .env                 your keys (git-ignored)
│   └── .env.example         committed template
└── frontend/                static site served by the backend at "/"
    ├── index.html           the app (HTML + CSS + JS in one file)
    ├── assets/              vendor/, fonts/, images/, CREDITS.md
    ├── src/input.css        Tailwind source
    ├── tailwind.config.js
    └── package.json         build-only tooling (npm run build:css)
```

**Single origin.** FastAPI mounts `frontend/` at `/` and `backend/generated`
at `/generated`, so the browser only ever talks to one host — no CORS
configuration is needed in practice.

---

## Quick start

### 1. Backend

```powershell
cd C:\Users\DannBerlinD\Python\Learn\ComicCraft

# one-time: create the virtualenv and install deps
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 2. Configure (optional)

```powershell
copy backend\.env.example backend\.env
```

Edit `backend\.env` and fill in your keys. **Every value is optional** — leave
anything blank and ComicCraft uses its built-in offline default.

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Google Gemini key for story/outline/panel copy. Empty ⇒ deterministic `MockGeminiClient`. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model used for every story call. |
| `IMAGE_PROVIDER` | `mock` | `mock` (offline PNGs) or `stable_diffusion` (real images). |
| `SD_API_URL` | `http://127.0.0.1:7860` | AUTOMATIC1111-compatible WebUI endpoint. |
| `SD_API_AUTH` | *(empty)* | WebUI basic auth as `user:password`. |
| `SD_WIDTH` / `SD_HEIGHT` | `768` / `512` | txt2img output size. |
| `SD_STEPS` / `SD_CFG_SCALE` | `28` / `7.0` | txt2img quality vs. speed. |
| `SD_SAMPLER_NAME` | `DPM++ 2M Karras` | txt2img sampler. |
| `MIN_PANEL_COUNT` / `MAX_PANEL_COUNT` | `1` / `20` | Accepted `panel_count` range. |
| `STORAGE_BASE` | `…\backend\generated` | **Must be absolute** (see gotchas). |
| `CORS_ORIGINS` | `["*"]` | JSON array. Only matters if you split origins. |
| `APP_ENV` / `LOG_LEVEL` | `development` / `INFO` | Environment name and log verbosity. |

Restart the server after editing — settings are read **once per process**.
Real environment variables always override the file.

> **Tip:** with nothing configured you get mock story data *and* placeholder
> PNGs, so the full pipeline (including PDF export) works on a clean checkout
> with no keys and no Stable Diffusion install.

### 3. Run

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Then open **http://127.0.0.1:8000/** — the frontend is served by the same
process, so no second server is required.

---

## Tests

```powershell
# 27 tests, no network or API keys needed
.venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"

# end-to-end pipeline without pytest:
# create -> poll status -> fetch -> regenerate -> continue -> export PDF
.venv\Scripts\python.exe backend\tests\smoke_e2e.py
```

Both force `IMAGE_PROVIDER=mock` and an empty `GEMINI_API_KEY`, so they never
hit a live service.

---

## API reference

All routes are prefixed with `/api`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness probe → `{"status":"ok"}` |
| `GET` | `/api/images/provider` | Active image provider + reachability |
| `GET` | `/api/users/me` | Current (anonymous) user |
| `GET` | `/api/users/me/comics` | Library listing for the current user |
| `POST` | `/api/comics` | **202** start generation → `{comic_id, status}` |
| `GET` | `/api/comics/{id}/status` | Poll → `{status, progress, current_step}` |
| `GET` | `/api/comics/{id}` | Full comic record |
| `POST` | `/api/comics/{id}/panels/{panel_id}/regenerate` | Re-render one panel |
| `POST` | `/api/comics/{id}/continue` | Extend the story with more panels |
| `POST` | `/api/comics/{id}/export` | Build PDF → `{download_url}` |
| `GET` | `/api/comics/{id}/download` | The generated PDF (`application/pdf`) |
| `DELETE` | `/api/comics/{id}` | **204** remove a comic |

Outside `/api`:

| Path | Serves |
|---|---|
| `/` | `frontend/index.html` (the app) |
| `/generated/images/<file>` | Panel artwork written during generation |

### Example: create a comic

```powershell
$body = @{
    prompt                = 'A brave fox exploring an enchanted forest'
    character_name        = 'Leo'
    character_description = 'A young orange fox wearing a blue scarf'
    setting               = 'Enchanted forest'
    tone                  = 'adventure'
    art_style             = 'anime'
    panel_count           = 3
} | ConvertTo-Json

$comic = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/comics `
    -Body $body -ContentType 'application/json'

# poll until READY
do {
    Start-Sleep -Milliseconds 400
    $s = Invoke-RestMethod "http://127.0.0.1:8000/api/comics/$($comic.comic_id)/status"
} while ($s.status -notin 'READY','FAILED')

# read it
Invoke-RestMethod "http://127.0.0.1:8000/api/comics/$($comic.comic_id)"

# export + download the PDF
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/comics/$($comic.comic_id)/export"
Invoke-WebRequest "http://127.0.0.1:8000/api/comics/$($comic.comic_id)/download" -OutFile comic.pdf
```

A generated panel object looks like:

```jsonc
{
  "id": "34oaulzgq7jw",
  "number": 1,
  "title": "...",

---

## AI providers

**Story** — `backend/app/dependencies.py` picks the client based on your key:

* `GEMINI_API_KEY` set → `GeminiClient` (live `google-genai` calls)
* `GEMINI_API_KEY` empty → `MockGeminiClient` (deterministic, offline)

**Images** — `IMAGE_PROVIDER` selects the generator:

* `mock` → `MockImageGenerator`, a real PNG drawn with Pillow (no service)
* `stable_diffusion` → `StableDiffusionImageGenerator`, `POST {SD_API_URL}/sdapi/v1/txt2img`

To use real images, launch an AUTOMATIC1111-compatible WebUI with the API
enabled (`--api`) and set `IMAGE_PROVIDER=stable_diffusion`. Any other value
raises `ValueError` at startup.

---

## Frontend notes

The site is intentionally **offline-first**:

* Tailwind is compiled once to `assets/vendor/tailwind.css` — no CDN.
* Lucide icons are vendored at `assets/vendor/lucide.min.js`.
* DM Sans + Fraunces are self-hosted `.woff2` with `assets/fonts/fonts.css`.
* 12 stock photos live in `assets/images/` (see `assets/CREDITS.md`).
* No Google Fonts, no Pexels, no jsPDF (export happens on the backend).

Rebuild the stylesheet only if you change `src/input.css`:

```powershell
cd frontend
npm install          # one-time, needs internet (recreates node_modules/)
npm run build:css
```

`frontend/node_modules/` is **not** shipped — it is build-only, git-ignored,
and safe to delete at any time (it costs ~11.6 MB). The committed output,
`assets/vendor/tailwind.css`, is all the running app needs. Deleting it does
not affect the site; you only need `npm install` again if you edit the CSS.

The hero uses a fixed full-screen `<video>` background
(`assets/hero-animation.mp4`, ~21 MB) with all page content layered on top as
an overlay. Because the video sits at `z-index:-2`, any opaque element above
it would hide it — Canva's inline `rgb()` backgrounds were converted to
translucent `rgba()`, and `body` uses the `background-color` longhand so the
inline `background` shorthand can't paint over it.

---

## Gotchas

* **`STORAGE_BASE` must be absolute.** A relative value crashes startup with
  `RuntimeError: Directory 'generated' does not exist`, because Starlette's
  `StaticFiles` resolves it against the process working directory — not
  `backend/`.
* **`CORS_ORIGINS` must be a JSON array** (`["*"]`), not a bare `*`. It's a
  `list[str]` field, so a non-JSON value fails to parse at startup.
* **Restart after editing `.env`** — settings are cached per process.
* **In-memory store** — comics live in process memory. Restarting the server
  empties the library (the PNG/PDF files remain on disk).
* **No auth** — `/api/users/me` always returns the same anonymous user.
* **No Playwright** — browser-level visual checks aren't wired up; it is not a
  dependency, so install it separately if you need screenshots.

---

## Credits

Third-party assets (fonts, icons, photography) are credited in
[`frontend/assets/CREDITS.md`](frontend/assets/CREDITS.md).

  "scene": "...",
  "narration": "...",
  "image_prompt": "...",
  "image_url": "/generated/images/mock-511940c1...-1790272194.png",
  "dialogue": [{ "character": "Leo", "text": "What lies beyond?" }],
  "status": "READY"
}
```

`dialogue` is always a list (the frontend's `panelDialogue()` handles both
list and string shapes defensively).
