"""End-to-end smoke test (no pytest required).

Runs the full pipeline against FastAPI's TestClient with the offline
providers (MockGeminiClient + MockImageGenerator) and verifies:
create -> poll status -> fetch comic -> regenerate panel -> continue -> export PDF.

Usage:
    python backend/tests/smoke_e2e.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make ``app`` importable when run directly (…/backend on sys.path).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force offline providers BEFORE the app/settings are imported.
os.environ["IMAGE_PROVIDER"] = "mock"
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.ai.gemini import MockGeminiClient  # noqa: E402
from app.ai.image_generator import MockImageGenerator  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.dependencies import get_gemini, get_image_provider  # noqa: E402
from app.main import app  # noqa: E402


def _wait_for_ready(client: TestClient, comic_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        resp = client.get(f"/api/comics/{comic_id}/status")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in ("READY", "FAILED"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"Comic did not settle in time; last status={last}")


def _report(failures: list[str]) -> int:
    print()
    if failures:
        print("SMOKE FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("SMOKE TEST PASSED (all checks green)")
    return 0


def main() -> int:
    # Deterministic, offline providers.
    settings = get_settings()
    app.dependency_overrides[get_gemini] = lambda: MockGeminiClient()
    app.dependency_overrides[get_image_provider] = lambda: MockImageGenerator(settings)

    failures: list[str] = []

    with TestClient(app) as client:
        # 1) Health --------------------------------------------------------
        health = client.get("/api/health")
        print("health:", health.status_code, health.json())
        if health.status_code != 200:
            failures.append("health endpoint failed")

        # 2) Image provider info ------------------------------------------
        prov = client.get("/api/images/provider")
        print("provider:", prov.status_code, prov.json())
        if prov.status_code != 200 or not prov.json().get("available"):
            failures.append("image provider not available")

        # 3) Create comic --------------------------------------------------
        payload = {
            "prompt": "A brave fox exploring an enchanted forest",
            "character_name": "Leo",
            "character_description": "A young orange fox wearing a blue scarf",
            "setting": "Enchanted forest",
            "tone": "adventure",
            "art_style": "anime",
            "panel_count": 3,
        }
        created = client.post("/api/comics", json=payload)
        print("create:", created.status_code, created.json())
        if created.status_code != 202:
            failures.append(f"create comic failed: {created.text}")
            return _report(failures)
        comic_id = created.json()["comic_id"]

        # 4) Poll status ---------------------------------------------------
        status = _wait_for_ready(client, comic_id)
        print("status:", status)
        if status["status"] != "READY":
            failures.append(f"generation did not reach ready: {status}")

        # 5) Fetch comic ---------------------------------------------------
        comic = client.get(f"/api/comics/{comic_id}")
        print("get comic:", comic.status_code)
        if comic.status_code != 200:
            failures.append("get comic failed")
            return _report(failures)
        body = comic.json()
        print("  title:", body["title"], "| panels:", len(body["panels"]))
        if len(body["panels"]) != 3:
            failures.append(f"expected 3 panels, got {len(body['panels'])}")
        for panel in body["panels"]:
            if not panel.get("image_url"):
                failures.append(f"panel {panel['number']} missing image")
        panel_id = body["panels"][0]["id"]

        # 6) Panel image served --------------------------------------------
        img = client.get(f"/api/images/{comic_id}/panels/{panel_id}")
        print("panel image:", img.status_code, img.headers.get("content-type"))
        if img.status_code != 200:
            failures.append("panel image route failed")

        # 7) Regenerate a panel --------------------------------------------
        regen = client.post(
            f"/api/comics/{comic_id}/panels/{panel_id}/regenerate",
            json={"instruction": "Make it night-time with fireflies"},
        )
        print("regenerate:", regen.status_code)
        if regen.status_code != 200:
            failures.append(f"regenerate failed: {regen.text}")

        # 8) Continue comic -------------------------------------------------
        cont = client.post(
            f"/api/comics/{comic_id}/continue",
            json={"instruction": "Continue the adventure", "panel_count": 2},
        )
        print("continue:", cont.status_code)
        if cont.status_code != 200:
            failures.append(f"continue failed: {cont.text}")
        else:
            new_total = len(cont.json()["panels"])
            print("  total panels after continue:", new_total)
            if new_total != 5:
                failures.append(f"expected 5 panels after continue, got {new_total}")

        # 9) Export PDF -----------------------------------------------------
        export = client.post(f"/api/comics/{comic_id}/export")
        print("export:", export.status_code, export.json())
        if export.status_code != 200:
            failures.append(f"export failed: {export.text}")
        else:
            dl = client.get(f"/api/comics/{comic_id}/download")
            print("download:", dl.status_code, dl.headers.get("content-type"))
            if dl.status_code != 200 or dl.headers.get("content-type") != "application/pdf":
                failures.append("download failed or wrong content type")
            elif not dl.content.startswith(b"%PDF"):
                failures.append("downloaded file is not a PDF")

        # 10) Users ---------------------------------------------------------
        me = client.get("/api/users/me")
        mine = client.get("/api/users/me/comics")
        print("users/me:", me.status_code, me.json())
        print("users/me/comics:", mine.status_code, "count:", len(mine.json()))
        if me.status_code != 200 or mine.status_code != 200:
            failures.append("user routes failed")
        elif len(mine.json()) < 1:
            failures.append("user comics list empty")

        # 11) Error handling -------------------------------------------------
        missing = client.get("/api/comics/does-not-exist")
        print("404 case:", missing.status_code, missing.json())
        if missing.status_code != 404:
            failures.append("missing comic did not 404")

    return _report(failures)


if __name__ == "__main__":
    raise SystemExit(main())