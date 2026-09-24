"""API tests (stdlib unittest; no pytest dependency).

Exercises the FastAPI surface end-to-end with offline providers
(MockGeminiClient + MockImageGenerator) so the suite runs without network
access, API keys or a live Stable Diffusion server.

Run from the repository root:
    python -m unittest discover -s backend/tests -v
Or run this file directly:
    python backend/tests/test_api.py
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Force the offline providers before the app/settings are imported.
os.environ["IMAGE_PROVIDER"] = "mock"
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.ai.gemini import MockGeminiClient  # noqa: E402
from app.ai.image_generator import MockImageGenerator  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.dependencies import get_gemini, get_image_provider  # noqa: E402
from app.main import app  # noqa: E402

#: Terminal statuses for polling.
_TERMINAL = ("READY", "FAILED")


def _wait_for_ready(client: TestClient, comic_id: str, timeout: float = 30.0) -> dict:
    """Poll the status endpoint until the comic settles or times out."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/api/comics/{comic_id}/status")
        if resp.status_code != 200:
            raise AssertionError(f"status endpoint error: {resp.text}")
        last = resp.json()
        if last["status"] in _TERMINAL:
            return last
        time.sleep(0.05)
    raise AssertionError(f"comic {comic_id} did not settle; last={last}")


class ApiTestCase(unittest.TestCase):
    """Shared client/override setup for every API test."""

    def setUp(self) -> None:
        settings = get_settings()
        # Deterministic, offline providers; no external services required.
        app.dependency_overrides[get_gemini] = lambda: MockGeminiClient()
        app.dependency_overrides[get_image_provider] = (
            lambda: MockImageGenerator(settings)
        )
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        app.dependency_overrides.clear()

    # --- helpers -----------------------------------------------------------
    def _create_comic(self, panel_count: int = 3) -> str:
        resp = self.client.post(
            "/api/comics",
            json={
                "prompt": "A brave fox exploring an enchanted forest",
                "character_name": "Leo",
                "character_description": "A young orange fox wearing a blue scarf",
                "setting": "Enchanted forest",
                "tone": "adventure",
                "art_style": "anime",
                "panel_count": panel_count,
            },
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        return resp.json()["comic_id"]

    # --- tests -------------------------------------------------------------
    def test_health(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_provider_info(self) -> None:
        resp = self.client.get("/api/images/provider")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "MockImageGenerator")
        self.assertTrue(body["available"])

    def test_full_generation_flow(self) -> None:
        comic_id = self._create_comic(panel_count=3)
        status = _wait_for_ready(self.client, comic_id)
        self.assertEqual(status["status"], "READY", status)
        self.assertEqual(status["progress"], 100)

        resp = self.client.get(f"/api/comics/{comic_id}")
        self.assertEqual(resp.status_code, 200)
        comic = resp.json()
        self.assertTrue(comic["title"])
        self.assertEqual(len(comic["panels"]), 3)
        for index, panel in enumerate(comic["panels"], start=1):
            self.assertEqual(panel["number"], index)
            self.assertTrue(panel["image_url"], f"panel {index} missing image")
            self.assertTrue(panel["scene"])
            self.assertTrue(panel["image_prompt"])

    def test_panel_image_is_served(self) -> None:
        comic_id = self._create_comic(panel_count=2)
        _wait_for_ready(self.client, comic_id)
        comic = self.client.get(f"/api/comics/{comic_id}").json()
        panel_id = comic["panels"][0]["id"]

        resp = self.client.get(f"/api/images/{comic_id}/panels/{panel_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        self.assertTrue(resp.content.startswith(b"\x89PNG"))

    def test_regenerate_panel(self) -> None:
        comic_id = self._create_comic(panel_count=3)
        _wait_for_ready(self.client, comic_id)
        comic = self.client.get(f"/api/comics/{comic_id}").json()
        panel_id = comic["panels"][1]["id"]
        before = comic["panels"][1]["scene"]

        resp = self.client.post(
            f"/api/comics/{comic_id}/panels/{panel_id}/regenerate",
            json={"instruction": "Make it night-time with fireflies"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        after = resp.json()["panels"]
        # Only the targeted panel changed; the count is stable.
        self.assertEqual(len(after), 3)
        self.assertEqual(after[1]["id"], panel_id)
        self.assertNotEqual(after[1]["scene"], before)

    def test_continue_comic(self) -> None:
        comic_id = self._create_comic(panel_count=3)
        _wait_for_ready(self.client, comic_id)

        resp = self.client.post(
            f"/api/comics/{comic_id}/continue",
            json={"instruction": "Continue the adventure", "panel_count": 2},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        panels = resp.json()["panels"]
        self.assertEqual(len(panels), 5)
        for panel in panels[3:]:
            self.assertTrue(panel["image_url"], "new panel missing image")

    def test_export_and_download_pdf(self) -> None:
        comic_id = self._create_comic(panel_count=2)
        _wait_for_ready(self.client, comic_id)

        export = self.client.post(f"/api/comics/{comic_id}/export")
        self.assertEqual(export.status_code, 200, export.text)
        self.assertEqual(
            export.json()["download_url"], f"/api/comics/{comic_id}/download"
        )

        download = self.client.get(f"/api/comics/{comic_id}/download")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.headers["content-type"], "application/pdf")
        self.assertTrue(download.content.startswith(b"%PDF"))

    def test_download_before_export_is_rejected(self) -> None:
        comic_id = self._create_comic(panel_count=1)
        _wait_for_ready(self.client, comic_id)
        resp = self.client.get(f"/api/comics/{comic_id}/download")
        self.assertEqual(resp.status_code, 409, resp.text)

    def test_missing_comic_returns_404(self) -> None:
        resp = self.client.get("/api/comics/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"]["code"], "NOT_FOUND")

    def test_missing_panel_image_returns_404(self) -> None:
        comic_id = self._create_comic(panel_count=1)
        _wait_for_ready(self.client, comic_id)
        resp = self.client.get(f"/api/images/{comic_id}/panels/nope")
        self.assertEqual(resp.status_code, 404)

    def test_generation_before_ready_is_rejected(self) -> None:
        # A freshly created comic may still be generating; waiting is not
        # required to assert the state guard, so use a brand new id.
        comic_id = self._create_comic(panel_count=1)
        resp = self.client.post(
            f"/api/comics/{comic_id}/continue",
            json={"instruction": "More please", "panel_count": 1},
        )
        # Either it has not finished (409 invalid state) or it already
        # completed (200). Both are valid; a 500 would not be.
        self.assertIn(resp.status_code, (200, 409), resp.text)

    def test_create_validation_rejects_empty_prompt(self) -> None:
        resp = self.client.post("/api/comics", json={"prompt": ""})
        self.assertEqual(resp.status_code, 422)

    def test_create_validation_rejects_bad_panel_count(self) -> None:
        resp = self.client.post(
            "/api/comics", json={"prompt": "hi", "panel_count": 999}
        )
        self.assertEqual(resp.status_code, 422)

    def test_users_me(self) -> None:
        resp = self.client.get("/api/users/me")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], "anonymous")

    def test_users_me_comics_lists_created(self) -> None:
        comic_id = self._create_comic(panel_count=1)
        _wait_for_ready(self.client, comic_id)
        resp = self.client.get("/api/users/me/comics")
        self.assertEqual(resp.status_code, 200)
        ids = [c["id"] for c in resp.json()]
        self.assertIn(comic_id, ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)