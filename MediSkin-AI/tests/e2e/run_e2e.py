"""Optional end-to-end browser test for MediSkin AI (Playwright + Chromium).

It builds throw-away fixtures in a temp directory (a RANDOM-WEIGHTS smoke-test model, matching class
names and metrics, a scratch SQLite DB and report folder), starts the app on a free port, drives the
real UI and shuts everything down. The fixtures exist only to exercise the application; the model
makes meaningless predictions and nothing from this test is ever written to ``models/``.

    pip install playwright && playwright install chromium
    python tests/e2e/run_e2e.py                      # full flow with a smoke-test model
    python tests/e2e/run_e2e.py --no-model           # shipped state: no trained model installed
    python tests/e2e/run_e2e.py --screenshots shots  # keep a screenshot of each page
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "Str0ng!Passw0rd"
EMAIL = "test.user@example.com"
RESULTS: list[tuple[str, bool]] = []


# --- Fixtures -------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_images(folder: Path) -> None:
    rng = np.random.default_rng(1)
    arr = np.clip(rng.normal([190, 140, 120], 18, (300, 400, 3)), 0, 255).astype("uint8")
    Image.fromarray(arr).save(folder / "skin.png")
    Image.fromarray(arr).save(folder / "skin.jpg", quality=90)
    (folder / "corrupt.png").write_bytes(b"this is not really an image" * 20)
    Image.new("RGB", (10, 10), (200, 150, 130)).save(folder / "tiny.png")
    (folder / "empty.png").write_bytes(b"")


def make_smoke_model(folder: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from training.evaluate import compute_metrics, save_metrics
    from training.train import build_model

    names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    model, _, _ = build_model(len(names), "cnn", (224, 224), False)
    model.save(folder / "skin_model.keras")
    (folder / "class_names.json").write_text(json.dumps(names))
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 120)
    predicted = np.where(rng.random(120) < 0.6, y, rng.integers(0, 7, 120))
    save_metrics(compute_metrics(y, predicted, names), path=folder / "model_metrics.json",
                 extra={"model": {"architecture": "SMOKE-TEST (random weights)"}, "dataset": {"test": 120}})


def start_server(folder: Path, with_model: bool, port: int) -> subprocess.Popen:
    def path(name: str, present: bool) -> str:
        return str(folder / (name if present else f"absent_{name}"))

    env = {**os.environ, "DATABASE_URL": f"sqlite:///{folder / 'e2e.db'}", "REPORTS_DIR": str(folder / "reports"),
           "MODEL_PATH": path("skin_model.keras", with_model), "CLASS_NAMES_PATH": path("class_names.json", with_model),
           "METRICS_PATH": path("model_metrics.json", with_model), "HISTORY_PATH": path("history.json", False)}
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port), "--server.headless", "true"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read() == b"ok":
                return process
        except OSError:
            time.sleep(1)
    process.kill()
    raise SystemExit("The app server did not start.")


# --- Helpers ----------------------------------------------------------------------
def step(name: str):
    """Run a function as a named check; failures are recorded, not raised, so every step reports."""
    def decorate(func):
        try:
            func()
            RESULTS.append((name, True))
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False))
            print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}")
        return func
    return decorate


def ready(page, text: str, timeout: int = 60000) -> None:
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout)


def nav(page, label: str) -> None:
    page.locator('[data-testid="stSidebar"]').get_by_role("link", name=label).click()
    page.wait_for_timeout(1200)


def upload(page, path: Path) -> None:
    page.locator('input[type="file"]').set_input_files(str(path))


def sign_in(page, email: str = EMAIL, password: str = PASSWORD) -> None:
    panel = page.locator('[role="tabpanel"]:visible')
    panel.get_by_label("Email").fill(email)
    panel.get_by_label("Password", exact=True).fill(password)
    panel.get_by_role("button", name="Sign in").click()


def register(page, name: str, email: str, password: str, confirm: str | None = None) -> None:
    page.get_by_role("tab", name="Create account").click()
    panel = page.locator('[role="tabpanel"]:visible')
    panel.get_by_label("Full name").fill(name)
    panel.get_by_label("Email").fill(email)
    panel.get_by_label("Password", exact=True).fill(password)
    panel.get_by_label("Confirm password").fill(confirm or password)
    panel.get_by_role("button", name="Create account").click()


class Shots:
    def __init__(self, folder: Path | None):
        self.folder = folder
        if folder:
            folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, page, name: str) -> None:
        if self.folder:
            page.wait_for_timeout(500)
            page.screenshot(path=str(self.folder / f"{name}.png"), full_page=True)


# --- Scenarios ------------------------------------------------------------------------
def full_flow(browser, base: str, images: Path, shot: Shots) -> None:
    page = browser.new_context(viewport={"width": 1366, "height": 900}, accept_downloads=True).new_page()

    @step("sign-in page shows disclaimer and no sidebar")
    def _():
        page.goto(base)
        ready(page, "Welcome")
        assert page.get_by_text("not a medical diagnosis").first.is_visible()
        assert not page.locator('[data-testid="stSidebar"]').is_visible()
        shot(page, "01_login")

    @step("registration rejects a weak password")
    def _():
        register(page, "Test User", EMAIL, "weak")
        ready(page, "Password must contain")

    @step("registration rejects mismatched passwords")
    def _():
        register(page, "Test User", EMAIL, PASSWORD, confirm="Different1!")
        ready(page, "Passwords do not match")

    @step("registration succeeds and opens the dashboard")
    def _():
        register(page, "Test User", EMAIL, PASSWORD)
        ready(page, "Welcome back, Test")
        assert page.locator('[data-testid="stSidebar"]').is_visible()
        assert page.get_by_text("No analyses yet").first.is_visible()
        shot(page, "02_home_empty")

    @step("analyze: corrupt, tiny and empty uploads are rejected")
    def _():
        nav(page, "Analyze Skin")
        ready(page, "Upload a clear photo")
        upload(page, images / "corrupt.png")
        ready(page, "corrupted or cannot be read")
        upload(page, images / "tiny.png")
        ready(page, "Image is too small")
        upload(page, images / "empty.png")
        page.wait_for_timeout(1500)
        assert page.get_by_role("button", name="Analyze image").is_disabled()
        shot(page, "03_analyze_rejected")

    @step("analyze: valid PNG is previewed and analysis shows the result card")
    def _():
        upload(page, images / "skin.png")
        ready(page, "400×300px")
        shot(page, "04_analyze_preview")
        page.get_by_role("button", name="Analyze image").click()
        for text in ("AI suggestion (not a diagnosis)", "Top predictions", "About this condition", "What to do next"):
            ready(page, text, timeout=120000)
        body = page.inner_text("body").lower()
        assert "model confidence" in body and "not a medical diagnosis" in body
        shot(page, "05_results")

    @step("results: attention visualization is labelled as non-clinical")
    def _():
        ready(page, "AI attention visualization")
        assert page.get_by_text("this is not a clinical interpretation").first.is_visible()

    @step("results: PDF report is generated and downloads as a valid PDF")
    def _():
        page.get_by_role("button", name="Generate PDF report").click()
        page.get_by_role("button", name="Download PDF report").wait_for(timeout=30000)
        with page.expect_download() as download:
            page.get_by_role("button", name="Download PDF report").click()
        target = images / "downloaded_report.pdf"
        download.value.save_as(str(target))
        assert target.read_bytes()[:5] == b"%PDF-" and target.stat().st_size > 2000
        assert page.get_by_text("AI suggestion (not a diagnosis)").first.is_visible()  # no reset on download

    @step("history: entry listed; viewing it reopens the stored result")
    def _():
        nav(page, "Prediction History")
        ready(page, "1 of 1 analysis")
        shot(page, "06_history")
        page.get_by_role("button", name="View").first.click()
        ready(page, "AI suggestion (not a diagnosis)")
        assert page.get_by_text("Image not stored").first.is_visible()

    @step("history: second analysis (JPG) is stored")
    def _():
        nav(page, "Analyze Skin")
        ready(page, "Upload a clear photo")
        upload(page, images / "skin.jpg")
        ready(page, "400×300px")
        page.get_by_role("button", name="Analyze image").click()
        ready(page, "AI suggestion (not a diagnosis)", timeout=120000)
        nav(page, "Prediction History")
        ready(page, "2 of 2 analyses")

    @step("history: a report can be created for an older entry")
    def _():
        page.get_by_role("button", name="PDF").last.click()
        page.wait_for_selector('[data-testid="stDownloadButton"]', timeout=20000)

    @step("history: delete asks for confirmation, then removes the entry")
    def _():
        page.get_by_role("button", name="Delete").first.click()
        ready(page, "Delete this analysis?")
        shot(page, "07_delete_dialog")
        page.get_by_role("dialog").get_by_role("button", name="Delete").click()
        ready(page, "1 of 1 analysis")

    @step("reports page lists reports with a download button")
    def _():
        nav(page, "Reports")
        ready(page, "report(s)")
        assert page.locator('[data-testid="stDownloadButton"]').count() >= 1
        shot(page, "08_reports")

    @step("hospitals: honest empty state and map-search link (no invented data)")
    def _():
        nav(page, "Hospitals")
        ready(page, "No facility data configured")
        assert page.get_by_role("link", name="Search dermatologists on a map").is_visible()
        shot(page, "09_hospitals")

    @step("analytics: metrics, class table and confusion matrix come from the metrics file")
    def _():
        nav(page, "Model Analytics")
        ready(page, "Class-wise performance")
        ready(page, "Confusion matrix")
        shot(page, "10_analytics")

    @step("feedback: rating required, then submitted with a success message")
    def _():
        nav(page, "Feedback")
        ready(page, "Tell us how useful")
        page.get_by_role("button", name="Submit feedback").click()
        ready(page, "Please choose a rating from 1 to 5")
        page.locator('[data-testid="stFeedback"] button').nth(3).click()
        page.get_by_label("Comments (optional)").fill("Clear layout. <b>test</b>")
        page.get_by_role("button", name="Submit feedback").click()
        ready(page, "Your feedback has been submitted")
        shot(page, "11_feedback")

    @step("profile: wrong current password is rejected")
    def _():
        nav(page, "Profile")
        ready(page, "Change password")
        page.get_by_label("Current password").fill("Wrong!Pass1")
        page.get_by_label("New password", exact=True).fill("An0ther!Passw0rd")
        page.get_by_label("Confirm new password").fill("An0ther!Passw0rd")
        page.get_by_role("button", name="Change password").click()
        ready(page, "Current password is incorrect")
        shot(page, "12_profile")

    @step("about page renders")
    def _():
        nav(page, "About")
        ready(page, "How it works")
        shot(page, "13_about")

    @step("dashboard lists the recent analysis")
    def _():
        nav(page, "Home")
        ready(page, "Recent analyses")
        assert page.get_by_text("Triage indicator").first.is_visible()
        shot(page, "14_home")

    @step("sign out; a deep link while signed out shows sign-in (no dialog) and no sidebar")
    def _():
        page.get_by_role("button", name="Sign out").click()
        ready(page, "Welcome")
        page.goto(base + "/history")
        ready(page, "Welcome")
        page.wait_for_timeout(800)
        assert not page.locator('[data-testid="stSidebar"]').is_visible()
        assert page.locator('[data-testid="stDialog"]').count() == 0

    @step("sign-in: wrong password gives a generic error")
    def _():
        sign_in(page, password="Nope!Nope1")
        ready(page, "Invalid email or password")

    @step("sign-in: correct credentials open the requested page; history persisted")
    def _():
        sign_in(page)
        page.locator('[data-testid="stSidebar"]').wait_for(state="visible", timeout=30000)
        nav(page, "Prediction History")
        ready(page, "1 of 1 analysis")

    @step("phone viewport: dashboard has no horizontal overflow")
    def _():
        mobile = browser.new_context(viewport={"width": 390, "height": 800}).new_page()
        mobile.goto(base)
        ready(mobile, "Welcome")
        sign_in(mobile)
        ready(mobile, "Welcome back, Test")
        mobile.wait_for_timeout(1200)
        shot(mobile, "15_mobile_home")
        assert mobile.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 2


def no_model_flow(browser, base: str, images: Path, shot: Shots) -> None:
    page = browser.new_context(viewport={"width": 1366, "height": 900}).new_page()
    page.goto(base)
    ready(page, "Welcome")
    register(page, "No Model", "nomodel@example.com", PASSWORD)

    @step("no model: dashboard says the model is not installed")
    def _():
        ready(page, "Welcome back, No")
        assert page.get_by_text("Not installed").first.is_visible()

    @step("no model: analyze explains how to train/install one and disables analysis")
    def _():
        nav(page, "Analyze Skin")
        ready(page, "AI model unavailable")
        assert "python -m training.train" in page.inner_text("body")
        upload(page, images / "skin.png")
        ready(page, "400×300px")
        assert page.get_by_role("button", name="Analyze image").is_disabled()
        shot(page, "20_no_model_analyze")

    @step("no model: analytics shows an empty state instead of invented numbers")
    def _():
        nav(page, "Model Analytics")
        ready(page, "No evaluation metrics yet")
        shot(page, "21_no_model_analytics")

    @step("no model: results page shows an empty state")
    def _():
        nav(page, "Results")
        ready(page, "No result to show")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-model", action="store_true", help="Test the shipped state without a trained model.")
    parser.add_argument("--screenshots", type=Path, help="Folder to save page screenshots into.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mediskin_e2e_") as tmp:
        folder = Path(tmp)
        make_images(folder)
        if not args.no_model:
            print("Building a throw-away random-weights model (for UI testing only)...")
            make_smoke_model(folder)
        port = free_port()
        server = start_server(folder, with_model=not args.no_model, port=port)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                scenario = no_model_flow if args.no_model else full_flow
                scenario(browser, f"http://localhost:{port}", folder, Shots(args.screenshots))
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=20)

    failed = [name for name, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} steps passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
