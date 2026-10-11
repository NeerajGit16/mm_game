"""Shared fixtures: one headless browser for every browser test."""

from __future__ import annotations

import pytest


def launch(p):
    for kwargs in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kwargs)
        except Exception:
            continue
    pytest.skip("no browser for Playwright: install Edge or Chrome, or run `playwright install chromium`")


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = launch(p)
        yield b
        b.close()
