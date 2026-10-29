"""Try to launch Playwright headless Chromium once; print the outcome."""
import sys, time
from playwright.sync_api import sync_playwright
t0 = time.time()
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox"], timeout=30000)
        print("LAUNCH OK", b.version)
        b.close()
except Exception as e:
    print("LAUNCH FAILED:", type(e).__name__)
    print(str(e)[:3000])
print("seconds", round(time.time() - t0, 2))
