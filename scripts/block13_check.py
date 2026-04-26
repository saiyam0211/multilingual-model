"""Block 13 checks: README links + Space health + local artifact presence."""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

REQUIRED_FILES = [
    "plots/asr_before_after.png",
    "plots/category_heatmap.png",
    "plots/reward_curve.png",
    "plots/attack_examples.png",
    "results/eval_baseline.jsonl",
    "results/eval_trained.jsonl",
    "results/manual_audit.csv",
    "results/asr_matrix.json",
]


def extract_links(md: str) -> list[str]:
    # markdown links and bare https URLs
    links = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", md)
    links += re.findall(r"(?<!\()(?<!\[)(https?://[^\s)]+)", md)
    # dedupe but keep order
    out = []
    seen = set()
    for l in links:
        l = l.rstrip("`.,;:!?)")
        if l and l not in seen:
            seen.add(l)
            out.append(l)
    return out


def main() -> None:
    errors = []
    md = README.read_text(encoding="utf-8")

    # File existence checks
    for rel in REQUIRED_FILES:
        p = ROOT / rel
        if not p.exists():
            errors.append(f"missing file: {rel}")

    # External links
    links = extract_links(md)
    ok_links = 0
    for url in links:
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code < 400:
                ok_links += 1
            elif r.status_code == 401 and "huggingface.co/" in url:
                # Private model links are acceptable references but not public.
                # Keep as warning, not hard error for local check.
                print(f"warning: private link (401): {url}")
            else:
                errors.append(f"link bad ({r.status_code}): {url}")
        except Exception as e:
            errors.append(f"link error: {url} ({e})")

    # Space health
    try:
        r = requests.get("https://saiyam0211-polyglot-redteam.hf.space/api/health", timeout=30)
        if r.status_code >= 400:
            errors.append(f"space health bad status: {r.status_code}")
        else:
            payload = r.json()
            if payload.get("status") not in {"healthy", "degraded"}:
                errors.append(f"space health unexpected payload: {payload}")
    except Exception as e:
        errors.append(f"space health error: {e}")

    report = {
        "links_total": len(links),
        "links_ok": ok_links,
        "errors": errors,
    }
    out = ROOT / "results" / "block13_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        print("Block 13 check: FAILED")
        for e in errors:
            print(" -", e)
    else:
        print("Block 13 check: PASSED")
    print(f"report: {out}")


if __name__ == "__main__":
    main()

