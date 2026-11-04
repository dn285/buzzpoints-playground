"""
Fetch question text for every public college/open set on buzzpoints.buzz.

Reuses the cached, rate-limited ``fetch`` from ``plackett-luce/scraper.py`` so
each set's JSON is downloaded once and then read from ``../.cache``.

Data source (no HTML scraping needed):
    /api/index                                      -> list of sets
    /api/data?path=sets/<slug>/tossups_detail.json  -> tossup text + answers
    /api/data?path=sets/<slug>/bonuses_detail.json  -> bonus text + answers

Only the top-level (original edition) detail files are read. Mirror editions
are listed under each question's ``versions`` and are not double-counted.
Questions repeated within a set (same packet read in several rounds) are kept once.

Output: corpus.json, one object per question:
    {"set", "set_name", "level", "date", "kind", "id", "round", "category",
     "text", "answer"}
where ``text`` is what gets read aloud (tossup text, or bonus leadin + parts)
and ``answer`` is the primary answer(s) only (alternates in [...] dropped).

Usage:
    python fetch.py [--levels college open] [--output corpus.json]
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "plackett-luce"))
from scraper import BuzzpointsScraper  # noqa: E402

SITE = "https://www.buzzpoints.buzz"
DEFAULT_LEVELS = ("college", "open")
# Be gentler than the base scraper's default: these files are 1-3 MB each.
DELAY = 1.5


def strip_html(s: str | None) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def primary_answer(ans_html: str | None) -> str:
    """Answerline up to the first '[' or '(' (drops alternates/editor notes)."""
    ans = strip_html(ans_html)
    return re.split(r"\s*[\[(]", ans, maxsplit=1)[0].strip()


def data_url(slug: str, filename: str) -> str:
    return f"{SITE}/api/data?path={quote(f'sets/{slug}/{filename}', safe='')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    ap.add_argument("--output", default="corpus.json")
    args = ap.parse_args()

    scraper = BuzzpointsScraper(SITE, cache_dir=str(_REPO_ROOT / ".cache"), delay=DELAY)

    # The index is small and changes as sets are added, so always fetch it fresh.
    index = scraper.session.get(f"{SITE}/api/index", timeout=30).json()["sets"]
    sets = [
        s for s in index
        if s.get("visibility") == "public" and s.get("hasAccess")
        and s.get("level") in args.levels
    ]
    print(f"{len(sets)} public sets at levels {args.levels}")

    rows = []
    for i, s in enumerate(sets, 1):
        slug = s["slug"]
        meta = {"set": slug, "set_name": s["name"], "level": s["level"],
                "date": s.get("tournamentDate") or ""}
        tossups = json.loads(scraper.fetch(data_url(slug, "tossups_detail.json")))
        for q in tossups.values():
            rows.append({
                **meta, "kind": "tossup", "id": q["id"], "round": q.get("round"),
                "category": q.get("category") or "",
                "text": strip_html(q.get("questionHtml")),
                "answer": primary_answer(q.get("answer")),
            })
        n_b = 0
        if s.get("hasBonuses"):
            bonuses = json.loads(scraper.fetch(data_url(slug, "bonuses_detail.json")))
            n_b = len(bonuses)
            for q in bonuses.values():
                # Use leadin + parts only; partConv repeats the part text.
                parts = [strip_html(p) for p in q.get("parts") or []]
                rows.append({
                    **meta, "kind": "bonus", "id": q["id"], "round": q.get("round"),
                    "category": q.get("category") or "",
                    "text": " ".join([strip_html(q.get("leadin"))] + parts).strip(),
                    "answer": " / ".join(primary_answer(a) for a in q.get("answers") or []),
                })
        print(f"  [{i}/{len(sets)}] {s['name']}: {len(tossups)} tossups, {n_b} bonuses")

    # Some sets repeat a packet across rounds (e.g. 2024 ACF Winter lists 580
    # tossups, ~half unique). Keep the first copy of each question per set.
    seen, deduped = set(), []
    for r in rows:
        key = (r["set"], r["kind"], r["text"].lower())
        if r["text"] and key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"Dropped {len(rows) - len(deduped)} repeated/empty questions")
    rows = deduped

    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(rows)} questions to {args.output}")


if __name__ == "__main__":
    main()
