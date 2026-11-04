"""
Text scraper for buzzpoints tossups and bonuses.

Reuses the fetch/cache logic from ``plackett-luce/scraper.py`` (the existing
``BuzzpointsScraper``) and adds the one thing it throws away: the actual question
text and answerlines on each detail page.

This module deliberately does NOT re-implement the buzzpoints pulling logic. It
imports the existing scraper and extends it, so any future fix to fetching /
caching is inherited for free. (The base scraper only knows about tossups, so the
bonus index parse lives here as a generic ``scrape_index`` that handles both.)

Output: a local JSON file, one object per science question. Shapes differ by type:

    tossup: {"slug", "category", "packet_tossup", "question", "answer"}
    bonus:  {"slug", "category", "packet_bonus", "leadin",
             "parts": [{"value", "question", "answer"}, ...]}

Usage:
    # tossups (default)
    python scraper_text.py --base-url https://<tournament>.vercel.app/set/<slug>
    # bonuses
    python scraper_text.py --type bonus \
        --base-url https://<tournament>.vercel.app/set/<slug>
"""

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# Import the existing scraper without copying its logic. It lives in the sibling
# plackett-luce/ folder, which isn't a package, so add it to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "plackett-luce"))
from scraper import BuzzpointsScraper  # noqa: E402


# Categories on the index page that we treat as "Science". The buzzpoints site
# labels science tossups as "Science" or a subcategory like "Science - Biology";
# we match on the leading token so all of Biology/Chemistry/Earth/Physics/Other
# are included. Override with --category if a future set uses a different label.
DEFAULT_SCIENCE_PREFIX = "science"


class TossupTextScraper(BuzzpointsScraper):
    """Adds question-text extraction to the buzz-only base scraper."""

    def scrape_tossup_text(self, slug: str) -> dict:
        """Return {packet_tossup, question, answer} for one tossup detail page.

        The detail page puts the question block in a ``div.md:basis-1/2`` whose
        children are, in order: an <h3> ("Packet N: Tossup M"), a <p> (the
        question), a <div> starting with "ANSWER:", then metadata. We locate the
        ANSWER div and read backwards from it so we don't depend on class names
        that the site may restyle.
        """
        url = f"{self.base_url}/tossup/{slug}"
        page = self.soup(url)

        answer_div = None
        for div in page.find_all("div"):
            text = div.get_text(" ", strip=True)
            if text.upper().startswith("ANSWER:"):
                answer_div = div
                break

        if answer_div is None:
            return {"packet_tossup": None, "question": None, "answer": None}

        answer = _clean(answer_div.get_text(" ", strip=True))
        answer = re.sub(r"^ANSWER:\s*", "", answer, flags=re.I)

        container = answer_div.parent

        # Question = the <p> immediately preceding the ANSWER div in the container.
        question = None
        prev_p = answer_div.find_previous_sibling("p")
        if prev_p is not None:
            question = _clean(prev_p.get_text(" ", strip=True))

        # Packet/tossup header = the <h3> in the same container.
        packet_tossup = None
        h3 = container.find("h3") if container else None
        if h3 is not None:
            packet_tossup = _clean(h3.get_text(" ", strip=True))

        return {
            "packet_tossup": packet_tossup,
            "question": question,
            "answer": answer,
        }

    def scrape_science_tossups(self, science_prefix: str = DEFAULT_SCIENCE_PREFIX) -> list[dict]:
        """Scrape index, filter to science, and pull text for each match."""
        index = self.scrape_tossup_index()
        science = [
            row
            for row in index
            if row["category"].strip().lower().startswith(science_prefix)
        ]
        print(f"  {len(science)} of {len(index)} tossups are science "
              f"(prefix '{science_prefix}')")

        rows = []
        for i, row in enumerate(science, 1):
            detail = self.scrape_tossup_text(row["slug"])
            if not detail["question"]:
                print(f"  ! no question text extracted for {row['slug']}")
                continue
            rows.append({
                "slug": row["slug"],
                "category": row["category"],
                **detail,
            })
            if i % 20 == 0 or i == len(science):
                print(f"  {i}/{len(science)} science tossups fetched")
        return rows

    # ------------------------------------------------------------------
    # Bonuses
    # ------------------------------------------------------------------

    def scrape_index(self, kind: str) -> list[dict]:
        """Generic index parse for kind in {"tossup", "bonus"}.

        The base scraper hardcodes the tossup index; bonuses use the same table
        layout (a Category column plus a link to the /<kind>/<slug> detail page),
        so this one method covers both. Returns list of {slug, category}.
        """
        url = f"{self.base_url}/{kind}"
        print(f"Fetching {kind} index: {url}")
        page = self.soup(url)

        table = page.find("table")
        if table is None:
            raise RuntimeError(f"No table found on {kind} index page")

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        cat_idx = next((i for i, h in enumerate(headers) if "category" in h), 0)

        rows = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if not tds:
                continue
            category = tds[cat_idx].get_text(strip=True) if cat_idx < len(tds) else ""
            slug = None
            for td in tds:
                a = td.find("a", href=True)
                if a and f"/{kind}/" in a["href"]:
                    href = a["href"].split("?")[0]
                    slug = href.rstrip("/").rsplit("/", 1)[-1]
                    break
            if slug:
                rows.append({"slug": slug, "category": category})

        print(f"  Found {len(rows)} {kind}es on index page"
              if kind == "bonus" else f"  Found {len(rows)} {kind}s on index page")
        return rows

    def scrape_bonus_text(self, slug: str) -> dict:
        """Return {packet_bonus, leadin, parts} for one bonus detail page.

        The detail page holds the bonus in a ``div.md:basis-1/2`` whose direct
        children are, in order: an <h3> ("Packet N: Bonus M"), a leadin <div>,
        then three part <div>s each prefixed "[10e/m/h]" and containing a nested
        "ANSWER:" <div>, then metadata. We locate the h3's container and read its
        children positionally rather than by class name (which the site restyles).
        """
        url = f"{self.base_url}/bonus/{slug}"
        page = self.soup(url)

        h3 = page.find("h3")
        if h3 is None:
            return {"packet_bonus": None, "leadin": None, "parts": []}
        block = h3.parent
        packet_bonus = _clean(h3.get_text(" ", strip=True))

        leadin = None
        parts = []
        for child in block.find_all("div", recursive=False):
            text = _clean(child.get_text(" ", strip=True))
            if not text:
                continue
            m = re.match(r"^\[(10[emh])\]\s*", text)
            if m:
                value = m.group(1)
                # Split the part's own question text from its nested ANSWER div.
                ans_div = child.find(
                    lambda t: t.name == "div"
                    and t.get_text(" ", strip=True).upper().startswith("ANSWER:")
                )
                answer = None
                if ans_div is not None:
                    answer = _clean(ans_div.get_text(" ", strip=True))
                    answer = re.sub(r"^ANSWER:\s*", "", answer, flags=re.I)
                    ans_div.extract()  # remove so it doesn't pollute the question
                question = _clean(child.get_text(" ", strip=True))
                question = re.sub(r"^\[10[emh]\]\s*", "", question)
                parts.append({"value": value, "question": question, "answer": answer})
            elif text.startswith("<") or text.upper().startswith("ANSWER:"):
                # metadata line (e.g. "<BWEB - Current Events> | ...") — skip
                continue
            elif leadin is None:
                # First non-part, non-meta div is the leadin.
                leadin = text

        return {"packet_bonus": packet_bonus, "leadin": leadin, "parts": parts}

    def scrape_science_bonuses(self, science_prefix: str = DEFAULT_SCIENCE_PREFIX) -> list[dict]:
        """Scrape bonus index, filter to science, and pull text for each match."""
        index = self.scrape_index("bonus")
        science = [
            row
            for row in index
            if row["category"].strip().lower().startswith(science_prefix)
        ]
        print(f"  {len(science)} of {len(index)} bonuses are science "
              f"(prefix '{science_prefix}')")

        rows = []
        for i, row in enumerate(science, 1):
            detail = self.scrape_bonus_text(row["slug"])
            if not detail["parts"]:
                print(f"  ! no parts extracted for {row['slug']}")
                continue
            rows.append({
                "slug": row["slug"],
                "category": row["category"],
                **detail,
            })
            if i % 20 == 0 or i == len(science):
                print(f"  {i}/{len(science)} science bonuses fetched")
        return rows


def _clean(text: str) -> str:
    """Normalise whitespace and the common mojibake seen on these pages."""
    if text is None:
        return None
    # The site serves some smart quotes / dashes as U+FFFD replacement chars.
    # Collapse those and normalise whitespace; we can't recover the exact glyph,
    # but a single quote placeholder is enough for clue analysis.
    text = text.replace("�", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Scrape science tossup/bonus text")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Set base URL, e.g. https://<tournament>.vercel.app/set/<set-slug>",
    )
    parser.add_argument(
        "--type", choices=["tossup", "bonus"], default="tossup",
        help="Question type to scrape (default: tossup)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: science_tossups.json / science_bonuses.json)",
    )
    parser.add_argument(
        "--cache-dir",
        # Anchor to the repo's shared .cache regardless of the current working
        # directory, so cached HTML always lands inside the repo (and stays
        # gitignored) whether you run from here or from the repo root.
        default=str(_REPO_ROOT / ".cache"),
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_SCIENCE_PREFIX,
        help="Lowercase category prefix to filter on (default: 'science')",
    )
    parser.add_argument("--delay", type=float, default=0.6)
    args = parser.parse_args()

    output = args.output or (
        "science_bonuses.json" if args.type == "bonus" else "science_tossups.json"
    )

    scraper = TossupTextScraper(args.base_url, cache_dir=args.cache_dir, delay=args.delay)
    if args.type == "bonus":
        rows = scraper.scrape_science_bonuses(science_prefix=args.category.lower())
    else:
        rows = scraper.scrape_science_tossups(science_prefix=args.category.lower())

    Path(output).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plural = "bonuses" if args.type == "bonus" else "tossups"
    print(f"\nSaved {len(rows)} science {plural} to {output}")


if __name__ == "__main__":
    main()
