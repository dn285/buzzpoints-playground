"""
Buzzpoints scraper: fetches tossup-level buzz data from a Next.js buzzpoints site.

Usage:
    python scraper.py --base-url https://2026-nationals.vercel.app/set/2026-acf-nationals
    python scraper.py --base-url https://... --output data.csv --cache-dir .cache
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_DELAY = 0.6  # seconds between requests


class BuzzpointsScraper:
    def __init__(self, base_url: str, cache_dir: str = ".cache", delay: float = DEFAULT_DELAY):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "buzzpoints-research-scraper/1.0"})
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        safe = url.replace("://", "__").replace("/", "_").replace("?", "_q_")
        return self.cache_dir / (safe[:200] + ".html")

    def fetch(self, url: str) -> str:
        path = self._cache_path(url)
        if path.exists():
            return path.read_text(encoding="utf-8")
        time.sleep(self.delay)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        path.write_text(resp.text, encoding="utf-8")
        return resp.text

    def soup(self, url: str) -> BeautifulSoup:
        return BeautifulSoup(self.fetch(url), "html.parser")

    # ------------------------------------------------------------------
    # Step 1a: tossup index
    # ------------------------------------------------------------------

    def scrape_tossup_index(self) -> list[dict]:
        """Returns list of {slug, category}."""
        url = f"{self.base_url}/tossup"
        print(f"Fetching tossup index: {url}")
        page = self.soup(url)

        rows = []
        table = page.find("table")
        if table is None:
            raise RuntimeError("No table found on tossup index page")

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        cat_idx = next((i for i, h in enumerate(headers) if "category" in h), None)
        if cat_idx is None:
            # fallback: first column
            cat_idx = 0

        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if not tds:
                continue
            # Category
            category = tds[cat_idx].get_text(strip=True) if cat_idx < len(tds) else ""
            # Answer link -> slug
            slug = None
            for td in tds:
                a = td.find("a", href=True)
                if a and "/tossup/" in a["href"]:
                    href = a["href"].split("?")[0]
                    slug = href.rstrip("/").rsplit("/", 1)[-1]
                    break
            if slug:
                rows.append({"slug": slug, "category": category})

        print(f"  Found {len(rows)} tossups on index page")
        return rows

    # ------------------------------------------------------------------
    # Step 1b: tossup detail pages
    # ------------------------------------------------------------------

    def scrape_tossup_detail(self, slug: str) -> list[dict]:
        """Returns list of buzz dicts for one tossup."""
        url = f"{self.base_url}/tossup/{slug}"
        page = self.soup(url)

        buzzes = []
        # Find the Buzzes table — look for a table with a "Buzzpoint" or "Value" header
        for table in page.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            if not any("buzz" in h or "value" in h or "player" in h for h in headers):
                continue

            # Map header names to column indices
            col = {h: i for i, h in enumerate(headers)}

            for tr in table.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if not tds:
                    continue

                def cell(name, fallback=None):
                    idx = col.get(name)
                    if idx is None:
                        # try partial match
                        idx = next((i for h, i in col.items() if name in h), None)
                    if idx is None or idx >= len(tds):
                        return fallback
                    return tds[idx]

                player_td = cell("player")
                team_td = cell("team")
                opp_td = cell("opponent")
                bp_td = cell("buzzpoint")
                val_td = cell("value")

                if player_td is None:
                    continue

                # Extract player slug
                player_name = player_td.get_text(strip=True)
                player_slug = None
                pa = player_td.find("a", href=True)
                if pa and "/player/" in pa["href"]:
                    player_slug = pa["href"].rstrip("/").rsplit("/", 1)[-1]

                # Team slug
                team_name = team_td.get_text(strip=True) if team_td else ""
                team_slug = None
                if team_td:
                    ta = team_td.find("a", href=True)
                    if ta and "/team/" in ta["href"]:
                        team_slug = ta["href"].rstrip("/").rsplit("/", 1)[-1]

                # Opponent slug
                opp_name = opp_td.get_text(strip=True) if opp_td else ""
                opp_slug = None
                if opp_td:
                    oa = opp_td.find("a", href=True)
                    if oa and "/team/" in oa["href"]:
                        opp_slug = oa["href"].rstrip("/").rsplit("/", 1)[-1]

                # Buzzpoint
                bp_text = bp_td.get_text(strip=True) if bp_td else ""
                buzzpoint = None
                try:
                    buzzpoint = int(bp_text)
                except (ValueError, TypeError):
                    pass

                # Value
                val_text = val_td.get_text(strip=True) if val_td else ""
                buzz_value = None
                try:
                    buzz_value = int(val_text)
                except (ValueError, TypeError):
                    pass

                if player_name:
                    buzzes.append({
                        "player_name": player_name,
                        "player_slug": player_slug,
                        "team_name": team_name,
                        "team_slug": team_slug,
                        "opponent_slug": opp_slug,
                        "buzzpoint": buzzpoint,
                        "buzz_value": buzz_value,
                    })
            break  # use first matching table

        return buzzes

    # ------------------------------------------------------------------
    # Step 1c: team pages — roster
    # ------------------------------------------------------------------

    def scrape_team_roster(self, team_slug: str) -> list[str]:
        """Returns list of player_slugs on this team."""
        url = f"{self.base_url}/team/{team_slug}"
        page = self.soup(url)

        slugs = []
        for a in page.find_all("a", href=True):
            if "/player/" in a["href"]:
                slug = a["href"].rstrip("/").rsplit("/", 1)[-1]
                if slug and slug not in slugs:
                    slugs.append(slug)
        return slugs

    # ------------------------------------------------------------------
    # Main scrape pipeline
    # ------------------------------------------------------------------

    def scrape_all(self) -> list[dict]:
        """
        Returns a flat list of (tossup, game, player) rows.
        Each row: tossup_slug, game_id, category, player_slug, team_slug,
                  opponent_slug, buzzpoint, buzz_value, won

        A single tossup is played in many games (round-robin format), so
        we group buzzes by (team, opponent) pair to get one record per game.
        """
        # 1a: index
        tossup_index = self.scrape_tossup_index()

        # 1b: detail pages
        all_buzzes: dict[str, list[dict]] = {}  # slug -> buzz list
        print(f"\nScraping {len(tossup_index)} tossup detail pages...")
        for i, entry in enumerate(tossup_index, 1):
            slug = entry["slug"]
            buzzes = self.scrape_tossup_detail(slug)
            all_buzzes[slug] = buzzes
            if i % 25 == 0 or i == len(tossup_index):
                print(f"  {i}/{len(tossup_index)} tossups fetched")

        # Collect unique team slugs from all buzzes
        team_slugs: set[str] = set()
        for buzzes in all_buzzes.values():
            for b in buzzes:
                if b["team_slug"]:
                    team_slugs.add(b["team_slug"])
                if b["opponent_slug"]:
                    team_slugs.add(b["opponent_slug"])

        # 1c: team rosters
        print(f"\nScraping {len(team_slugs)} team pages for rosters...")
        rosters: dict[str, list[str]] = {}
        for i, ts in enumerate(sorted(team_slugs), 1):
            rosters[ts] = self.scrape_team_roster(ts)
            if i % 10 == 0 or i == len(team_slugs):
                print(f"  {i}/{len(team_slugs)} teams fetched")

        # Build player->team mapping (a player is always on one team at a tournament)
        player_team: dict[str, str] = {}
        for ts, players in rosters.items():
            for ps in players:
                player_team[ps] = ts
        # Buzz data is more reliable than roster scrape for player->team
        for buzzes in all_buzzes.values():
            for b in buzzes:
                if b["player_slug"] and b["team_slug"]:
                    player_team[b["player_slug"]] = b["team_slug"]

        # 1d: build flat rows, one game at a time
        rows = []
        total_games = 0

        for entry in tossup_index:
            slug = entry["slug"]
            category = entry["category"]
            buzzes = all_buzzes.get(slug, [])

            if not buzzes:
                continue

            # Group buzzes by game: frozenset({team_slug, opponent_slug}).
            # Each unique pair is one game playing this tossup.
            games: dict[frozenset, list[dict]] = {}
            for b in buzzes:
                if b["team_slug"] and b["opponent_slug"]:
                    key = frozenset({b["team_slug"], b["opponent_slug"]})
                    games.setdefault(key, []).append(b)

            if not games:
                continue

            for game_key, game_buzzes in games.items():
                team_a, team_b = sorted(game_key)
                game_id = f"{team_a}__{team_b}"
                total_games += 1

                # All players in the room = union of the two rosters
                roster_a = set(rosters.get(team_a, []))
                roster_b = set(rosters.get(team_b, []))
                all_in_room = roster_a | roster_b

                # Augment with anyone who actually buzzed in this game
                for b in game_buzzes:
                    if b["player_slug"]:
                        all_in_room.add(b["player_slug"])

                # Winner = first correct buzz in this game (table is sorted by buzzpoint)
                winner_slug = None
                for b in sorted(game_buzzes, key=lambda x: x["buzzpoint"] or 9999):
                    if b["buzz_value"] in (10, 15) and b["player_slug"]:
                        winner_slug = b["player_slug"]
                        break

                # Build buzz lookup for this game's players
                buzz_by_player = {b["player_slug"]: b for b in game_buzzes if b["player_slug"]}

                for ps in all_in_room:
                    b = buzz_by_player.get(ps)
                    ts = player_team.get(ps) or (team_a if ps in roster_a else team_b)
                    opp = team_b if ts == team_a else team_a
                    rows.append({
                        "tossup_slug": slug,
                        "game_id": game_id,
                        "category": category,
                        "player_slug": ps,
                        "team_slug": ts,
                        "opponent_slug": opp,
                        "buzzpoint": b["buzzpoint"] if b else None,
                        "buzz_value": b["buzz_value"] if b else None,
                        "won": (ps == winner_slug),
                    })

        print(f"\nTotal games: {total_games}")
        print(f"Total rows (game x tossup x player triples): {len(rows)}")
        return rows


def save_csv(rows: list[dict], path: str):
    if not rows:
        print("No rows to save.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape buzzpoints tossup data")
    parser.add_argument("--base-url", required=True, help="Base URL e.g. https://2026-nationals.vercel.app/set/2026-acf-nationals")
    parser.add_argument("--output", default="../buzzpoints_data.csv", help="Output CSV path")
    parser.add_argument("--cache-dir", default="../.cache", help="Directory to cache raw HTML")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests")
    args = parser.parse_args()

    scraper = BuzzpointsScraper(args.base_url, cache_dir=args.cache_dir, delay=args.delay)
    rows = scraper.scrape_all()
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()
