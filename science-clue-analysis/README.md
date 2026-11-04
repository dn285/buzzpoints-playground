# Science clue analysis

Pull the text of every **science** tossup *or bonus* from a buzzpoints set, then
have Claude flag problem clues:

- **tossups** — clues that are **non-unique** (could point to a different answer)
  or **erroneous** (factually wrong / misattributed), plus anything **suspect**.
- **bonuses** — **errors only** (bonuses aren't a buzzer race, so uniqueness isn't
  judged), plus anything **suspect**.

This folder is a downstream consumer of the existing buzzpoints scraper — it does
**not** re-implement the pulling logic. `scraper_text.py` imports and extends
`BuzzpointsScraper` from `../plackett-luce/scraper.py`, adding the text extraction
that the buzz-only scraper skips.

Everything this produces (`*.json`, `report*.md`, cached HTML in `../.cache`) is
gitignored, so nothing lands on git.

## Requirements

```bash
pip install anthropic requests beautifulsoup4
export ANTHROPIC_API_KEY=sk-ant-...        # required for the analysis step
```

## Run (per tournament)

Three steps per question type. Only the `--base-url` changes between tournaments.
`--type` (default `tossup`) selects the scraper mode and the analyzer's default
rubric/input/output; bonuses write to their own files.

```bash
cd science-clue-analysis

# --- TOSSUPS ---
python scraper_text.py --base-url https://<tournament>.vercel.app/set/<set-slug>
python analyze.py                                   # -> flags.json
python report.py                                    # -> report.md

# --- BONUSES ---
python scraper_text.py --type bonus --base-url https://<tournament>.vercel.app/set/<set-slug>
python analyze.py --type bonus                      # -> flags_bonus.json
python report.py --input flags_bonus.json --output report_bonus.md
```

Concrete example (2026 COOT):

```bash
BASE=https://buzzpoints-ukqb-2026-coot-67854uhtig.vercel.app/set/2026-coot
python scraper_text.py --base-url $BASE && python analyze.py && python report.py
python scraper_text.py --type bonus --base-url $BASE && python analyze.py --type bonus \
  && python report.py --input flags_bonus.json --output report_bonus.md
```

### Cheap smoke test

`analyze.py --limit N` only considers the first N items, so you can sanity-check
cost and output before running the whole set:

```bash
python analyze.py --limit 5                   # tossups
python analyze.py --type bonus --limit 5      # bonuses
```

## Resuming, re-runs, and watching progress live

`analyze.py` is **resumable by default**. It loads any existing `flags.json`,
**skips every tossup already recorded there**, and only calls the model for the
rest. So a re-run — or picking up after a crash / Ctrl-C — is free; it won't
re-pay for work already done.

- **Re-analyze everything** (e.g. after editing `rubric.md`): `--overwrite`.
- `flags.json` is rewritten after **every** tossup and again at the end, so it's
  always current even if the process is killed.

**Timing** is printed per tossup (`… (3.14s)`) and a run summary reports total +
average. Each record in `flags.json` also stores `elapsed_s`, so you can find the
slow questions after the fact.

**Watch results stream in live.** Every finished tossup is appended immediately
to a per-run log `runs/analyze-<timestamp>.jsonl` (one JSON object per line).
In a second terminal:

```bash
tail -f runs/analyze-*.jsonl
```

Each line is a complete result (slug, findings, `elapsed_s`), so you see problems
as they're found rather than waiting for the whole run. The `runs/` logs are kept
as a per-run history (gitignored); `flags.json` is the consolidated durable cache
that `report.py` reads.

## Tuning what gets flagged

The rubric is an **external file** that becomes Claude's system prompt — edit it
to change what counts as a defect, no code changes needed. There are two:
`rubric.md` (tossups) and `rubric_bonus.md` (bonuses, errors-only). `--type`
picks the right default; `--rubric` overrides it.

Note the bonus `FLAG` format has an extra `<part>` field (leadin/10e/10m/10h) so
the report can point you at the right part; the tossup format omits it. If you
change either pipe format, update `parse_flags()` in `analyze.py`.

For **per-question** guidance, add a `## per-question` section to the rubric with
`slug: note` lines. The matching note is appended to that item's prompt, so you
can tell the model exactly what to scrutinize on specific items:

```markdown
## per-question
- glycolysis: pay special attention to whether the Bifido-shunt clue is unique.
- anaphase: check the Pacman mechanism attribution carefully.
```

The slug is the last path segment of a tossup's URL (also the `slug` field in
`science_tossups.json`).

Use a different rubric file without touching the default:

```bash
python analyze.py --rubric my_tuned_rubric.md
```

## Files

| File                   | Role                                                              |
| ---------------------- | ----------------------------------------------------------------- |
| `scraper_text.py`      | Extends `BuzzpointsScraper`; pulls tossup/bonus text → JSON       |
| `rubric.md`            | Tossup system prompt + optional per-question notes                |
| `rubric_bonus.md`      | Bonus system prompt (errors-only) + optional per-question notes   |
| `analyze.py`           | Sends each item to Claude (Haiku 4.5); `--type` selects mode      |
| `report.py`            | Renders a flags cache → report markdown (auto-detects type)       |
| `science_tossups.json` / `science_bonuses.json` | Scraped text (generated, gitignored)     |
| `flags.json` / `flags_bonus.json` | Durable findings cache (generated, gitignored)         |
| `runs/*.jsonl`         | Per-run live logs, tail-friendly (generated, gitignored)          |
| `report.md` / `report_bonus.md` | Human-readable reports (generated, gitignored)           |

## What this is (and isn't)

A **recall-first screen**, not a verdict. It uses a cheap model to flag anything
glaring or even suspect, on the assumption that **you validate every flag
yourself**. Missing a real problem is the failure mode we optimize against; false
positives are expected and fine. Each flag is one of:

- **non-unique** — the clue could also fit a different answer (names the alternate)
- **error** — looks factually wrong / misattributed
- **suspect** — something just seems off; worth a human look

## Notes

- **Model / cost:** `claude-haiku-4-5`, **no extended thinking**, one plain-text
  call per tossup. Cost is dominated by output tokens, so we avoid thinking
  (billed as output) and keep responses to a few short `FLAG` lines — target is
  **well under 1¢/tossup**. The rubric is prompt-cached. If Haiku misses too much,
  bump `MODEL` in `analyze.py` to `claude-sonnet-4-6` (~3× the cost) or add a small
  thinking budget — but that's the expensive knob, so change it deliberately.
- **Output format:** the model returns `CLEAN` or `FLAG | <type> | <quote> |
  <reason>` lines, parsed leniently by `parse_flags()` in `analyze.py`. There is
  no structured-output schema (it added output cost and was the likely cause of
  earlier per-tossup errors). Each record also stores the model's `raw` text.
- **Errors:** a tossup that errors is cached with empty findings and surfaced in
  the run summary and the report; re-run with `--overwrite` to retry it.
- **Category filter:** matches index categories starting with `science`
  (Biology / Chemistry / Earth / Physics / Other). Override with
  `scraper_text.py --category <prefix>` if a set labels science differently.
- **Caching:** raw HTML is cached under `../.cache`, so re-running the scraper for
  the same tournament makes no new network requests.
