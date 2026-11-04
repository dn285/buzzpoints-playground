"""
LLM-judged clue analysis for science tossups.

Reads ``science_tossups.json`` (produced by scraper_text.py), sends each tossup's
question + answerline to Claude, and asks it to flag two failure modes:

  * non-unique clues  — a clue sentence that could point to a DIFFERENT answer
                        than the intended one (breaks pyramidality / buzzer race)
  * erroneous clues   — a clue that is factually wrong or misattributed

Model: claude-haiku-4-5, no extended thinking, one plain-text call per tossup.
This is a deliberately cheap recall-first screen (target < 1 cent/tossup): a human
validates every flag afterward, so we optimize for not missing glaring problems
rather than for careful adjudication. Cost is dominated by output tokens, so we
avoid extended thinking (billed as output) and keep the response to a few short
FLAG lines. The rubric lives in an EXTERNAL file (rubric.md) and is prompt-cached.

Per-question guidance: add a "## per-question" section to the rubric file with
`slug: note` lines (see rubric.md). The matching note is appended to that
question's prompt, so you can tell the model what to focus on for specific items.

Resumable by default: analyze.py loads any existing flags.json, skips every
tossup already recorded there, and only calls the model for the rest. So a
re-run (or a crash-resume) is free. Use --overwrite to re-analyze everything
(e.g. after editing the rubric).

Live progress: every finished tossup is appended immediately (line-buffered,
flushed) to a per-run log runs/analyze-<timestamp>.jsonl — one JSON object per
line — so you can watch results stream in with:

    tail -f runs/analyze-*.jsonl

The durable cache flags.json is rewritten after each tossup AND at the end, so
it is always current even if the process is killed.

Output:
    runs/analyze-<ts>.jsonl  live per-tossup log (JSON Lines, tail-friendly)
    flags.json               consolidated durable cache (JSON array)
report.py turns flags.json into report.md.

Requires ANTHROPIC_API_KEY in the environment.

Usage:
    python analyze.py                       # tossups: analyze new, resume the rest
    python analyze.py --type bonus          # bonuses (science_bonuses.json -> flags_bonus.json)
    python analyze.py --overwrite           # re-analyze everything
    python analyze.py --limit 5             # first 5 only (cheap smoke test)
    python analyze.py --rubric myrubric.md  # use a tuned rubric
    python analyze.py --input other.json --output other_flags.json

--type selects the default rubric/input/output; any of --rubric/--input/--output
overrides its default.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5"
_HERE = Path(__file__).resolve().parent
RUNS_DIR = _HERE / "runs"

# Per-type defaults for rubric / input / output. --rubric/--input/--output
# override any of these.
TYPE_DEFAULTS = {
    "tossup": {
        "rubric": _HERE / "rubric.md",
        "input": "science_tossups.json",
        "output": "flags.json",
    },
    "bonus": {
        "rubric": _HERE / "rubric_bonus.md",
        "input": "science_bonuses.json",
        "output": "flags_bonus.json",
    },
}
# Backwards-compatible name used elsewhere / by external callers.
DEFAULT_RUBRIC = TYPE_DEFAULTS["tossup"]["rubric"]

# Valid flag types the rubric asks for. Anything else the model emits is kept
# verbatim under "suspect" so we never silently drop a flag.
KNOWN_TYPES = {"non-unique", "non_unique", "error", "erroneous", "suspect"}
TYPE_CANONICAL = {
    "non-unique": "non_unique", "non_unique": "non_unique",
    "error": "error", "erroneous": "error",
    "suspect": "suspect",
}
# Bonus part labels; used to recognise the optional <part> field.
PART_LABELS = {"leadin", "10e", "10m", "10h"}


def parse_flags(text: str, has_part: bool = False) -> list[dict]:
    """Parse the model's plain-text response into a list of finding dicts.

    Tossup format (has_part=False), one per line:
        FLAG | <type> | <verbatim quote> | <reason>
    Bonus format (has_part=True) adds a part locator as the second field:
        FLAG | <type> | <part> | <verbatim quote> | <reason>

    A response of just "CLEAN" (or empty) yields no findings.

    Lenient on purpose: this is a recall-first screen, so we'd rather keep a
    malformed-but-present flag than drop it. Any line that looks like a flag is
    kept; missing fields degrade gracefully. Each finding always has a `part`
    key ("" for tossups) so downstream code is uniform.
    """
    findings = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.upper() == "CLEAN":
            continue
        # Only lines that look like a flag; ignore stray prose.
        if "flag" not in line.lower() and "|" not in line:
            continue

        fields = [p.strip() for p in line.split("|")]
        # Drop a leading "FLAG" token if present.
        if fields and fields[0].lower().startswith("flag"):
            fields = fields[1:]

        ftype = TYPE_CANONICAL.get(fields[0].lower(), "suspect") if fields else "suspect"

        part = ""
        rest = fields[1:]
        if has_part:
            # Next field is the part locator when it looks like one; if the model
            # omitted it, don't consume a real field.
            if rest and rest[0].lower() in PART_LABELS:
                part = rest[0].lower()
                rest = rest[1:]

        quote = rest[0] if len(rest) >= 1 else ""
        reason = " | ".join(rest[1:]) if len(rest) >= 2 else ""

        findings.append({
            "type": ftype,
            "part": part,
            "quote": quote,
            "reason": reason,
        })
    return findings


def load_rubric(path: Path) -> tuple[str, dict]:
    """Load the rubric file.

    Returns (system_prompt, per_question_notes). The system prompt is everything
    in the file EXCEPT the "## per-question" section (and HTML comments, which are
    tuning notes for humans). per_question_notes maps slug -> note string.
    """
    text = path.read_text(encoding="utf-8")

    # Strip HTML comments (human tuning notes) from the prompt.
    text_no_comments = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    per_question = {}
    # Split off a "## per-question" section if present.
    m = re.search(r"^##\s*per-question\s*$", text_no_comments, flags=re.I | re.M)
    if m:
        system_part = text_no_comments[: m.start()]
        pq_part = text_no_comments[m.end():]
        for line in pq_part.splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or ":" not in line:
                continue
            slug, note = line.split(":", 1)
            per_question[slug.strip()] = note.strip()
    else:
        system_part = text_no_comments

    return system_part.strip(), per_question


def format_item(item: dict, item_type: str) -> str:
    """Render one tossup/bonus into the user-message body."""
    if item_type == "bonus":
        lines = [f"Category: {item['category']}", ""]
        if item.get("leadin"):
            lines.append(f"Leadin: {item['leadin']}")
            lines.append("")
        for p in item.get("parts", []):
            lines.append(f"[{p['value']}] {p['question']}")
            lines.append(f"ANSWER: {p['answer']}")
            lines.append("")
        return "\n".join(lines).strip()
    # tossup
    return (
        f"Category: {item['category']}\n"
        f"Intended answer: {item['answer']}\n\n"
        f"Question text:\n{item['question']}"
    )


def analyze_one(client, item: dict, system_prompt: str, note: str | None,
                item_type: str = "tossup") -> dict:
    """One cheap plain-text call per item (tossup or bonus); return {findings, raw}.

    No extended thinking and no structured-output schema — both drive output-token
    cost and the schema+thinking combo was the likely source of the earlier errors.
    We ask for plain FLAG lines (rubric*.md) and parse them leniently.
    """
    user_content = format_item(item, item_type)
    if note:
        user_content += f"\n\nExtra guidance for THIS question:\n{note}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,  # a handful of short FLAG lines; caps worst-case cost
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},  # rubric is stable → cache it
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    return {"findings": parse_flags(raw, has_part=(item_type == "bonus")), "raw": raw}


def load_cache(path: Path) -> dict:
    """Load existing flags.json into a slug -> record dict (empty if absent)."""
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["slug"]: r for r in records}


def write_cache(path: Path, cache: dict, order: list[str]):
    """Write the cache back as a JSON array, preserving scrape order where known.

    ``order`` is the slug order from the input file; any cached slug not in it
    (e.g. left over from a previous input) is appended afterward so nothing is
    dropped.
    """
    seen = set()
    ordered = []
    for slug in order:
        if slug in cache:
            ordered.append(cache[slug])
            seen.add(slug)
    for slug, rec in cache.items():
        if slug not in seen:
            ordered.append(rec)
    path.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="LLM-judge science tossup/bonus clues")
    parser.add_argument("--type", choices=["tossup", "bonus"], default="tossup",
                        help="Question type to analyze (default: tossup). Selects the "
                             "default rubric/input/output unless you override them.")
    parser.add_argument("--input", default=None,
                        help="Input JSON (default: science_tossups.json / science_bonuses.json)")
    parser.add_argument("--output", default=None,
                        help="Cache JSON (default: flags.json / flags_bonus.json)")
    parser.add_argument("--rubric", default=None,
                        help="Rubric file that becomes the system prompt "
                             "(default: rubric.md / rubric_bonus.md)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only consider the first N items (for a cheap test)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-analyze every item, ignoring cached results")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between calls (rate-limit cushion)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in the environment. "
                 "Set it and re-run:  export ANTHROPIC_API_KEY=sk-ant-...")

    defaults = TYPE_DEFAULTS[args.type]
    rubric_path = Path(args.rubric) if args.rubric else defaults["rubric"]
    input_path = Path(args.input) if args.input else Path(defaults["input"])
    cache_path = Path(args.output) if args.output else Path(defaults["output"])

    system_prompt, per_question = load_rubric(rubric_path)
    print(f"Type: {args.type} | rubric {rubric_path.name} "
          f"({len(system_prompt)} chars, {len(per_question)} per-question notes)")

    items = json.loads(input_path.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    input_order = [t["slug"] for t in items]

    cache = {} if args.overwrite else load_cache(cache_path)

    # Split into what needs analyzing vs. what's already cached.
    todo = [t for t in items if t["slug"] not in cache]
    skipped = len(items) - len(todo)
    if args.overwrite:
        print(f"--overwrite: re-analyzing all {len(items)} {args.type}s")
    else:
        print(f"{len(todo)} to analyze, {skipped} already cached (skipped)")

    if not todo:
        print("Nothing to do. Use --overwrite to re-analyze.")
        return

    # Per-run JSONL log for live tailing: one finished tossup per line, flushed
    # immediately so `tail -f runs/analyze-*.jsonl` shows results as they land.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Millisecond precision so two runs in the same second get distinct files.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    log_path = RUNS_DIR / f"analyze-{ts}.jsonl"
    print(f"Live log (tail -f):  {log_path}\n")

    client = anthropic.Anthropic()
    run_start = time.time()

    with open(log_path, "w", encoding="utf-8") as log:
        for i, item in enumerate(todo, 1):
            note = per_question.get(item["slug"])
            t0 = time.time()
            try:
                analysis = analyze_one(client, item, system_prompt, note, args.type)
            except Exception as e:  # keep going; one bad item shouldn't kill the run
                print(f"  ! error on {item['slug']}: {e}")
                analysis = {"findings": [], "raw": "", "error": str(e)}
            elapsed = round(time.time() - t0, 2)

            n = len(analysis.get("findings", []))
            # Common fields plus a type-neutral "packet" label; keep the original
            # type-specific fields so report.py can show questions/answers/parts.
            record = {
                "type": args.type,
                "slug": item["slug"],
                "category": item["category"],
                "packet": item.get("packet_tossup") or item.get("packet_bonus"),
                "elapsed_s": elapsed,
                "analysis": analysis,
            }
            if args.type == "bonus":
                record["leadin"] = item.get("leadin")
                record["parts"] = item.get("parts", [])
            else:
                record["answer"] = item.get("answer")
                record["question"] = item.get("question")

            # 1) stream to the live JSONL log and flush so tail -f sees it now
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            os.fsync(log.fileno())

            # 2) fold into the durable cache and rewrite it (crash-safe)
            cache[item["slug"]] = record
            write_cache(cache_path, cache, input_order)

            flag_note = f"{n} finding(s)" if n else "clean"
            print(f"  [{i}/{len(todo)}] {item['slug']}: {flag_note} ({elapsed}s)")
            if args.sleep:
                time.sleep(args.sleep)

    total_elapsed = round(time.time() - run_start, 1)
    n_flagged = sum(1 for r in cache.values() if r["analysis"].get("findings"))
    n_errors = sum(1 for r in cache.values() if r["analysis"].get("error"))
    avg = round(total_elapsed / len(todo), 2) if todo else 0
    plural = "bonuses" if args.type == "bonus" else "tossups"
    print(f"\nAnalyzed {len(todo)} {plural} in {total_elapsed}s (avg {avg}s/tossup).")
    print(f"{n_flagged} of {len(cache)} cached {plural} have at least one finding.")
    if n_errors:
        print(f"WARNING: {n_errors} {args.type}(s) errored (cached with empty findings; "
              f"re-run with --overwrite to retry).")
    print(f"Live log:      {log_path}")
    print(f"Durable cache: {cache_path}")


if __name__ == "__main__":
    main()
