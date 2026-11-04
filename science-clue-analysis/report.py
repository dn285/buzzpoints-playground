"""
Render a flags cache (from analyze.py) into a human-readable report.md.

Works for both tossup and bonus runs — the item type is auto-detected from the
records (a `type` field; records without one are treated as tossups for backward
compatibility). This is a recall-first screen, so the report is a worklist: every
flagged item is listed for a human to validate. Clean items are summarised in a
footer so nothing is silently dropped.

Each finding has a `type` (error / non_unique / suspect), a `quote`, a `reason`,
and — for bonuses — a `part` locator (leadin / 10e / 10m / 10h). There are no
severities; the model flags anything glaring or suspect and you validate.

Usage:
    python report.py                                   # flags.json -> report.md
    python report.py --input flags_bonus.json --output report_bonus.md
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TYPE_LABEL = {"non_unique": "Non-unique", "error": "Error", "suspect": "Suspect"}
# Ordering within an item: hard errors first, then ambiguity, then vibes.
TYPE_ORDER = {"error": 0, "non_unique": 1, "suspect": 2}


def _item_type(results: list[dict]) -> str:
    """Detect whether this cache is tossups or bonuses."""
    for r in results:
        if r.get("type") in ("tossup", "bonus"):
            return r["type"]
        if "parts" in r:  # old bonus record without an explicit type
            return "bonus"
    return "tossup"


def _packet(r: dict) -> str:
    """Type-neutral packet label, tolerant of old and new record shapes."""
    return r.get("packet") or r.get("packet_tossup") or r.get("packet_bonus") or r["slug"]


def _answer_summary(r: dict, item_type: str) -> str:
    """Short answer descriptor for headers/footers."""
    if item_type == "bonus":
        answers = [p.get("answer", "") for p in r.get("parts", [])]
        # Keep it compact: strip acceptable-answer brackets for the summary.
        short = [a.split("[")[0].strip() for a in answers if a]
        return " / ".join(short) if short else ""
    return r.get("answer", "")


def _finding_line(f: dict, item_type: str) -> list[str]:
    """Render one finding as markdown lines."""
    label = TYPE_LABEL.get(f.get("type"), f.get("type", "flag"))
    part = f.get("part", "")
    quote = f.get("quote", "")
    reason = f.get("reason", "")
    loc = f"[{part}] " if (item_type == "bonus" and part) else ""
    out = []
    if quote:
        out.append(f"- **{label}:** {loc}\"{quote}\"")
    else:
        out.append(f"- **{label}**{(' ' + loc).rstrip() if loc else ''}")
    if reason:
        out.append(f"  - {reason}")
    return out


def render(results: list[dict]) -> str:
    item_type = _item_type(results)
    noun = item_type  # "tossup" or "bonus"
    plural = "bonuses" if item_type == "bonus" else "tossups"
    Plural = plural.capitalize()

    total = len(results)
    flagged = [r for r in results if r["analysis"].get("findings")]
    errored = [r for r in results if r["analysis"].get("error")]
    clean = total - len(flagged) - len([r for r in errored if not r["analysis"].get("findings")])

    by_type = Counter()
    for r in flagged:
        for f in r["analysis"]["findings"]:
            by_type[f.get("type", "suspect")] += 1

    lines = []
    lines.append(f"# Science {noun} clue analysis")
    lines.append("")
    lines.append("*Recall-first screen — validate every flag independently.*")
    lines.append("")
    lines.append(f"- {Plural} analyzed: **{total}**")
    lines.append(f"- {Plural} with at least one flag: **{len(flagged)}**")
    lines.append(f"- Clean {plural}: **{clean}**")
    # For bonuses the non_unique lens is off by default, but count it if present.
    lines.append(f"- Total flags: **{sum(by_type.values())}** "
                 f"({by_type.get('error', 0)} error, "
                 f"{by_type.get('non_unique', 0)} non-unique, "
                 f"{by_type.get('suspect', 0)} suspect)")
    if errored:
        lines.append(f"- ⚠ {Plural} that errored during analysis: **{len(errored)}** "
                     f"(re-run with --overwrite to retry)")
    lines.append("")

    # Group flagged items by subcategory.
    by_cat = defaultdict(list)
    for r in flagged:
        by_cat[r["category"]].append(r)

    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines.append("")
        cat_rows = sorted(by_cat[cat], key=lambda r: -len(r["analysis"]["findings"]))
        for r in cat_rows:
            ans = _answer_summary(r, item_type)
            suffix = f" — answer: {ans}" if item_type == "tossup" and ans else (
                f" — {ans}" if ans else "")
            lines.append(f"### {_packet(r)}{suffix}")
            lines.append("")
            findings = sorted(
                r["analysis"]["findings"],
                key=lambda f: TYPE_ORDER.get(f.get("type"), 3),
            )
            for f in findings:
                lines.extend(_finding_line(f, item_type))
            lines.append("")

    # Footer: errored items (need a retry), then clean ones.
    if errored:
        lines.append("## Errored (no result — retry with --overwrite)")
        lines.append("")
        for r in sorted(errored, key=lambda r: r["category"]):
            ans = _answer_summary(r, item_type)
            err = r["analysis"].get("error", "")
            tail = f" — {ans}" if ans else ""
            lines.append(f"- {r['category']}: {_packet(r)}{tail} — `{err}`")
        lines.append("")

    clean_rows = [r for r in results
                  if not r["analysis"].get("findings")
                  and not r["analysis"].get("error")]
    if clean_rows:
        lines.append(f"## Clean {plural} (no flags)")
        lines.append("")
        for r in sorted(clean_rows, key=lambda r: r["category"]):
            ans = _answer_summary(r, item_type)
            tail = f" — {ans}" if ans else ""
            lines.append(f"- {r['category']}: {_packet(r)}{tail}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render clue-analysis report")
    parser.add_argument("--input", default="flags.json")
    parser.add_argument("--output", default="report.md")
    args = parser.parse_args()

    results = json.loads(Path(args.input).read_text(encoding="utf-8"))
    Path(args.output).write_text(render(results), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
