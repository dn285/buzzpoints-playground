"""
Find every animal mention in corpus.json (from fetch.py) using lexicon.py.

Matching rules:
- Whole-word, case-insensitive, longest phrase first ("sea lion" is one
  mention, not also a "lion"). Letters-only boundaries, so "monkey's" and
  "monkey-like" count but "dragonfly" never counts as "dragon".
- Pronunciation guides like (“bee-OH-shuh”) are removed first; otherwise
  they produce fake hits ("bee", "ram", "duck", "bat").
- STOP_PHRASES ("papal bull", "bat mitzvah", ...) are matched and discarded.
- ``lower_only`` terms must be lowercase ("turkey", not "Turkey").
- All-caps matches are acronyms and are skipped ("CAT theory", "DOG1").
- ``noun_only`` terms in the question text must be tagged NOUN/PROPN/ADJ by
  spaCy, which drops verb uses ("bear no grudge", "the flag flies", "ram and
  sink") but keeps names. Answer lines are not tagged (always noun phrases).

Both the question text and the primary answer are searched; a question whose
answer is "monkeys" is a monkey question even if the text says "these animals".

Output: mentions.csv (one row per mention) and questions.csv (one row per
question with its word count), both consumed by report.py.

Usage:
    python count.py [--corpus corpus.json]
"""

import argparse
import csv
import json
import re

import spacy

from lexicon import STOP_PHRASES, entries

# Letters, accented letters, combining accents, and the "·" syllable divider
# quizbowl writers use in long words ("aldol·ase", "Ovon·ram·wen").
LETTERS = "A-Za-zÀ-ɏ̀-ͯ·"

# Parenthetical pronunciation guides: (“bee-OH-shuh”), ("ram"), (REE-strud).
PRON_QUOTED = re.compile(r"\(\s*[“\"][^()]*?[”\"]\s*\)")
PRON_BARE = re.compile(r"\((?=[^()]*-)(?=[^()]*[A-Z]{2})[A-Za-z’'\-\s·]+\)")


# ADJ keeps hyphenated compounds the tagger calls adjectives ("rat-catching").
KEEP_POS = ("NOUN", "PROPN", "ADJ")


def clean(text: str) -> str:
    text = text.replace("(*)", " ")  # power mark; confuses the POS tagger
    text = PRON_QUOTED.sub(" ", text)
    text = PRON_BARE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_matcher():
    lex = {e["form"].lower(): e for e in entries()}
    for p in STOP_PHRASES:
        lex[p.lower()] = None
    alt = "|".join(re.escape(f) for f in sorted(lex, key=len, reverse=True))
    rx = re.compile(rf"(?<![{LETTERS}])(?:{alt})(?![{LETTERS}])", re.I)
    return rx, lex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.json")
    args = ap.parse_args()

    corpus = json.load(open(args.corpus, encoding="utf-8"))
    rx, lex = build_matcher()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    # Pass 1: regex candidates; collect texts that need POS tagging.
    candidates, to_tag = [], {}
    questions = []
    for qi, q in enumerate(corpus):
        fields = {"text": clean(q["text"]), "answer": clean(q["answer"])}
        questions.append({
            "set": q["set"], "set_name": q["set_name"], "level": q["level"],
            "date": q["date"], "kind": q["kind"], "id": q["id"], "round": q["round"],
            "category": q["category"],
            "words": sum(len(v.split()) for v in fields.values()),
        })
        for field, s in fields.items():
            for m in rx.finditer(s):
                e = lex[m.group(0).lower()]
                if e is None:
                    continue  # stop phrase
                if e["lower_only"] and m.group(0) != m.group(0).lower():
                    continue
                # All-caps = acronym (CAT theory, COBRA art group, DOG1 gene).
                if len(m.group(0)) > 1 and m.group(0).isupper():
                    continue
                c = {"qi": qi, "field": field, "start": m.start(), "end": m.end(),
                     "match": m.group(0), **e}
                candidates.append(c)
                # Answer lines are noun phrases with no sentence around them
                # (the tagger calls a bare "crows" a verb), so only tag text.
                if e["noun_only"] and field == "text":
                    to_tag[(qi, field)] = s

    # Pass 2: POS-tag only the texts that contain a noun_only candidate.
    keys = list(to_tag)
    docs = dict(zip(keys, nlp.pipe((to_tag[k] for k in keys), batch_size=64)))

    kept, dropped_verb = [], 0
    for c in candidates:
        if c["noun_only"] and c["field"] == "text":
            doc = docs[(c["qi"], c["field"])]
            span = doc.char_span(c["start"], c["end"], alignment_mode="expand")
            if span is None or span[-1].pos_ not in KEEP_POS:
                dropped_verb += 1
                continue
        kept.append(c)

    with open("mentions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "kind", "id", "category", "field", "group", "lemma",
                    "match", "mythical", "context"])
        for c in kept:
            q, s = corpus[c["qi"]], (clean(corpus[c["qi"]][c["field"]]))
            ctx = s[max(0, c["start"] - 70):c["end"] + 50]
            w.writerow([q["set"], q["kind"], q["id"], q["category"], c["field"],
                        c["group"], c["lemma"], c["match"], int(c["mythical"]), ctx])

    with open("questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(questions[0]))
        w.writeheader()
        w.writerows(questions)

    print(f"{len(candidates)} candidate matches, {dropped_verb} dropped as non-noun, "
          f"{len(kept)} mentions kept -> mentions.csv")
    print(f"{len(questions)} questions, {sum(q['words'] for q in questions):,} words "
          f"-> questions.csv")


if __name__ == "__main__":
    main()
