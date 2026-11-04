# Animal mentions

Did 2026 Chicago Open mention monkeys (or any animal) more than other sets? Counts
animal words across every public college/open set on buzzpoints.buzz and compares
rates. No LLM calls; everything runs locally.

## Requirements

```bash
pip install requests beautifulsoup4 pandas numpy spacy
python -m spacy download en_core_web_sm
```

## Run

```bash
cd animal-mentions
python fetch.py    # -> corpus.json   (~80 requests, cached in ../.cache, 1.5 s apart)
python count.py    # -> mentions.csv, questions.csv
python report.py   # -> report.md, per_set.csv   (--focus <set-slug> to change the focus set)
python monkey_claims.py  # -> monkey_claims.md (follow-up: focus set vs its series and its season)
python visual_report.py  # -> monkey_visuals.html (sparklines, heat grids, cooldown curve)
```

`visual_report.py` shows how monkey mentions are distributed inside each set and
re-scores "perceived" density with a cooldown: within a packet, a monkey question
counts only if at least k other questions were read since the last counted one
(read order: tossup n, then bonus n). The cooldown idea follows Hintzman (1969):
judged frequency rises with the spacing of repetitions and levels off after
about 3-4 intervening items.

`monkey_claims.py` checks two claims about the focus set's monkey/ape (`primates`)
mentions: (1) more than any other edition of its series (`--series "Chicago Open"`),
and (2) the most among sets from its season (`--season-start/--season-end`, default
Sept 2025 – Aug 2026) with at least `--min-packets` (default 10) packets. Sets are
ranked by **mentions per packet** (the absolute hits a player hears in a packet,
rather than per word). Robustness checks: raw
count, questions with a mention per packet, and per packet without the set's single
most monkey-heavy question. Every mention is listed for reference.

`fetch.py` reads the site's JSON API (`/api/index`, `/api/data?path=sets/<slug>/...`)
through the existing cached `BuzzpointsScraper.fetch`, so reruns don't hit the network.
Only sets that are `public` and at level `college` or `open` are included.

## What counts as a mention

- **Lexicon:** `lexicon.py`, 22 groups (primates, dogs, birds, …), each with
  common names, species, and class words ("primate", "reptile"). For example, the
  monkey group includes monkey, ape, simian, chimp, gorilla, macaque, lemur, singerie, Hanuman, Sun Wukong.
- **Names, titles and idioms count.** Stephen Crane, Swan Lake, Jim Crow, Wolf Hall and
  "busy beaver" are all mentions. The question is how often the words show up, not
  how often a question is about animals.
- **Mythical creatures count** toward the real animal they are based on (Pegasus →
  horses, Cerberus → dogs, griffin → birds) and are flagged `mythical`. Dragons and
  Leviathan are left out (not based on one animal). Animal-headed gods (Anubis,
  Ganesha) are not counted.
- **Excluded words**, whose usual sense in quizbowl text isn't an animal: mole
  (chemistry unit), seal, cardinal, monarch, cricket, buck, kid, sloth, bass, ray,
  Dalmatian, kite, rook, and more. The full list and reasons are in `lexicon.EXCLUDED`.
- **Precision rules** in `count.py`:
  - pronunciation guides like (“bee-OH-shuh”) are stripped first
  - all-caps acronyms are skipped (CAT theory, DOG1)
  - "turkey", "ram" and "worm" must be lowercase (not Turkey, RAM, the Diet of Worms)
  - stop phrases are discarded (papal bull, bat mitzvah)
  - verb-prone words (bear, fly, duck, swallow, hawk, …) must be tagged as nouns by
    spaCy in context, which drops "bear no grudge" and "the flag flies"
- **Both question text and the primary answer** are searched (alternates in
  `[...]` are not). Bonus `partConv` duplicates and repeated packets are dropped.

Hand audit: in a random sample of 150 mentions, 147 were correct under these rules.
The 3 errors (two acronyms, plus "big cats" filed under domestic cats) were fixed. A
missed verb ("six bear the name") shows the kind of error that remains, at well
under 1% of mentions.

## Metrics

- **Rate:** mentions per 10,000 words (question text + primary answers), so
  longer sets aren't favoured.
- **% of questions** with at least one mention.
- **Ratio** of the focus set's rate to all other sets pooled, and the focus set's
  **rank** among all sets for each group.

There are no confidence intervals. The counts are exact for these sets as written,
and the reports describe what was read, not a lasting tendency of the writers. The
uncertainty that remains is in the counting rules (lexicon choices, a missed verb
here and there). The "without top question" and raw-count robustness checks speak
to that better than sampling intervals would.

## Caveats

- Category mix isn't controlled for. A set heavier on biology or mythology will
  mention more animals for reasons that have nothing to do with the writers.
- 2024 ACF Nationals has only 2 packets on buzzpoints, so its numbers swing on single questions.
- Sets were de-duplicated on exact text. Slightly edited reprints of the same
  question within a set (~16 cases) are both kept.
