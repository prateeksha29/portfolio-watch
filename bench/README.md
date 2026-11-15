# Model benchmark

A fixed, labeled test set taken from our own news, for choosing models and tuning prompts. Items are copied out of
the database, so the 14-day retention never deletes them. Labels were written by Claude (2026-11-01) from the
source text only, before any candidate output was seen; spot-check them before relying on close results.

## Steps

```
python -m fin_news.bench sample [dedup topic kind stories filings search ask]   # freeze items (done once)
python -m fin_news.bench embed --pool        # embed with every candidate, write search results to judge
python -m fin_news.bench embed               # score embedding candidates
python -m fin_news.bench llm stories|filings|ask [--models m1 m2] [--prompt name] [--split dev|test]
python -m fin_news.bench grade stories|filings|ask   # blind packets for grading free text
python -m fin_news.bench report              # reports/bench-<date>.md
python -m fin_news.bench tech                # bench/tech.json: every number in the Benchmarks page's Technical view,
                                              #   with 95% bootstrap intervals (re-run after new runs or grades)
```

LLM runs use the shared GPU through llama-swap: run them away from the 6am / 2pm / 6pm pipeline runs.
Embedding candidates other than the production model run on the CPU (`st:` backends): install sentence-transformers
(and torch) in `.venv` first; they were removed after the 2026-11 benchmark. Cached vectors were removed too;
`bench/runs/embed/results.json` and `reports/bench-*.md` keep the scores. A winner must be added to llama-swap before
the app can use it.

## Files

Only `prompts/` and `tech.json` are in the repository: the items, labels, outputs and grades contain news
article text and the owner's own questions, so they stay on the machine (`.gitignore`).


| Path | What |
|---|---|
| `data/<task>.jsonl` | frozen items; `<task>_current.jsonl` = the pipeline's existing output, kept apart |
| `data/search.jsonl`, `data/ask_questions.jsonl` | hand-written queries and questions |
| `labels/<task>.jsonl` | reference labels, keyed by item id |
| `runs/<task>/<model>[__<prompt>].jsonl` | model outputs with timing |
| `runs/embed/` | cached vectors and embedding scores |
| `grading/` | blind packets (`*_packet.jsonl`), letter-to-run keys (`*_key.json`) and grades |
| `prompts/<task>/<name>.txt` | prompt variants for tuning |
| `tech.json` | numbers for the Benchmarks page's Technical view (`bench tech`); served-model vectors cached in `runs/embed/llama_<dim>/` |

Splits: a stable third of every task is `test` (by item id hash). Tune on `dev`; check the final choice once on `test`.

## Labeling rules

- **dedup** (`same`): the same specific news event. Rewordings and different outlets' reports of one event count;
  so do pieces about the same quarterly report (results, call transcript, slides) and market wraps of the same
  trading session. Generic previews and opinion pieces on the same company do not, nor do templated headlines about
  different companies. Option-chain quote pages are marked `exclude` (junk the filter should drop).
- **topic** (`topics`): every watched theme the article is mainly about, at the level of the industry or market.
  An article about one company counts only when that company's news is itself a theme event (e.g. a TSMC packaging
  price rise for "semiconductor industry"). Excluded-country macro, crypto and commodities count as none.
- **kind**: news (reports a new fact, incl. analyst actions and market wraps), analysis (opinion, preview, forecast,
  explainer, listicle), press_release (issued by the company or a law firm / report vendor), transcript (call
  transcript, call summary or highlights, slides). Quote and chart pages are `exclude`.
- **stories**: `on_topic` per the app's rule (mainly about the named holding/theme); `ambiguous_topic` items are
  left out of that score. `sentiment` = effect on a shareholder of the named holding; `accept` lists other
  defensible answers. `facts` = 2-5 points a good summary should contain.
- **filings**: same, without on_topic. `f000` is an empty filing: any specific figure is invented.
- **search**: graded relevance of pooled results: 2 = answers the query, 1 = related, 0 = not.
- **ask**: per question, `must` points a good answer covers and `must_not` mistakes.

## Grading free text

`grade` writes each item's reference with every run's answer under a random letter. Grades go in
`grading/<task>_grades.jsonl`, one line per answer:
`{"id": "s003", "letter": "B", "hit": [0, 2], "wrong": 0, "violations": [], "note": ""}`
(`hit` = indices of facts / must points covered, `wrong` = statements the source contradicts or doesn't support,
`violations` = indices of must_not points).
