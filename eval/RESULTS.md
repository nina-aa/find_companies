# Evaluation results

_2026-08-30 11:34 · 7 queries · 7/7 fully green_

| Query | Verdict | Retrieved | Returned | Revised | LLM calls | Cost $ | Latency ms |
|---|---|--:|--:|:--:|--:|--:|--:|
| Q1 | ✅ PASS | 100 | 10 | — | 2 | 0.00174 | 16688 |
| Q2 | ✅ PASS | 100 | 10 | — | 2 | 0.00171 | 13625 |
| Q3 | ✅ PASS | 100 | 10 | — | 2 | 0.00151 | 10233 |
| Q4 | ✅ PASS | 100 | 10 | — | 2 | 0.00140 | 8907 |
| Q5 | ✅ PASS | 0 | 0 | — | 1 | 0.00038 | 1920 |
| S1 | ✅ PASS | 22 | 10 | — | 2 | 0.00127 | 7547 |
| S3 | ✅ PASS | 100 | 10 | — | 2 | 0.00109 | 6407 |

## Per-query detail

### Q1
- ✅ parsed: mandatory location — want {'Finland'} got mandate={'Finland'} plan={'Finland'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate={'Fintech'} plan={'Fintech'}
- ✅ parsed: capabilities overlap — want {'banking analytics', 'fraud detection'} got {'banking analytics', 'fraud detection'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** Ranking defensible: the only company matching BOTH topics (240 emp, 2018) ranks #1; firms failing the <250 / >2015 preferences drop below via preference_score. One clean verbatim span per topic. No revision, as expected.

### Q2
- ✅ parsed: mandatory location — want {'Nordic'} got mandate={'Nordic'} plan={'Finland', 'Sweden', 'Norway'}
- ✅ parsed: mandatory industry — want {'Energy'} got mandate={'Energy'} plan={'Energy'}
- ✅ parsed: capabilities overlap — want {'energy forecasting'} got {'energy forecasting'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** All rows are Nordic Energy with an 'energy forecasting' span. The 50-250 band correctly does not filter. relevance_score is flat (model weakness) so intra-tier order is bm25 + preference_score.

### Q3
- ✅ parsed: mandatory location — want {'Europe'} got mandate={'Europe'} plan={'Germany', 'Sweden', 'Finland', 'Netherlands', 'France', 'UK', 'Norway'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate=∅ plan={'Fintech'}
- ✅ parsed: capabilities overlap — want {'fraud detection'} got {'fraud detection technology'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** Exclusion correctly matches nothing and is reported as such. Every candidate is European Fintech with a fraud-detection span, but 'serves European banks' is not in the one- sentence descriptions, so all come back 'partial' with that gap labelled inference - the correct ceiling given the source.

### Q4
- ✅ parsed: mandatory location — want {'Germany'} got mandate={'Germany'} plan={'Germany'}
- ✅ parsed: mandatory industry — want {'Biotech'} got mandate={'Biotech'} plan={'Biotech'}
- ✅ parsed: capabilities overlap — want {'drug discovery'} got {'drug discovery'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** 53 German Biotech drug-discovery firms founded >2018 exist, so no revision (correct). After the deterministic pool re-rank the returned list is all founded >=2019 - the founding-year preference is respected without ever filtering.

### Q5
- ✅ parsed: mandatory location — want {'Finland'} got mandate={'Finland'} plan={'Finland'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate={'Fintech'} plan={'Fintech'}
- ✅ all returned companies exist — 0/0
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ correctly empty (abstention) — empty=True reason='no companies satisfy the mandatory criteria: industry in Fin'
- ✅ budget respected — llm=1 iter=0 rev=0

**Judgment (manual):** Correct abstention: employee_count max is 5000. Feasibility short-circuits after the interpret call only (1 LLM call, no validation spend) and returns an empty list with the failing criteria named.

### S1
- ✅ parsed: mandatory location — want {'Germany'} got mandate={'Germany'} plan={'Germany'}
- ✅ parsed: mandatory industry — want {'Biotech'} got mandate={'Biotech'} plan={'Biotech'}
- ✅ parsed: capabilities overlap — want {'gene editing'} got {'gene editing'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** Tight real-data filter (German Biotech gene-editing founded >=2021, ~22 rows). Returns a non-empty ranked list without padding beyond the real pool; spans are clean gene-editing quotes.

### S3
- ✅ parsed: mandatory location — want {'Germany'} got mandate={'Germany'} plan={'Germany'}
- ✅ parsed: mandatory industry — want {'Energy'} got mandate={'Energy'} plan={'Energy'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** Non-vacuous exclusion: ~193 of 614 German Energy firms mention 'smart grid' and are removed, and the response reports the count. Remaining candidates are German Energy with no smart-grid in the text.

_Ranking quality, exclusion correctness and evidence *sufficiency* are judgment calls — the **Judgment (manual)** notes above, not auto-scored._
