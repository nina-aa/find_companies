# Evaluation results

_2026-08-31 10:46 · 7 queries · 7/7 fully green_

| Query | Verdict | Retrieved | Returned | Revised | LLM calls | Cost $ | Latency ms |
|---|---|--:|--:|:--:|--:|--:|--:|
| Q1 | ✅ PASS | 100 | 10 | — | 2 | 0.00158 | 11187 |
| Q2 | ✅ PASS | 100 | 10 | — | 2 | 0.00159 | 9515 |
| Q3 | ✅ PASS | 100 | 10 | — | 2 | 0.00158 | 11922 |
| Q4 | ✅ PASS | 100 | 10 | — | 2 | 0.00138 | 7952 |
| Q5 | ✅ PASS | 0 | 0 | — | 1 | 0.00059 | 3204 |
| S1 | ✅ PASS | 22 | 10 | — | 2 | 0.00138 | 8312 |
| S3 | ✅ PASS | 100 | 10 | — | 1 | 0.00058 | 2672 |

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

**Judgment (manual):** 5 of 304 filter-matches meet every preference, and all 5 now reach the top of the list (the retrieve step runs a preference-boosted second search so they survive the bm25 cut). Ranks 6-10 are pref 1/2, score 0.82, with the missed preference named.

### Q2
- ✅ parsed: mandatory location — want {'Nordic'} got mandate={'Nordic'} plan={'Sweden', 'Norway', 'Finland'}
- ✅ parsed: mandatory industry — want {'Energy'} got mandate={'Energy'} plan={'Energy'}
- ✅ parsed: capabilities overlap — want {'energy forecasting'} got {'energy forecasting'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** The recall fix at work: 25 of 665 meet the 50-250 employee band, and all 10 returned are from that set (pref 2/2, score 1.00). The summary reports 25 and flags the returned 10 as an arbitrary slice of them.

### Q3
- ✅ parsed: mandatory location — want ∅ got mandate=∅ plan=∅
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate={'Fintech'} plan={'Fintech'}
- ✅ parsed: capabilities overlap — want {'fraud detection'} got {'fraud detection'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

**Judgment (manual):** No mandatory country filter (correct — the mandate never says where the company is). Europe is a location preference, so European fraud-detection Fintechs rank first. Every row is mand 3/4: industry + capability met, 'serves European banks' unverifiable and labelled an inference + caveat. Exclusion matches nothing (correctly).

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

**Judgment (manual):** 219 German Biotech drug-discovery firms, no revision. 'founded after 2018' is a preference: the preference-boosted retrieval surfaces founded>=2019 firms (pref 1/1, score 1.00) above older ones (pref 0/1, score 0.65).

### Q5
- ✅ parsed: mandatory location — want {'Finland'} got mandate={'Finland'} plan={'Finland'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate={'Fintech'} plan={'Fintech'}
- ✅ all returned companies exist — 0/0
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ correctly empty (abstention) — empty=True reason='no companies satisfy the mandatory criteria: industry in Fin'
- ✅ budget respected — llm=1 iter=0 rev=0

**Judgment (manual):** Correct abstention. employee_count max in the data is 5000, so >5000 is impossible; feasibility short-circuits after the interpret call only (1 LLM call).

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

**Judgment (manual):** Tight real filter (German Biotech gene-editing founded >=2021, 22 rows). No preferences, so all 22 sit in one tier -> the 10 returned are flagged an arbitrary slice. mand 4/4 (location, industry, founded>=2021, gene-editing).

### S3
- ✅ parsed: mandatory location — want {'Germany'} got mandate={'Germany'} plan={'Germany'}
- ✅ parsed: mandatory industry — want {'Energy'} got mandate={'Energy'} plan={'Energy'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=1 iter=1 rev=0

**Judgment (manual):** Non-vacuous exclusion: 421 of 614 German Energy firms remain after removing 'smart grid'. Purely structural + keyword exclusion -> LLM validation skipped (1 LLM call total). mand 3/3 (location, industry, not-smart-grid).

_Ranking quality, exclusion correctness and evidence *sufficiency* are judgment calls — the **Judgment (manual)** notes above, not auto-scored._
