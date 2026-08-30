# Evaluation results

_2026-08-30 10:41 · 7 queries · 7/7 fully green_

| Query | Verdict | Retrieved | Returned | Revised | LLM calls | Cost $ | Latency ms |
|---|---|--:|--:|:--:|--:|--:|--:|
| Q1 | ✅ PASS | 100 | 10 | — | 2 | 0.00176 | 15077 |
| Q2 | ✅ PASS | 100 | 10 | — | 2 | 0.00172 | 12344 |
| Q3 | ✅ PASS | 100 | 10 | — | 2 | 0.00151 | 10577 |
| Q4 | ✅ PASS | 100 | 10 | — | 2 | 0.00141 | 10265 |
| Q5 | ✅ PASS | 0 | 0 | — | 1 | 0.00038 | 2094 |
| S1 | ✅ PASS | 22 | 10 | — | 2 | 0.00127 | 7327 |
| S3 | ✅ PASS | 100 | 10 | — | 2 | 0.00103 | 5109 |

## Per-check detail

### Q1
- ✅ parsed: mandatory location — want {'Finland'} got mandate={'Finland'} plan={'Finland'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate={'Fintech'} plan={'Fintech'}
- ✅ parsed: capabilities overlap — want {'fraud detection', 'banking analytics'} got {'fraud detection', 'banking analytics'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

### Q2
- ✅ parsed: mandatory location — want {'Nordic'} got mandate={'Nordic'} plan={'Sweden', 'Finland', 'Norway'}
- ✅ parsed: mandatory industry — want {'Energy'} got mandate={'Energy'} plan={'Energy'}
- ✅ parsed: capabilities overlap — want {'energy forecasting'} got {'energy forecasting'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

### Q3
- ✅ parsed: mandatory location — want {'Europe'} got mandate={'Europe'} plan={'France', 'Finland', 'Norway', 'UK', 'Germany', 'Sweden', 'Netherlands'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate=∅ plan={'Fintech'}
- ✅ parsed: capabilities overlap — want {'fraud detection'} got {'fraud detection technology'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

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

### Q5
- ✅ parsed: mandatory location — want {'Finland'} got mandate={'Finland'} plan={'Finland'}
- ✅ parsed: mandatory industry — want {'Fintech'} got mandate={'Fintech'} plan={'Fintech'}
- ✅ all returned companies exist — 0/0
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ correctly empty (abstention) — empty=True reason='no companies satisfy the mandatory criteria: industry in Fin'
- ✅ budget respected — llm=1 iter=0 rev=0

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

### S3
- ✅ parsed: mandatory location — want {'Germany'} got mandate={'Germany'} plan={'Germany'}
- ✅ parsed: mandatory industry — want {'Energy'} got mandate={'Energy'} plan={'Energy'}
- ✅ all returned companies exist — 10/10
- ✅ returned rows satisfy mandatory filters — violations: []
- ✅ evidence quotes grounded — ungrounded: 0
- ✅ revision performed == expected — want False got False
- ✅ returned at least one result
- ✅ budget respected — llm=2 iter=1 rev=0

_Ranking quality, exclusion correctness and evidence *sufficiency* are judgment columns — reviewed by hand, not scored here._
