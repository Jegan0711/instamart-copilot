### Input guardrails and redaction

| item | value |
|---|---|
| empty tickets (skipped_empty, no model call) | 4 |
| tickets above MAX_INPUT_TOKENS=400 (flagged_long, no model call) | 4 (each estimated at 734 tokens) |
| emails redacted | 55 |
| phone numbers redacted | 91 |
| order ids redacted | 132 |
| total substitutions | 278 |
| PII matches left after re-applying all three patterns | 0 |

System prompt sizes: zero-shot 45 estimated tokens, few-shot 140 estimated tokens.

### Resilient calls (zero-shot full run)

TransientLLMErrors raised: 19; all recovered by a retry (no permanent errors). Wait range observed: 0.0453-0.0562 s (allowed for retry 0: 0.045-0.0575 s).

First retry log lines:

```
RETRY ticket=T0002 retry_number=0 wait=0.0530s reason=503 server busy
RETRY ticket=T0116 retry_number=0 wait=0.0453s reason=503 server busy
RETRY ticket=T0153 retry_number=0 wait=0.0484s reason=503 server busy
```

### Status counts per full run

| status | zero-shot | few-shot |
|---|---|---|
| classified_json | 135 | 135 |
| classified_keyword | 12 | 12 |
| flagged_invalid_output | 9 | 9 |
| skipped_empty | 4 | 4 |
| flagged_long | 4 | 4 |

### Zero-shot vs few-shot

| prompt | tickets classified | accuracy | total tokens | tokens per classified ticket |
|---|---|---|---|---|
| zero-shot | 147 | 126/147 = 85.7% | 11428 | 77.7 |
| few-shot | 147 | 147/147 = 100.0% | 26248 | 178.6 |

**Zero-shot confusion table (rows = true category, columns = predicted)**

| true_category | billing | delivery | technical | fraud |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

### Token budget demo (zero-shot, TOKEN_BUDGET=6000)

74 tickets ended **budget_exhausted** and were routed to human review; tokens spent before stopping: 6021.

### Final routing (few-shot, TOKEN_BUDGET=100000) - outputs/triage_results.csv

| route | priority | tickets |
|---|---|---|
| human_review | NORMAL | 17 |
| premium_desk | HIGH | 11 |
| security_team | HIGH | 34 |
| support_queue | NORMAL | 102 |

Rule-validated (classified_json or classified_keyword): 147 tickets. Sent to human review: 17 tickets (flagged_invalid_output: 9, skipped_empty: 4, flagged_long: 4).
