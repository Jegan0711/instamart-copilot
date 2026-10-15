### Chunks (40 words, overlap 10)

| document | chunks |
|---|---|
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| refund_policy.txt | 4 |
| TOTAL | 15 |

**retrieve('refund policy late delivery')** top 3 sources: refund_policy.txt (0.2719), refund_policy.txt (0.2514), refund_policy.txt (0.2454)

### Retrieval and grounding evaluation (policy_eval.csv)

| chunking | expected_doc in top-3 | answer contains expected_fact | fallback on unanswerable |
|---|---|---|---|
| overlap 10 | 12/12 | 12/12 | 2/2 |
| overlap 0 | 12/12 | 10/12 | 2/2 |

**Questions that break with overlap 0**

| question | expected fact | answer given | doc retrieved |
|---|---|---|---|
| What delivery time makes a completed order eligible for a full refund? | more than 45 minutes | receives a full refund of the order amount. | True |
| How soon is money returned when a payment fails but is deducted? | 48 hours | If a payment fails but money is deducted, the amount is returned automatically
 
1000 must be approved by a human support lead before money is returned. | True |

### The seven scenarios (outputs/agent_runs.jsonl)

| scenario | status | refund INR | model calls | tokens used | expected |
|---|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1067 | REFUND |
| S2 | REFUND | 135.00 | 5 | 1067 | REFUND |
| S3 | ESCALATE | 1999.00 | 5 | 1054 | ESCALATE |
| S4 | DENIED | 0.00 | 4 | 741 | DENIED |
| S5 | DENIED | 0.00 | 3 | 439 | DENIED |
| S6 | NOT_FOUND | 0.00 | 2 | 140 | NOT_FOUND |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3204 | FALLBACK |

### Stop-safely demonstrations (S1)

| demo | status | model calls | tokens used | tools executed |
|---|---|---|---|---|
| token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 | 456 | 2 |
| time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 | 0 | 0 |

Token-limit hand-off text: HANDOFF TO HUMAN AGENT (FALLBACK_TOKEN_LIMIT): stopped after 3 model call(s), 456 tokens; tools run so far: get_order, search_policy. Please review the request manually.

### Crash and resume (S1)

Simulated crash: `simulated crash after tool #2`. Tools executed before the crash: get_order, search_policy. After `resume_agent('S1-crash-demo')`: status **REFUND**, refund INR 128.25, 5 model calls in total.

Tools executed across the crash and the resume, in order: get_order -> search_policy -> calculate_refund -> issue_refund (4 executions, none repeated).
