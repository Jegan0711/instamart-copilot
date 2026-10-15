"""Assemble README.md from the narrative below and the generated tables in reports/*.md (so every number is copied by code)."""


def frag(name):
    return open(f"reports/{name}.md", encoding="utf-8").read().strip()


README = r"""# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected parts, built on **synthetic, fictional data** (brand name used only for the scenario). All amounts are in INR. No paid service, API key or network call is used: every language-model call goes through the supplied offline `mock_llm.py`.

| Part | What it does | Main files |
|---|---|---|
| A | Audit and clean the raw exports, store them in SQLite, answer 7 SQL questions, pandas cross-check | `part_a_data.py`, `sql/queries.sql`, `validate.py` |
| B | RFM + K-means customer segments, PCA, leakage-free lapse model, retention list | `part_b_segments_lapse.py` |
| C | Support-ticket triage with redaction, retries, output validation, token budget and routing | `part_c_triage.py` |
| D | Policy RAG assistant (TF-IDF) and a tool-using refund agent with safe stops and crash-resume | `part_d_rag_agent.py` |

`generate_data.py` and `mock_llm.py` are the supplied files, committed **unmodified**. `mdutil.py` only writes Markdown tables; `build_readme.py` assembles this README from the generated tables in `reports/` so the numbers below are copied by code, not typed. Fixed seeds are used everywhere (`random_state=42`, `random.seed(42)`). Python 3.12, pandas, numpy, scikit-learn, joblib (see `requirements.txt`).

## Reproduce

Run from an empty folder (Windows: activate with `.venv\Scripts\activate`):

1. `git clone <repository-url> instamart-copilot && cd instamart-copilot` (branch `main`)
2. `python -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
6. `python part_a_data.py` (clean CSVs, `data/instamart.db`, SQL results)
7. `python validate.py` (prints `ALL CHECKPOINTS PASSED`)
8. `python part_b_segments_lapse.py`
9. `python part_c_triage.py`
10. `python part_d_rag_agent.py`

Optional: `python build_readme.py` regenerates this README from the generated reports. Parts must be run in this order because each consumes the previous outputs.

## Part A

Raw exports are audited, cleaned with explicit rules, stored in `data/clean/*.csv` and `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`), and queried with the seven queries in `sql/queries.sql`. `validate.py` asserts 24 reference checkpoints (all of tasks 3 and 4 plus database checks) and prints `ALL CHECKPOINTS PASSED`.

Cleaning rules, in order. Orders: (1) convert `order_ts`; (2) drop exact duplicates; (3) per `order_id` keep the record with the latest `order_ts`; (4) coerce `order_amount` to numeric and refill unparseable values with `round(price_inr x quantity x (1 - discount_pct/100), 2)`; (5) set negative `delivery_minutes` to missing, fill completed orders with the median of their product category among completed orders (`groupby` + `transform`), and leave cancelled orders missing because a cancelled order was never delivered. Customers: strip and title-case city, map gender to male/female/unknown, treat ages outside 10-100 as missing and fill with the city median (`groupby` + `transform`), convert `signup_date`. Rows changed per rule are in the table below (a few raw bad values, e.g. 106 negative times versus 105 changed, sat in duplicate or pending rows that rules 2-3 had already removed).

@@A@@

**Why ties cause RANK() != ROW_NUMBER():** when two customers in a city have exactly the same total spend, RANK() gives them the same rank (and skips the next number) while ROW_NUMBER() still numbers them 1, 2, 3 in sequence, so the two columns differ from the second tied customer onward.

**Exploratory answers (about 150 words).** *What is the data about?* 11,110 completed orders over six months (April-September 2025) in six cities, one row per order with product, quantity, discount, amount and delivery time. *Distributions:* amounts are strongly right-skewed (mean 269.91 versus median 135.00, skewness 4.61, Q1 64.80, Q3 297.00); most baskets are small grocery items. *Outliers:* 897 orders lie above the upper fence of 645.30 and none below the lower fence; they are real high-priced electronics baskets, not errors, so they were kept. *Relationships:* every correlation is weak; the largest in absolute value is order_amount with discount_pct at -0.095, just ahead of order_amount with quantity at +0.087. Correlation of this size does not establish causation: a discount is not shown to lower the amount, it may simply reflect which products are discounted.

## Part B

Features use only orders before 2025-09-01 plus the clean customer table. **No encoder, scaler or resampler saw test rows** (scaling and one-hot encoding sit inside a scikit-learn `Pipeline`, so grid-search folds refit them on training folds only; over-sampling is applied to the training rows after the split). **No data from 2025-09-01 onwards entered the features** (asserted in code), and **the segment labels from the clustering were not used as features**.

@@B@@

**Choosing K.** The elbow is clear at K=4: inertia falls from 1,268.2 (K=3) to 743.1 (K=4) but only to 617.4 at K=5, after which the curve flattens. K=4 also has the best silhouette of all K (0.526), and it gives four segments the business can act on, so K=4 was chosen from {3, 4, 5}. **Why K=8 has lower inertia than K=2 yet that cannot pick K:** inertia is the total squared distance of points to their cluster centre, so adding clusters can only lower it (with one cluster per customer it would reach zero); it measures fit, not usefulness, so it must be combined with the elbow, the silhouette and interpretability.

**Segments.** Cluster numbers are only labels with no ordering. The names and actions above come from the profile; the segment with the highest mean monetary value (138 customers, about INR 7,130, only 6.8 orders) is flagged `is_premium`. The 419-customer segment orders about 19.4 times with mean recency 8.8 days (the habit to protect); the 182-customer segment has not ordered for about 109 days on average (win-back).

**Model choice.** The random forest was selected on its higher cross-validated F1 (0.681 versus 0.651), before looking at the test set, and it also leads on the test set (F1 0.655, precision 0.659, recall 0.652 versus 0.648 / 0.644 / 0.652 for logistic regression). Missing a customer who is about to lapse loses all of their future orders, whereas wrongly targeting a loyal customer costs one coupon or message, so recall matters more than precision here, and the forest is at least as good on recall and slightly better on precision. Both models clear the required test F1 of 0.55 and beat the baseline, which has accuracy 61.6% but F1 of 0. Best parameters differ from the brief's reference (logistic regression C=100 instead of 1) because extra features were added.

**Class balance.** Over-sampling the lapsed class moved **recall** most (0.652 to 0.854, +0.20) while precision fell from 0.644 to 0.589 (F1 0.648 to 0.697). For the retention team this means the balanced model finds about 85% of the customers who will lapse but roughly 41% of those contacted will be loyal customers, so it suits cheap campaigns (coupons, notifications) and not expensive ones.

**Reading the model.** Random forest: frequency (0.280) and monetary (0.271) dominate, then recency (0.115), average delivery time (0.061) and cancellation rate (0.052), so how much and how often someone has ordered predicts lapse far better than city or gender. Logistic regression signs: *frequency* is negative (-1.698), customers who ordered often are far less likely to lapse; *recency* is positive (+0.04, tiny), a longer gap since the last order slightly raises lapse risk; *monetary* is positive (+0.285) once frequency is held fixed, so big spend over few orders is a weaker loyalty signal than many orders (frequency and monetary are correlated, so individual coefficients should be read with care).

**Retention list is illustrative.** The Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label and `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation. (The model's own features still use only pre-September orders.)

## Part C

Order of the pipeline: empty/long guardrails, redaction, prompt, retry-wrapped call, output validation, token budget, routing from `outputs/customer_segments.csv`. The user message sent to the model is exactly the redacted ticket text, and an assertion confirms no email, phone number or order id appears in any prompt.

@@C@@

**When the few-shot tokens pay off.** The few-shot prompt costs 178.6 tokens per classified ticket against 77.7 for zero-shot (26,248 versus 11,428 total tokens, 2.3 times as much), but it lifts accuracy from 85.7% to 100%, fixing the 21 tickets zero-shot got wrong (11 fraud and 10 technical tickets labelled billing). That is worth paying for when a wrong label is costly, as with fraud tickets, which the escalation policy requires to reach the security team within 2 hours; it is not worth paying for billing and delivery tickets, which zero-shot already got right. With a hard token budget the longer prompt also exhausts the budget sooner, so the choice is a budget decision, not only an accuracy one.

**Which outputs were rule-validated and which went to humans.** 147 tickets (135 valid JSON + 12 keyword-extracted replies) passed the output rules and were routed automatically (fraud to the security team with HIGH priority, premium customers to the premium desk with HIGH priority, the rest to the support queue). 17 tickets went to human review: 9 invalid model outputs (never guessed), 4 empty tickets and 4 over-long tickets. In the budget demo, tickets that hit the budget were also routed to human review.

## Part D

The refund agent reads `data/instamart.db` (Part A) and reuses the retrieval from the RAG assistant. Refund rules implemented in the tools: only completed orders; delivery above 45 minutes means 100% of `order_amount`; refunds above INR 1000 are escalated, not issued.

@@D@@

**TF and IDF.** Term frequency measures how often a word appears inside a chunk (how much that chunk is about it), and inverse document frequency measures how rare the word is across all chunks, so words that occur in many chunks (such as "refund" here) count for less than distinctive ones. A limitation is that TF-IDF only matches words, not meaning: a question phrased "money back" shares no word with a chunk that says "refund".

**Zero overlap.** Without overlap two answerable questions fail: *"What delivery time makes a completed order eligible for a full refund?"* (the key sentence "delivered in more than 45 minutes, the customer receives a full refund" is cut by a chunk boundary, so the retrieved chunk holds only "receives a full refund of the order amount") and *"How soon is money returned when a payment fails but is deducted?"* (the sentence is cut before "within 48 hours", so the answer is a truncated fragment). Overlap keeps boundary sentences whole in at least one chunk. The ledger `outputs/refund_ledger.jsonl` holds the three actions of the main run (two issued refunds and one escalation); the crash-and-resume demo writes to a separate `outputs/refund_ledger_demo.jsonl` so S1 is not refunded twice in the real ledger. Demonstration runs are in `outputs/agent_demo_runs.jsonl`; checkpoints are in `state/`.

**ReAct versus plan-and-execute (measured ReAct numbers; plan-and-execute counts are estimates).** My ReAct agent used 5 model calls for a refund or escalation (S1: 1,067 tokens), 4 for an on-time denial (S4: 741), 3 for a cancelled order (S5: 439) and 2 for NOT_FOUND (S6: 140). A plan-and-execute design would ask the model once for the whole step list and once to summarise, so about 2 calls on the happy path, and fewer tokens because the growing message history (S7 reached 3,204 tokens in six calls) is not re-sent every step. But it plans before it has seen the order: a cancelled or missing order only shows up after get_order, so it must replan (an extra call) or run steps that were never needed, and four of the seven scenarios branch this way (cancelled, not found, on time, above INR 1000). ReAct simply stops at the observation, for example after 2 calls for NOT_FOUND. For this refund workflow, which moves money and branches on database facts, I would choose ReAct with the step, token and time limits and checkpoints; the extra cost is small (at most 1,067 tokens per request) and each decision uses the latest observation. I would reconsider plan-and-execute only at very high volume.

## Limitations and Next Steps

- **Premium flag is post-hoc.** It uses whole-window RFM that includes September, so the retention list is illustrative and not a leakage-free evaluation; a clean version would compute segments from pre-cutoff orders only.
- **The mock model is rule-based.** The few-shot gain (85.7% to 100%) comes from the mock reading examples, so it shows the pipeline and cost trade-off, not real model quality. All data is synthetic.
- **Lapse model.** One cutoff and one 20% test split (232 customers) give noisy metrics; K was restricted to {3, 4, 5}; the balanced model trades precision for recall and is not the selected model. Next: time-based validation and cost-weighted thresholds.
- **Retrieval.** TF-IDF matches words, not meaning; an embedding model would handle paraphrases, but is outside the allowed libraries.
- **Agent.** A crash between a tool executing and its checkpoint being saved could repeat that tool on resume; real refunds need idempotency keys. Only one tool call per model reply is supported.
- **Storage.** SQLite is single-writer and fine for this project, not for production traffic.
"""

for key, name in (("@@A@@", "part_a"), ("@@B@@", "part_b"), ("@@C@@", "part_c"), ("@@D@@", "part_d")):
    README = README.replace(key, frag(name))
open("README.md", "w", encoding="utf-8").write(README)
print("README.md written,", len(README.split()), "words")
