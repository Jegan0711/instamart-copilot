# Instamart Customer Intelligence & Support Copilot

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

### Audit of the raw tables

**customers.csv** - shape 1200 rows x 5 columns

| column | dtype | missing | missing_pct |
|---|---|---|---|
| customer_id | str | 0 | 0 |
| city | str | 0 | 0 |
| signup_date | str | 0 | 0 |
| age | float64 | 61 | 5.08 |
| gender | str | 30 | 2.5 |

**products.csv** - shape 40 rows x 4 columns

| column | dtype | missing | missing_pct |
|---|---|---|---|
| product_id | str | 0 | 0 |
| product_name | str | 0 | 0 |
| category | str | 0 | 0 |
| price_inr | int64 | 0 | 0 |

**orders.csv** - shape 13084 rows x 9 columns

| column | dtype | missing | missing_pct |
|---|---|---|---|
| order_id | str | 0 | 0 |
| customer_id | str | 0 | 0 |
| product_id | str | 0 | 0 |
| quantity | int64 | 0 | 0 |
| order_ts | str | 0 | 0 |
| status | str | 0 | 0 |
| discount_pct | int64 | 0 | 0 |
| order_amount | str | 0 | 0 |
| delivery_minutes | float64 | 2541 | 19.42 |

**Audit figures**

| check | value |
|---|---|
| raw orders.csv rows | 13084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id values | 12309 |
| non-numeric order_amount entries | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules and rows/values changed

| table | rule | rows/values changed | note |
|---|---|---|---|
| customers | city: strip spaces + title case | 476 | 6 cities remain |
| customers | gender mapped to male/female/unknown | 950 | blank -> unknown (30 rows) |
| customers | age outside 10-100 set to missing | 6 |  |
| customers | missing age filled with city median (groupby+transform) | 67 | 0 missing left |
| customers | signup_date converted to datetime | 1200 |  |
| orders | 1. order_ts converted to datetime | 13084 |  |
| orders | 2. exact duplicate rows removed | 119 | 12965 rows left |
| orders | 3. keep latest order_ts per order_id (drops earlier pending records) | 656 | 12309 rows left |
| orders | 4. unparseable order_amount refilled from price x qty x (1-disc) | 112 | 0 missing left |
| orders | 5a. negative delivery_minutes set to missing | 105 |  |
| orders | 5b. completed: missing delivery_minutes filled with category median | 774 |  |
| orders | 5c. cancelled: delivery_minutes left missing (never delivered) | 1199 | not imputed |

**Customers after cleaning**

| city | customers |
|---|---|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| gender | customers |
|---|---|
| female | 610 |
| male | 560 |
| unknown | 30 |

Missing ages: 0; age range 18-60.

**Orders after cleaning**

| check | value |
|---|---|
| rows | 12309 |
| completed | 11110 |
| cancelled | 1199 |
| pending | 0 |
| missing order_amount | 0 |
| missing delivery_minutes (completed) | 0 |
| missing delivery_minutes (cancelled) | 1199 |
| sum of order_amount, completed (INR) | 2,998,668.80 |

### SQL results (sql/queries.sql, run with pd.read_sql_query)

| query | what | result |
|---|---|---|
| Q1 | customers with total of completed orders > 500 above 8000 | 30 customers |
| Q2 | revenue by city x category | 30 rows; sum INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 | products priced above the average catalogue price | 11 products |
| Q4 | customers with no completed order | 25 customers |
| Q5 | customers above the average completed spend | 550 customers; average INR 2,552.06 |
| Q6 | rank / dense_rank / row_number per city | 1175 rows; 8 rows with RANK() != ROW_NUMBER() |
| Q7 | daily revenue with running total | 183 daily rows; last cumulative INR 2,998,668.80 |

**Q6 top customer per city**

| city | customer_id | total_spend |
|---|---|---|
| Bengaluru | C0082 | 12,856.2 |
| Chennai | C0930 | 10,151.4 |
| Delhi | C0291 | 13,742.6 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.1 |

### pandas cross-check

pivot_table grand total = INR 2,998,668.80; SQL Q2 total = INR 2,998,668.80; equal: True

| city | Dairy | Electronics | Grocery | Personal Care | Snacks | Total |
|---|---|---|---|---|---|---|
| Bengaluru | 87,075.05 | 233,909.45 | 110,915.65 | 20,193.85 | 71,549.4 | 523,643.4 |
| Chennai | 78,301.8 | 195,508.8 | 93,817.45 | 27,675.4 | 68,325.7 | 463,629.15 |
| Delhi | 117,271.7 | 158,549.1 | 112,658.05 | 23,471.5 | 90,879.45 | 502,829.8 |
| Hyderabad | 83,731 | 228,913.1 | 109,475.4 | 24,800.05 | 81,670.35 | 528,589.9 |
| Mumbai | 75,209.85 | 214,738.4 | 78,970.15 | 38,018 | 72,122.1 | 479,058.5 |
| Pune | 90,349.55 | 187,629.25 | 108,456.6 | 34,275.15 | 80,207.5 | 500,918.05 |
| Total | 531,938.95 | 1,219,248.1 | 614,293.3 | 168,433.95 | 464,754.5 | 2,998,668.8 |

**Per city: sum, mean, count of completed order_amount**

| city | total | mean | orders |
|---|---|---|---|
| Bengaluru | 523,643.4 | 276.47 | 1894 |
| Chennai | 463,629.15 | 272.72 | 1700 |
| Delhi | 502,829.8 | 243.62 | 2064 |
| Hyderabad | 528,589.9 | 284.95 | 1855 |
| Mumbai | 479,058.5 | 297 | 1613 |
| Pune | 500,918.05 | 252.48 | 1984 |

**Distribution of completed order_amount**

| statistic | value |
|---|---|
| mean | 269.91 |
| median | 135.00 |
| skewness | 4.61 |
| Q1 | 64.80 |
| Q3 | 297.00 |
| lower fence | -283.50 |
| upper fence | 645.30 |
| outliers below lower fence | 0 |
| outliers above upper fence | 897 |

**Correlation matrix (completed orders)**

| index | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1 | -0 | -0.01 |
| quantity | 0.087 | -0 | 1 | -0.013 |
| delivery_minutes | -0.003 | -0.01 | -0.013 | 1 |

Largest absolute correlation: order_amount vs discount_pct = -0.095.

**Why ties cause RANK() != ROW_NUMBER():** when two customers in a city have exactly the same total spend, RANK() gives them the same rank (and skips the next number) while ROW_NUMBER() still numbers them 1, 2, 3 in sequence, so the two columns differ from the second tied customer onward.

**Exploratory answers (about 150 words).** *What is the data about?* 11,110 completed orders over six months (April-September 2025) in six cities, one row per order with product, quantity, discount, amount and delivery time. *Distributions:* amounts are strongly right-skewed (mean 269.91 versus median 135.00, skewness 4.61, Q1 64.80, Q3 297.00); most baskets are small grocery items. *Outliers:* 897 orders lie above the upper fence of 645.30 and none below the lower fence; they are real high-priced electronics baskets, not errors, so they were kept. *Relationships:* every correlation is weak; the largest in absolute value is order_amount with discount_pct at -0.095, just ahead of order_amount with quantity at +0.087. Correlation of this size does not establish causation: a discount is not shown to lower the amount, it may simply reflect which products are discounted.

## Part B

Features use only orders before 2025-09-01 plus the clean customer table. **No encoder, scaler or resampler saw test rows** (scaling and one-hot encoding sit inside a scikit-learn `Pipeline`, so grid-search folds refit them on training folds only; over-sampling is applied to the training rows after the split). **No data from 2025-09-01 onwards entered the features** (asserted in code), and **the segment labels from the clustering were not used as features**.

### RFM table (snapshot 2025-10-01)

| statistic | value |
|---|---|
| customers with >= 1 completed order | 1175 |
| recency median / min / max | 19 / 0 / 180 |
| frequency mean / max | 9.46 / 33 |
| monetary mean / max (INR) | 2,552.06 / 14,766.55 |

### K-means for K = 2..8 (standardised RFM, n_init=10, random_state=42)

| K | inertia | silhouette |
|---|---|---|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.48 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

### Segment profile (K=4)

| segment_id | size | mean_recency | mean_frequency | mean_monetary | segment_name | recommended_action | premium |
|---|---|---|---|---|---|---|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Light Buyers | nudge a second order with a small coupon on groceries and dairy | False |
| 1 | 138 | 24.12 | 6.77 | 7,129.7 | Premium High-Value Spenders | personal account manager, priority support and early electronics offers | True |
| 2 | 419 | 8.77 | 19.4 | 3,410.02 | Frequent Essentials Regulars | protect the habit: loyalty points, bundles and a subscription for staples | False |
| 3 | 182 | 109.04 | 2.2 | 685.99 | Dormant At-Risk Buyers | win-back campaign with a time-limited discount before they churn | False |

**PCA (2 components) explained_variance_ratio_**: 0.634 and 0.218, total 0.852

### Lapse problem (cutoff 2025-09-01)

| item | value |
|---|---|
| cohort (completed order before cutoff) | 1156 |
| lapsed | 443 (38.32%) |
| train / test rows | 924 / 232 |
| baseline 'never lapsed' test accuracy | 61.6% |
| baseline lapsed-class F1 | 0.00 |
| features | recency, frequency, monetary, avg_discount, cancel_rate, avg_delivery, age, tenure_days, city, gender, fav_category |
| missing values in X | 0 |

### Preprocessing without leakage

One-hot (drop first) and StandardScaler live inside a scikit-learn Pipeline fitted on training rows only. Scaler mean of frequency (train rows only): **8.115**; full-cohort mean of frequency: **8.029**. They differ because the scaler never sees the 232 test rows, so its mean comes from a different (training-only) subset. No feature column has a missing value (asserted in code).

### Tuning results (cv=5, scoring=f1)

| model | best parameters | best CV F1 |
|---|---|---|
| LogisticRegression | C = 100 | 0.651 |
| RandomForest (100 trees) | max_depth = 8, min_samples_leaf = 10 | 0.681 |

### Test-set evaluation (lapsed = positive class; confusion matrix rows = actual, columns = predicted [not lapsed, lapsed])

| model | confusion matrix | precision | recall | F1 |
|---|---|---|---|---|
| Baseline (never lapsed) | [[143, 0], [89, 0]] | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model (higher best CV F1, chosen before looking at the test set): Random forest**; test F1 = 0.655.

### Class-balance experiment (over-sampling lapsed training rows with replacement, random_state=42)

Training rows after balancing: 570 not lapsed / 570 lapsed.

| model | precision | recall | F1 |
|---|---|---|---|
| Logistic regression, original | 0.644 | 0.652 | 0.648 |
| Logistic regression, balanced training set | 0.589 | 0.854 | 0.697 |

### Top-5 random-forest feature importances

| feature | importance |
|---|---|
| num__frequency | 0.2796 |
| num__monetary | 0.2712 |
| num__recency | 0.1151 |
| num__avg_delivery | 0.0613 |
| num__cancel_rate | 0.0523 |

### Logistic-regression coefficients (standardised numeric features)

| feature | coefficient |
|---|---|
| num__recency | 0.04 |
| num__frequency | -1.698 |
| num__monetary | 0.285 |
| num__avg_discount | 0.004 |
| num__cancel_rate | -0.124 |
| num__avg_delivery | 0.079 |
| num__age | 0.023 |
| num__tenure_days | -0.05 |

**joblib round trip**: reloaded model reproduces the original test predictions exactly: True.

### Retention list

40 test-set customers are in the Premium segment; the top 10 by predicted lapse probability were written to outputs/retention_list.csv. Of these, 1 truly lapsed.

| customer_id | lapse_probability | actually_lapsed |
|---|---|---|
| C0862 | 0.491 | 0 |
| C1112 | 0.4841 | 0 |
| C0060 | 0.4798 | 0 |
| C0802 | 0.4796 | 0 |
| C1008 | 0.4778 | 0 |
| C1000 | 0.4624 | 0 |
| C0525 | 0.4597 | 1 |
| C0133 | 0.4562 | 0 |
| C0267 | 0.4525 | 0 |
| C0167 | 0.4522 | 0 |

**Choosing K.** The elbow is clear at K=4: inertia falls from 1,268.2 (K=3) to 743.1 (K=4) but only to 617.4 at K=5, after which the curve flattens. K=4 also has the best silhouette of all K (0.526), and it gives four segments the business can act on, so K=4 was chosen from {3, 4, 5}. **Why K=8 has lower inertia than K=2 yet that cannot pick K:** inertia is the total squared distance of points to their cluster centre, so adding clusters can only lower it (with one cluster per customer it would reach zero); it measures fit, not usefulness, so it must be combined with the elbow, the silhouette and interpretability.

**Segments.** Cluster numbers are only labels with no ordering. The names and actions above come from the profile; the segment with the highest mean monetary value (138 customers, about INR 7,130, only 6.8 orders) is flagged `is_premium`. The 419-customer segment orders about 19.4 times with mean recency 8.8 days (the habit to protect); the 182-customer segment has not ordered for about 109 days on average (win-back).

**Model choice.** The random forest was selected on its higher cross-validated F1 (0.681 versus 0.651), before looking at the test set, and it also leads on the test set (F1 0.655, precision 0.659, recall 0.652 versus 0.648 / 0.644 / 0.652 for logistic regression). Missing a customer who is about to lapse loses all of their future orders, whereas wrongly targeting a loyal customer costs one coupon or message, so recall matters more than precision here, and the forest is at least as good on recall and slightly better on precision. Both models clear the required test F1 of 0.55 and beat the baseline, which has accuracy 61.6% but F1 of 0. Best parameters differ from the brief's reference (logistic regression C=100 instead of 1) because extra features were added.

**Class balance.** Over-sampling the lapsed class moved **recall** most (0.652 to 0.854, +0.20) while precision fell from 0.644 to 0.589 (F1 0.648 to 0.697). For the retention team this means the balanced model finds about 85% of the customers who will lapse but roughly 41% of those contacted will be loyal customers, so it suits cheap campaigns (coupons, notifications) and not expensive ones.

**Reading the model.** Random forest: frequency (0.280) and monetary (0.271) dominate, then recency (0.115), average delivery time (0.061) and cancellation rate (0.052), so how much and how often someone has ordered predicts lapse far better than city or gender. Logistic regression signs: *frequency* is negative (-1.698), customers who ordered often are far less likely to lapse; *recency* is positive (+0.04, tiny), a longer gap since the last order slightly raises lapse risk; *monetary* is positive (+0.285) once frequency is held fixed, so big spend over few orders is a weaker loyalty signal than many orders (frequency and monetary are correlated, so individual coefficients should be read with care).

**Retention list is illustrative.** The Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label and `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation. (The model's own features still use only pre-September orders.)

## Part C

Order of the pipeline: empty/long guardrails, redaction, prompt, retry-wrapped call, output validation, token budget, routing from `outputs/customer_segments.csv`. The user message sent to the model is exactly the redacted ticket text, and an assertion confirms no email, phone number or order id appears in any prompt.

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

**When the few-shot tokens pay off.** The few-shot prompt costs 178.6 tokens per classified ticket against 77.7 for zero-shot (26,248 versus 11,428 total tokens, 2.3 times as much), but it lifts accuracy from 85.7% to 100%, fixing the 21 tickets zero-shot got wrong (11 fraud and 10 technical tickets labelled billing). That is worth paying for when a wrong label is costly, as with fraud tickets, which the escalation policy requires to reach the security team within 2 hours; it is not worth paying for billing and delivery tickets, which zero-shot already got right. With a hard token budget the longer prompt also exhausts the budget sooner, so the choice is a budget decision, not only an accuracy one.

**Which outputs were rule-validated and which went to humans.** 147 tickets (135 valid JSON + 12 keyword-extracted replies) passed the output rules and were routed automatically (fraud to the security team with HIGH priority, premium customers to the premium desk with HIGH priority, the rest to the support queue). 17 tickets went to human review: 9 invalid model outputs (never guessed), 4 empty tickets and 4 over-long tickets. In the budget demo, tickets that hit the budget were also routed to human review.

## Part D

The refund agent reads `data/instamart.db` (Part A) and reuses the retrieval from the RAG assistant. Refund rules implemented in the tools: only completed orders; delivery above 45 minutes means 100% of `order_amount`; refunds above INR 1000 are escalated, not issued.

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
| How soon is money returned when a payment fails but is deducted? | 48 hours | If a payment fails but money is deducted, the amount is returned automatically 1000 must be approved by a human support lead before money is returned. | True |

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
