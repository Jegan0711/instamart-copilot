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
