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
