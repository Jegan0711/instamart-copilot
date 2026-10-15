"""Part A -- audit raw exports, clean them, store them in SQLite, run the seven SQL queries, pandas cross-check."""
import os, re, sqlite3
import numpy as np
import pandas as pd
from mdutil import md_table, write_report

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
for d in (CLEAN, "outputs", "reports"):
    os.makedirs(d, exist_ok=True)
rep = []


def read(name):
    # only truly blank cells are missing; the text "N/A" must stay visible as an unparseable amount
    return pd.read_csv(f"{RAW}/{name}", keep_default_na=False, na_values=[""])


customers_raw, products_raw, orders_raw = read("customers.csv"), read("products.csv"), read("orders.csv")

# ------------------------------------------------------------------ audit
rep.append("### Audit of the raw tables")
for name, df in (("customers.csv", customers_raw), ("products.csv", products_raw), ("orders.csv", orders_raw)):
    t = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                      "missing_pct": (df.isna().mean() * 100).round(2)}).rename_axis("column").reset_index()
    rep.append(f"**{name}** - shape {df.shape[0]} rows x {df.shape[1]} columns\n\n" + md_table(t))

amt_num = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
dm_num = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
age = customers_raw["age"]
audit = [
    ("raw orders.csv rows", len(orders_raw)),
    ("exact duplicate rows", int(orders_raw.duplicated().sum())),
    ("rows with status pending", int((orders_raw.status == "pending").sum())),
    ("distinct order_id values", int(orders_raw.order_id.nunique())),
    ("non-numeric order_amount entries", int((amt_num.isna() & orders_raw.order_amount.notna()).sum())),
    ("negative delivery_minutes", int((dm_num < 0).sum())),
    ("blank delivery_minutes", int(orders_raw.delivery_minutes.isna().sum())),
    ("customers: blank age", int(age.isna().sum())),
    ("customers: age outside 10-100", int(((age < 10) | (age > 100)).sum())),
    ("customers: distinct raw city spellings", int(customers_raw.city.nunique())),
    ("customers: distinct non-blank raw gender spellings", int(customers_raw.gender.nunique())),
    ("customers: blank gender", int(customers_raw.gender.isna().sum())),
]
rep.append("**Audit figures**\n\n" + md_table(pd.DataFrame(audit, columns=["check", "value"])))

# ------------------------------------------------------------------ clean customers
changes = []
c = customers_raw.copy()
before = c.city.copy()
c["city"] = c.city.str.strip().str.title()
changes.append(("customers", "city: strip spaces + title case", int((before != c.city).sum()), f"{c.city.nunique()} cities remain"))

gmap = {"male": "male", "m": "male", "female": "female", "f": "female", "": "unknown"}
before = c.gender.copy()
c["gender"] = c.gender.fillna("").str.strip().str.lower().map(gmap)
assert c.gender.notna().all()
changes.append(("customers", "gender mapped to male/female/unknown", int((before.fillna("") != c.gender).sum()), "blank -> unknown (30 rows)"))

bad_age = (c.age < 10) | (c.age > 100)
c.loc[bad_age, "age"] = np.nan
n_missing_age = int(c.age.isna().sum())
c["age"] = c.age.fillna(c.groupby("city")["age"].transform("median"))
changes.append(("customers", "age outside 10-100 set to missing", int(bad_age.sum()), ""))
changes.append(("customers", "missing age filled with city median (groupby+transform)", n_missing_age, f"{int(c.age.isna().sum())} missing left"))
c["signup_date"] = pd.to_datetime(c["signup_date"])
changes.append(("customers", "signup_date converted to datetime", len(c), ""))

# ------------------------------------------------------------------ clean orders
o = orders_raw.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])
changes.append(("orders", "1. order_ts converted to datetime", len(o), ""))
n = len(o); o = o.drop_duplicates(); changes.append(("orders", "2. exact duplicate rows removed", n - len(o), f"{len(o)} rows left"))
n = len(o); o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
changes.append(("orders", "3. keep latest order_ts per order_id (drops earlier pending records)", n - len(o), f"{len(o)} rows left"))
o = o.merge(products_raw[["product_id", "category", "price_inr"]], on="product_id", how="left")
amt = pd.to_numeric(o["order_amount"], errors="coerce")
calc = (o.price_inr * o.quantity * (1 - o.discount_pct / 100)).round(2)
n_bad_amt = int(amt.isna().sum())
o["order_amount"] = amt.fillna(calc)
changes.append(("orders", "4. unparseable order_amount refilled from price x qty x (1-disc)", n_bad_amt, f"{int(o.order_amount.isna().sum())} missing left"))
neg = o.delivery_minutes < 0
o.loc[neg, "delivery_minutes"] = np.nan
changes.append(("orders", "5a. negative delivery_minutes set to missing", int(neg.sum()), ""))
done = o.status == "completed"
med = o[done].groupby("category")["delivery_minutes"].transform("median")
n_fill = int(o.loc[done, "delivery_minutes"].isna().sum())
o.loc[done, "delivery_minutes"] = o.loc[done, "delivery_minutes"].fillna(med)
changes.append(("orders", "5b. completed: missing delivery_minutes filled with category median", n_fill, ""))
changes.append(("orders", "5c. cancelled: delivery_minutes left missing (never delivered)", int((o.status == "cancelled").sum()), "not imputed"))
orders = o[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]]
orders = orders.sort_values("order_id").reset_index(drop=True)

rep.append("### Cleaning rules and rows/values changed\n\n" + md_table(pd.DataFrame(changes, columns=["table", "rule", "rows/values changed", "note"])))

cmp_ = c.city.value_counts().rename_axis("city").reset_index(name="customers")
rep.append("**Customers after cleaning**\n\n" + md_table(cmp_) + "\n\n" + md_table(c.gender.value_counts().rename_axis("gender").reset_index(name="customers")) +
           f"\n\nMissing ages: {int(c.age.isna().sum())}; age range {c.age.min():.0f}-{c.age.max():.0f}.")
st = orders.status.value_counts()
rep.append("**Orders after cleaning**\n\n" + md_table(pd.DataFrame([
    ("rows", len(orders)), ("completed", int(st["completed"])), ("cancelled", int(st["cancelled"])),
    ("pending", int((orders.status == "pending").sum())), ("missing order_amount", int(orders.order_amount.isna().sum())),
    ("missing delivery_minutes (completed)", int(orders[orders.status == "completed"].delivery_minutes.isna().sum())),
    ("missing delivery_minutes (cancelled)", int(orders[orders.status == "cancelled"].delivery_minutes.isna().sum())),
    ("sum of order_amount, completed (INR)", f"{orders[orders.status == 'completed'].order_amount.sum():,.2f}")], columns=["check", "value"])))

# ------------------------------------------------------------------ store
customers_out = c.assign(signup_date=c.signup_date.dt.strftime("%Y-%m-%d"))
orders_out = orders.assign(order_ts=orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S"))
customers_out.to_csv(f"{CLEAN}/customers.csv", index=False)
products_raw.to_csv(f"{CLEAN}/products.csv", index=False)
orders_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
customers_out.to_sql("customers", con, index=False)
products_raw.to_sql("products", con, index=False)
orders_out.to_sql("orders", con, index=False)

# ------------------------------------------------------------------ SQL
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d+)[^\n]*\n", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}
res = {k: pd.read_sql_query(q, con) for k, q in queries.items()}
sql_rows = []
q1, q2, q3, q4, q5, q6, q7 = (res[f"Q{i}"] for i in range(1, 8))
sql_rows.append(("Q1", "customers with total of completed orders > 500 above 8000", f"{len(q1)} customers"))
top = q2.iloc[0]
sql_rows.append(("Q2", "revenue by city x category", f"{len(q2)} rows; sum INR {q2.revenue.sum():,.2f}; largest {top.city} / {top.category} INR {top.revenue:,.2f}"))
sql_rows.append(("Q3", "products priced above the average catalogue price", f"{len(q3)} products"))
sql_rows.append(("Q4", "customers with no completed order", f"{len(q4)} customers"))
sql_rows.append(("Q5", "customers above the average completed spend", f"{len(q5)} customers; average INR {q5.avg_total.iloc[0]:,.2f}"))
diff = int((q6.rnk != q6.row_num).sum())
sql_rows.append(("Q6", "rank / dense_rank / row_number per city", f"{len(q6)} rows; {diff} rows with RANK() != ROW_NUMBER()"))
sql_rows.append(("Q7", "daily revenue with running total", f"{len(q7)} daily rows; last cumulative INR {q7.cumulative_revenue.iloc[-1]:,.2f}"))
rep.append("### SQL results (sql/queries.sql, run with pd.read_sql_query)\n\n" + md_table(pd.DataFrame(sql_rows, columns=["query", "what", "result"])))
t6 = q6[q6.rnk == 1][["city", "customer_id", "total_spend"]].copy()
rep.append("**Q6 top customer per city**\n\n" + md_table(t6))

# ------------------------------------------------------------------ pandas cross-check
full = pd.read_sql_query("""SELECT o.*, c.city, p.category FROM orders o JOIN customers c USING(customer_id)
                            JOIN products p USING(product_id)""", con)
comp = full[full.status == "completed"]
pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total").round(2)
grand = float(pv.loc["Total", "Total"])
assert abs(grand - q2.revenue.sum()) < 0.01
rep.append(f"### pandas cross-check\n\npivot_table grand total = INR {grand:,.2f}; SQL Q2 total = INR {q2.revenue.sum():,.2f}; equal: {abs(grand - q2.revenue.sum()) < 0.01}\n\n" + md_table(pv.round(2), index=True))
ga = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), orders=("order_amount", "count")).round(2).reset_index()
rep.append("**Per city: sum, mean, count of completed order_amount**\n\n" + md_table(ga))

x = comp.order_amount
q1_, q3_ = x.quantile(.25), x.quantile(.75); iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
stats = pd.DataFrame([("mean", f"{x.mean():.2f}"), ("median", f"{x.median():.2f}"), ("skewness", f"{x.skew():.2f}"),
                      ("Q1", f"{q1_:.2f}"), ("Q3", f"{q3_:.2f}"), ("lower fence", f"{lo:.2f}"), ("upper fence", f"{hi:.2f}"),
                      ("outliers below lower fence", int((x < lo).sum())), ("outliers above upper fence", int((x > hi).sum()))], columns=["statistic", "value"])
rep.append("**Distribution of completed order_amount**\n\n" + md_table(stats))
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
pairs.sort(key=lambda t: -abs(t[2]))
rep.append("**Correlation matrix (completed orders)**\n\n" + md_table(corr.round(3), index=True) +
           f"\n\nLargest absolute correlation: {pairs[0][0]} vs {pairs[0][1]} = {pairs[0][2]:.3f}.")
write_report("part_a.md", rep)
con.close()
print("Part A done:", len(customers_out), "customers,", len(orders_out), "orders")
