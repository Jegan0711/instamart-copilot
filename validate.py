"""Validate the Part A reference checkpoints. Prints ALL CHECKPOINTS PASSED when every assertion holds."""
import sqlite3
import pandas as pd

raw_o = pd.read_csv("data/raw/orders.csv", keep_default_na=False, na_values=[""])
raw_c = pd.read_csv("data/raw/customers.csv", keep_default_na=False, na_values=[""])
cc = pd.read_csv("data/clean/customers.csv")
co = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")
done = co[co.status == "completed"]

checks = [
    ("raw order rows", len(raw_o) == 13084),
    ("exact duplicates", raw_o.duplicated().sum() == 119),
    ("pending rows", (raw_o.status == "pending").sum() == 656),
    ("distinct order_id", raw_o.order_id.nunique() == 12309),
    ("non-numeric amounts", (pd.to_numeric(raw_o.order_amount, errors="coerce").isna() & raw_o.order_amount.notna()).sum() == 120),
    ("negative delivery", (pd.to_numeric(raw_o.delivery_minutes, errors="coerce") < 0).sum() == 106),
    ("blank delivery", raw_o.delivery_minutes.isna().sum() == 2541),
    ("blank ages", raw_c.age.isna().sum() == 61),
    ("ages outside 10-100", ((raw_c.age < 10) | (raw_c.age > 100)).sum() == 6),
    ("raw city spellings", raw_c.city.nunique() == 24),
    ("raw gender spellings", raw_c.gender.nunique() == 10),
    ("blank genders", raw_c.gender.isna().sum() == 30),
    ("six cities with reference counts", cc.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("genders", cc.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("no missing age, 18-60", cc.age.isna().sum() == 0 and cc.age.between(18, 60).all()),
    ("clean order rows / unique", len(co) == 12309 and co.order_id.is_unique),
    ("completed / cancelled", (co.status == "completed").sum() == 11110 and (co.status == "cancelled").sum() == 1199),
    ("no pending", (co.status == "pending").sum() == 0),
    ("no missing amount", co.order_amount.isna().sum() == 0),
    ("delivery missing only for cancelled", done.delivery_minutes.isna().sum() == 0 and co[co.status == "cancelled"].delivery_minutes.isna().sum() == 1199),
    ("completed revenue", round(done.order_amount.sum(), 2) == 2998668.80),
    ("db tables", {r[0] for r in con.execute("select name from sqlite_master where type='table'")} == {"customers", "products", "orders"}),
    ("db order count", con.execute("select count(*) from orders").fetchone()[0] == 12309),
]
for name, ok in checks:
    assert ok, f"checkpoint failed: {name}"
print("ALL CHECKPOINTS PASSED")
