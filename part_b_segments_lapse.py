"""Part B -- RFM table, K-means segments, PCA, lapse prediction (leakage-free), persistence, retention list."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mdutil import md_table, write_report

os.makedirs("outputs", exist_ok=True)
rep = []
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
done = orders[orders.status == "completed"]

# ------------------------------------------------------------------ 1. RFM
SNAP = pd.Timestamp("2025-10-01 00:00:00")
rfm = done.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum")).reset_index()
rfm["recency"] = (SNAP - rfm.last_ts).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
rep.append("### RFM table (snapshot 2025-10-01)\n\n" + md_table(pd.DataFrame([
    ("customers with >= 1 completed order", len(rfm)),
    ("recency median / min / max", f"{rfm.recency.median():.0f} / {rfm.recency.min()} / {rfm.recency.max()}"),
    ("frequency mean / max", f"{rfm.frequency.mean():.2f} / {rfm.frequency.max()}"),
    ("monetary mean / max (INR)", f"{rfm.monetary.mean():,.2f} / {rfm.monetary.max():,.2f}")], columns=["statistic", "value"])))

# ------------------------------------------------------------------ 2. K-means
scaler = StandardScaler()
Xs = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
ktab = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    ktab.append((k, round(km.inertia_, 1), round(silhouette_score(Xs, km.labels_), 3)))
ktab = pd.DataFrame(ktab, columns=["K", "inertia", "silhouette"])
rep.append("### K-means for K = 2..8 (standardised RFM, n_init=10, random_state=42)\n\n" + md_table(ktab))
cand = ktab[ktab.K.isin([3, 4, 5])].sort_values("silhouette", ascending=False)
K = 4
assert cand.iloc[0].K == 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Xs)
rfm["segment_id"] = km.labels_

# ------------------------------------------------------------------ 3. profile, names, PCA
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"))
prem = prof.mean_monetary.idxmax()
rest = prof.drop(prem)
freq_seg = rest.mean_frequency.idxmax()
rest2 = rest.drop(freq_seg)
dormant = rest2.mean_recency.idxmax()
occasional = rest2.drop(dormant).index[0]
NAMES = {prem: ("Premium High-Value Spenders", "personal account manager, priority support and early electronics offers"),
         freq_seg: ("Frequent Essentials Regulars", "protect the habit: loyalty points, bundles and a subscription for staples"),
         occasional: ("Occasional Light Buyers", "nudge a second order with a small coupon on groceries and dairy"),
         dormant: ("Dormant At-Risk Buyers", "win-back campaign with a time-limited discount before they churn")}
rfm["segment_name"] = rfm.segment_id.map(lambda s: NAMES[s][0])
rfm["is_premium"] = rfm.segment_id == prem
prof = prof.round(2).reset_index()
prof["segment_name"] = prof.segment_id.map(lambda s: NAMES[s][0])
prof["recommended_action"] = prof.segment_id.map(lambda s: NAMES[s][1])
prof["premium"] = prof.segment_id == prem
rep.append("### Segment profile (K=4)\n\n" + md_table(prof))
pca = PCA(n_components=2, random_state=42).fit(Xs)
evr = pca.explained_variance_ratio_
rep.append(f"**PCA (2 components) explained_variance_ratio_**: {evr[0]:.3f} and {evr[1]:.3f}, total {evr.sum():.3f}")
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)

# ------------------------------------------------------------------ 4. lapse problem
CUT, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
pre = orders[orders.order_ts < CUT]                      # ONLY orders before the cutoff feed the features
pre_done = pre[pre.status == "completed"].merge(products[["product_id", "category"]], on="product_id")
future = orders[(orders.order_ts >= CUT) & (orders.order_ts < END) & (orders.status == "completed")]
g = pre_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUT - g.order_ts.max()).dt.days,
    "frequency": g.order_id.nunique(),
    "monetary": g.order_amount.sum(),
    "avg_discount": g.discount_pct.mean(),
    "avg_delivery": g.delivery_minutes.mean(),
    "fav_category": g.category.agg(lambda s: s.mode().iloc[0]),
})
feat["cancel_rate"] = pre.assign(c=pre.status == "cancelled").groupby("customer_id").c.mean().reindex(feat.index)
cu = customers.set_index("customer_id").loc[feat.index]
feat["city"], feat["gender"], feat["age"] = cu.city, cu.gender, cu.age
feat["tenure_days"] = (CUT - cu.signup_date).dt.days
y = (~feat.index.isin(future.customer_id)).astype(int)
y = pd.Series(y, index=feat.index, name="lapsed")
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = feat[NUM + CAT]
assert X.isna().sum().sum() == 0
assert pre.order_ts.max() < CUT
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
base_pred = np.zeros(len(y_te), dtype=int)
base_acc, base_f1 = accuracy_score(y_te, base_pred), f1_score(y_te, base_pred, zero_division=0)
rep.append("### Lapse problem (cutoff 2025-09-01)\n\n" + md_table(pd.DataFrame([
    ("cohort (completed order before cutoff)", len(X)), ("lapsed", f"{int(y.sum())} ({y.mean() * 100:.2f}%)"),
    ("train / test rows", f"{len(X_tr)} / {len(X_te)}"),
    ("baseline 'never lapsed' test accuracy", f"{base_acc * 100:.1f}%"), ("baseline lapsed-class F1", f"{base_f1:.2f}"),
    ("features", ", ".join(NUM + CAT)), ("missing values in X", int(X.isna().sum().sum()))], columns=["item", "value"])))


def make_pipe(clf):
    pre_ = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", pre_), ("clf", clf)])


def metrics(model, Xe, ye):
    p = model.predict(Xe)
    return dict(cm=confusion_matrix(ye, p), precision=precision_score(ye, p, zero_division=0),
                recall=recall_score(ye, p, zero_division=0), f1=f1_score(ye, p, zero_division=0))


# ------------------------------------------------------------------ 5/6. tuning (preprocessing is inside the Pipeline => refitted per fold on training folds only)
gl = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)), {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
gr = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                  {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
lr, rf = gl.best_estimator_, gr.best_estimator_
sc = lr.named_steps["prep"].named_transformers_["num"]
learned = sc.mean_[NUM.index("frequency")]
rep.append("### Preprocessing without leakage\n\nOne-hot (drop first) and StandardScaler live inside a scikit-learn Pipeline fitted on training rows only. "
           f"Scaler mean of frequency (train rows only): **{learned:.3f}**; full-cohort mean of frequency: **{X.frequency.mean():.3f}**. "
           "They differ because the scaler never sees the 232 test rows, so its mean comes from a different (training-only) subset. "
           "No feature column has a missing value (asserted in code).")
rep.append("### Tuning results (cv=5, scoring=f1)\n\n" + md_table(pd.DataFrame([
    ("LogisticRegression", f"C = {gl.best_params_['clf__C']}", round(gl.best_score_, 3)),
    ("RandomForest (100 trees)", f"max_depth = {gr.best_params_['clf__max_depth']}, min_samples_leaf = {gr.best_params_['clf__min_samples_leaf']}", round(gr.best_score_, 3))],
    columns=["model", "best parameters", "best CV F1"])))
mlr, mrf = metrics(lr, X_te, y_te), metrics(rf, X_te, y_te)
rows = []
for name, m in (("Baseline (never lapsed)", dict(cm=confusion_matrix(y_te, base_pred), precision=0, recall=0, f1=0)),
                ("Logistic regression", mlr), ("Random forest", mrf)):
    cm = m["cm"]
    rows.append((name, f"[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]", round(m["precision"], 3), round(m["recall"], 3), round(m["f1"], 3)))
rep.append("### Test-set evaluation (lapsed = positive class; confusion matrix rows = actual, columns = predicted [not lapsed, lapsed])\n\n" +
           md_table(pd.DataFrame(rows, columns=["model", "confusion matrix", "precision", "recall", "F1"])))
selected, sel_name = (rf, "Random forest") if gr.best_score_ >= gl.best_score_ else (lr, "Logistic regression")
msel = mrf if sel_name == "Random forest" else mlr
assert msel["f1"] >= 0.55
rep.append(f"**Selected model (higher best CV F1, chosen before looking at the test set): {sel_name}**; test F1 = {msel['f1']:.3f}.")

# ------------------------------------------------------------------ 7. class balance experiment (training set only)
tr = X_tr.copy(); tr["lapsed"] = y_tr
n_add = int((tr.lapsed == 0).sum() - (tr.lapsed == 1).sum())
extra = tr[tr.lapsed == 1].sample(n=n_add, replace=True, random_state=42)
bal = pd.concat([tr, extra])
lr_bal = clone(lr).fit(bal[NUM + CAT], bal.lapsed)
mbal = metrics(lr_bal, X_te, y_te)
rep.append(f"### Class-balance experiment (over-sampling lapsed training rows with replacement, random_state=42)\n\nTraining rows after balancing: {int((bal.lapsed == 0).sum())} not lapsed / {int((bal.lapsed == 1).sum())} lapsed.\n\n" +
           md_table(pd.DataFrame([("Logistic regression, original", round(mlr["precision"], 3), round(mlr["recall"], 3), round(mlr["f1"], 3)),
                                  ("Logistic regression, balanced training set", round(mbal["precision"], 3), round(mbal["recall"], 3), round(mbal["f1"], 3))],
                                 columns=["model", "precision", "recall", "F1"])))

# ------------------------------------------------------------------ 8. interpretation, persistence, retention list
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(5)
rep.append("### Top-5 random-forest feature importances\n\n" + md_table(imp.round(4).rename_axis("feature").reset_index(name="importance")))
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["prep"].get_feature_names_out())
rep.append("### Logistic-regression coefficients (standardised numeric features)\n\n" + md_table(coef[[f"num__{n}" for n in NUM]].round(3).rename_axis("feature").reset_index(name="coefficient")))
joblib.dump(selected, "outputs/lapse_model.joblib")
re_loaded = joblib.load("outputs/lapse_model.joblib")
same = bool(np.array_equal(re_loaded.predict(X_te), selected.predict(X_te)) and np.allclose(re_loaded.predict_proba(X_te), selected.predict_proba(X_te)))
assert same
rep.append(f"**joblib round trip**: reloaded model reproduces the original test predictions exactly: {same}.")
seg = pd.read_csv("outputs/customer_segments.csv")
prem_ids = set(seg[seg.is_premium].customer_id)
sub = X_te[X_te.index.isin(prem_ids)]
ret = pd.DataFrame({"customer_id": sub.index, "lapse_probability": selected.predict_proba(sub)[:, 1].round(4),
                    "actually_lapsed": y_te.loc[sub.index].values}).sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
rep.append(f"### Retention list\n\n{len(sub)} test-set customers are in the Premium segment; the top {len(ret)} by predicted lapse probability were written to outputs/retention_list.csv. "
           f"Of these, {int(ret.actually_lapsed.sum())} truly lapsed.\n\n" + md_table(ret))
write_report("part_b.md", rep)
print("Part B done; selected:", sel_name, "test F1", round(msel["f1"], 3))
