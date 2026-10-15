"""Part C -- support ticket triage with guardrails, retries, output validation, token budget and routing."""
import json, os, random, re, time
import pandas as pd
import mock_llm
from mock_llm import chat, est_tokens, TransientLLMError
from mdutil import md_table, write_report

MAX_INPUT_TOKENS = 400
BASE_DELAY, MAX_RETRIES = 0.05, 3
CATS = ["billing", "delivery", "technical", "fraud"]
random.seed(42)
os.makedirs("outputs", exist_ok=True)

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
segments = pd.read_csv("outputs/customer_segments.csv")
seg_name = dict(zip(segments.customer_id, segments.segment_name))
seg_prem = dict(zip(segments.customer_id, segments.is_premium.astype(str).str.lower() == "true"))

# ------------------------------------------------------------------ redaction (order matters: email, phone, order id)
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text):
    counts = {}
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        counts[name] = n
    return text, counts


def has_pii(text):
    return any(p.search(text) for _, p, _ in PATTERNS)


# ------------------------------------------------------------------ prompts
ZERO_SHOT = ('You are a support ticket router. CLASSIFY the customer ticket into exactly one category: '
             'billing, delivery, technical or fraud. Reply with JSON only, no other text, in the form '
             '{"category": "<category>", "reason": "<short reason>"}.')
FEW_SHOT = ZERO_SHOT + '''
EXAMPLES:
Ticket: Someone used my wallet balance without permission, this is fraud. -> {"category": "fraud", "reason": "unauthorised use of account"}
Ticket: I was charged twice for [ORDER_ID], please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: The app crashes whenever I open the cart. -> {"category": "technical", "reason": "app fault"}
Ticket: My order has not arrived and the rider does not answer. -> {"category": "delivery", "reason": "late or missing delivery"}'''
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT and "fraud" in FEW_SHOT


# ------------------------------------------------------------------ resilient call
def call_with_retry(messages, ticket_id, log):
    retries = 0
    while True:
        try:
            return chat(messages)
        except TransientLLMError as e:
            if retries >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retries * random.uniform(0.9, 1.15)
            log.append(f"RETRY ticket={ticket_id} retry_number={retries} wait={wait:.4f}s reason={e}")
            time.sleep(wait)
            retries += 1


# ------------------------------------------------------------------ output validation
def validate(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = words & set(CATS)
        return ("classified_keyword", found.pop()) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and isinstance(obj.get("category"), str) and obj["category"] in CATS:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""


def route(status, category, customer_id):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if seg_prem.get(customer_id, False):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_pipeline(system_prompt, token_budget=None):
    mock_llm.reset()                                   # fresh counters for every full run
    log, prompts, total, rows = [], [], 0, []
    for t in tickets.itertuples(index=False):
        rec = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, status="", category="", tokens_used=0, true_category=t.true_category)
        if not t.text.strip():
            rec["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            red, _ = redact(t.text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red}]
            assert not has_pii(messages[1]["content"]), "PII reached a prompt"
            prompts.append(messages[1]["content"])
            resp = call_with_retry(messages, t.ticket_id, log)
            rec["tokens_used"] = resp.usage.total_tokens
            total += resp.usage.total_tokens
            rec["status"], rec["category"] = validate(resp.choices[0].message.content)
        rec["segment_name"] = seg_name.get(t.customer_id, "")
        rec["route"], rec["priority"] = route(rec["status"], rec["category"], t.customer_id)
        rows.append(rec)
    assert not any(has_pii(p) for p in prompts)
    return pd.DataFrame(rows), log, total


def summarise(df, total):
    ok = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = int((ok.category == ok.true_category).sum())
    return dict(classified=len(ok), correct=correct, accuracy=100 * correct / len(ok), total_tokens=total,
                tokens_per_ticket=total / len(ok))


COLS = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
rep = []

# ---- guardrail + redaction facts
n_empty = int((tickets.text.str.strip() == "").sum())
long_tok = [est_tokens(x) for x in tickets.text if x.strip() and est_tokens(x) > MAX_INPUT_TOKENS]
tot = {"email": 0, "phone": 0, "order_id": 0}
redacted_all = []
for x in tickets.text:
    r, c = redact(x)
    redacted_all.append(r)
    for k in tot:
        tot[k] += c[k]
left = sum(len(p.findall(r)) for r in redacted_all for _, p, _ in PATTERNS)
assert left == 0
rep.append("### Input guardrails and redaction\n\n" + md_table(pd.DataFrame([
    ("empty tickets (skipped_empty, no model call)", n_empty),
    (f"tickets above MAX_INPUT_TOKENS={MAX_INPUT_TOKENS} (flagged_long, no model call)", f"{len(long_tok)} (each estimated at {long_tok[0]} tokens)"),
    ("emails redacted", tot["email"]), ("phone numbers redacted", tot["phone"]), ("order ids redacted", tot["order_id"]),
    ("total substitutions", sum(tot.values())), ("PII matches left after re-applying all three patterns", left)], columns=["item", "value"])))
rep.append(f"System prompt sizes: zero-shot {est_tokens(ZERO_SHOT)} estimated tokens, few-shot {est_tokens(FEW_SHOT)} estimated tokens.")

# ---- zero-shot and few-shot, no budget
zs, zlog, ztot = run_pipeline(ZERO_SHOT)
fs, flog, ftot = run_pipeline(FEW_SHOT)
with open("outputs/retry_log_zero_shot.txt", "w") as f:
    f.write("\n".join(zlog) + "\n")
waits = [float(re.search(r"wait=([\d.]+)s", l).group(1)) for l in zlog]
assert all(0.045 <= w <= 0.0575 for w in waits) and all("retry_number=0" in l for l in zlog)
sz, sf = summarise(zs, ztot), summarise(fs, ftot)
stat = lambda d: d.status.value_counts().to_dict()
rep.append("### Resilient calls (zero-shot full run)\n\n" + f"TransientLLMErrors raised: {len(zlog)}; all recovered by a retry (no permanent errors). Wait range observed: {min(waits):.4f}-{max(waits):.4f} s (allowed for retry 0: 0.045-0.0575 s).\n\nFirst retry log lines:\n\n```\n" + "\n".join(zlog[:3]) + "\n```")
rep.append("### Status counts per full run\n\n" + md_table(pd.DataFrame(
    [(s, stat(zs).get(s, 0), stat(fs).get(s, 0)) for s in ["classified_json", "classified_keyword", "flagged_invalid_output", "skipped_empty", "flagged_long"]],
    columns=["status", "zero-shot", "few-shot"])))
rep.append("### Zero-shot vs few-shot\n\n" + md_table(pd.DataFrame([
    ("zero-shot", sz["classified"], f"{sz['correct']}/{sz['classified']} = {sz['accuracy']:.1f}%", sz["total_tokens"], round(sz["tokens_per_ticket"], 1)),
    ("few-shot", sf["classified"], f"{sf['correct']}/{sf['classified']} = {sf['accuracy']:.1f}%", sf["total_tokens"], round(sf["tokens_per_ticket"], 1))],
    columns=["prompt", "tickets classified", "accuracy", "total tokens", "tokens per classified ticket"])))
okz = zs[zs.status.isin(["classified_json", "classified_keyword"])]
ct = pd.crosstab(okz.true_category, okz.category).reindex(index=["billing", "delivery", "technical", "fraud"], columns=["billing", "delivery", "technical", "fraud"], fill_value=0)
rep.append("**Zero-shot confusion table (rows = true category, columns = predicted)**\n\n" + md_table(ct, index=True))

# ---- budget demo
budget = 6000
bd, blog, btot = run_pipeline(ZERO_SHOT, token_budget=budget)
bd[COLS].to_csv("outputs/triage_budget_demo.csv", index=False)
n_ex = int((bd.status == "budget_exhausted").sum())
assert n_ex >= 1
rep.append(f"### Token budget demo (zero-shot, TOKEN_BUDGET={budget})\n\n{n_ex} tickets ended **budget_exhausted** and were routed to human review; tokens spent before stopping: {btot}.")

# ---- final result
fin, _, ftot2 = run_pipeline(FEW_SHOT, token_budget=100000)
fin[COLS].to_csv("outputs/triage_results.csv", index=False)
assert len(fin) == 164
rc = fin.groupby(["route", "priority"]).size().reset_index(name="tickets")
rep.append("### Final routing (few-shot, TOKEN_BUDGET=100000) - outputs/triage_results.csv\n\n" + md_table(rc) +
           f"\n\nRule-validated (classified_json or classified_keyword): {int(fin.status.isin(['classified_json', 'classified_keyword']).sum())} tickets. "
           f"Sent to human review: {int((fin.route == 'human_review').sum())} tickets ({', '.join(f'{k}: {v}' for k, v in fin[fin.route == 'human_review'].status.value_counts().items())}).")
write_report("part_c.md", rep)
print("Part C done:", sz, sf, "budget exhausted:", n_ex)
