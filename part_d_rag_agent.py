"""Part D -- policy RAG assistant (TF-IDF retrieval) and a tool-using refund agent (ReAct loop) on mock_llm."""
import glob, json, os, sqlite3, time, uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat
from mdutil import md_table, write_report

POLICY_DIR, DB, OUT, STATE = "data/policies", "data/instamart.db", "outputs", "state"
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
os.makedirs(OUT, exist_ok=True); os.makedirs(STATE, exist_ok=True)


# ============================================================ RAG
def load_docs():
    return {os.path.basename(p): open(p, encoding="utf-8").read() for p in sorted(glob.glob(f"{POLICY_DIR}/*.txt"))}


def chunk_words(text, size=40, overlap=10):
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):        # this chunk reaches the last word -> stop
            break
        start += step
    return chunks


class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = [{"source": fn, "text": c} for fn, t in load_docs().items() for c in chunk_words(t, size, overlap)]
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        idx = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in idx]


RETRIEVER = Retriever(40, 10)


def retrieve(query, k=3, retriever=None):
    return (retriever or RETRIEVER).retrieve(query, k)


RAG_SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer only from the CONTEXT supplied by the user. "
              "If the context does not contain the answer, reply that you could not find the information in the policy documents.")


def answer(question, retriever=None):
    hits = retrieve(question, 3, retriever)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = chat([{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}])
    return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits]}


def evaluate(retriever):
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = 0
    n_ans = n_none = 0
    failed = []
    for q in qa.itertuples(index=False):
        r = answer(q.question, retriever)
        if q.expected_doc == "NONE":
            n_none += 1
            fallback += r["answer"].startswith("I could not find")
        else:
            n_ans += 1
            d, f = q.expected_doc in r["sources"], q.expected_fact in r["answer"]
            doc_hit += d; fact_hit += f
            if not (d and f):
                failed.append((q.question, q.expected_fact, r["answer"], d))
    return dict(n_ans=n_ans, doc_hit=int(doc_hit), fact_hit=int(fact_hit), n_none=n_none, fallback=int(fallback), failed=failed)


# ============================================================ tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, desc, props):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": list(props)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the Instamart database by its order id.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the most relevant chunks.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ============================================================ ReAct agent
AGENT_SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, calculate_refund, "
                "issue_refund and escalate_to_human. Finish with one line that starts with REFUND, ESCALATE, DENIED or NOT_FOUND, then a colon and a short reason.")


class SimulatedCrash(Exception):
    pass


def _checkpoint(state):
    keys = ["run_id", "scenario_id", "messages", "tokens_used", "step", "tools_executed", "llm_calls", "steps", "limits"]
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump({k: state[k] for k in keys}, f, indent=1, default=str)


def _result(state, status, final):
    refund = 0.0
    for s in state["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            refund = s["observation"]["refund_inr"]
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": final,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"], "refund_inr": refund}


def _fallback(state, status):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    final = (f"HANDOFF TO HUMAN AGENT ({status}): stopped after {state['llm_calls']} model call(s), {state['tokens_used']} tokens; "
             f"tools run so far: {done}. Please review the request manually.")
    return _result(state, status, final)


def _loop(state, crash_after_tools=None):
    lim, t0 = state["limits"], time.monotonic()
    while True:
        if time.monotonic() - t0 >= lim["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")
        if state["llm_calls"] >= lim["max_iterations"]:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS")
        resp = chat(state["messages"], TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > lim["token_limit"]:           # stop BEFORE executing the requested tool
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _result(state, msg.content.split(":", 1)[0].strip(), msg.content)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        try:
            obs = TOOL_MAP[tc.name](**tc.arguments)
        except Exception as e:                                   # unknown tool or bad arguments -> observation, not a crash
            obs = {"error": str(e)}
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        _checkpoint(state)
        if crash_after_tools and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "tools_executed": [], "llm_calls": 0, "steps": [],
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state)


# ============================================================ main
if __name__ == "__main__":
    rep = []
    # ---- fresh outputs
    for p in [LEDGER, os.path.join(OUT, "agent_runs.jsonl"), os.path.join(OUT, "agent_demo_runs.jsonl"), os.path.join(OUT, "refund_ledger_demo.jsonl")] + glob.glob(f"{STATE}/*.json"):
        if os.path.exists(p):
            os.remove(p)
    open(os.path.join(STATE, ".gitkeep"), "w").close()

    # ---- chunks
    counts = {fn: len(chunk_words(t)) for fn, t in load_docs().items()}
    rep.append("### Chunks (40 words, overlap 10)\n\n" + md_table(pd.DataFrame([(k, v) for k, v in counts.items()] + [("TOTAL", sum(counts.values()))], columns=["document", "chunks"])))
    top = retrieve("refund policy late delivery")
    rep.append("**retrieve('refund policy late delivery')** top 3 sources: " + ", ".join(f"{h['source']} ({h['score']})" for h in top))

    # ---- evaluation
    e10, e0 = evaluate(RETRIEVER), evaluate(Retriever(40, 0))
    rep.append("### Retrieval and grounding evaluation (policy_eval.csv)\n\n" + md_table(pd.DataFrame([
        ("overlap 10", f"{e10['doc_hit']}/{e10['n_ans']}", f"{e10['fact_hit']}/{e10['n_ans']}", f"{e10['fallback']}/{e10['n_none']}"),
        ("overlap 0", f"{e0['doc_hit']}/{e0['n_ans']}", f"{e0['fact_hit']}/{e0['n_ans']}", f"{e0['fallback']}/{e0['n_none']}")],
        columns=["chunking", "expected_doc in top-3", "answer contains expected_fact", "fallback on unanswerable"])))
    rep.append("**Questions that break with overlap 0**\n\n" + (md_table(pd.DataFrame([(q, f, a, d) for q, f, a, d in e0["failed"]], columns=["question", "expected fact", "answer given", "doc retrieved"])) if e0["failed"] else "none"))

    # ---- scenarios
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    rows, runs_log = [], open(os.path.join(OUT, "agent_runs.jsonl"), "w", encoding="utf-8")
    for s in sc.itertuples(index=False):
        r = run_agent(s.message, s.scenario_id)
        runs_log.write(json.dumps({"scenario_id": r["scenario_id"], "status": r["status"], "final_answer": r["final_answer"], "llm_calls": r["llm_calls"],
                                   "tokens_used": r["tokens_used"], "refund_inr": r["refund_inr"], "steps": r["steps"]}) + "\n")
        exp = s.expected_outcome
        assert r["status"].startswith(exp) or r["status"] == exp or (exp == "FALLBACK" and r["status"] == "FALLBACK_MAX_ITERATIONS"), (s.scenario_id, r["status"])
        assert abs(r["refund_inr"] - float(s.expected_refund_inr)) <= 0.01, (s.scenario_id, r["refund_inr"])
        rows.append((s.scenario_id, r["status"], f"{r['refund_inr']:.2f}", r["llm_calls"], r["tokens_used"], s.expected_outcome))
    runs_log.close()
    rep.append("### The seven scenarios (outputs/agent_runs.jsonl)\n\n" + md_table(pd.DataFrame(rows, columns=["scenario", "status", "refund INR", "model calls", "tokens used", "expected"])))

    # ---- stop-safely demos
    demo = open(os.path.join(OUT, "agent_demo_runs.jsonl"), "w", encoding="utf-8")
    s1 = sc.iloc[0]
    d_tok = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-demo")
    d_time = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    assert d_tok["status"] == "FALLBACK_TOKEN_LIMIT" and d_time["status"] == "FALLBACK_TIMEOUT"
    for lbl, r in (("token_limit=200", d_tok), ("time_limit_seconds=0", d_time)):
        demo.write(json.dumps({"demo": lbl, "scenario_id": r["scenario_id"], "status": r["status"], "final_answer": r["final_answer"],
                               "llm_calls": r["llm_calls"], "tokens_used": r["tokens_used"], "steps": r["steps"]}) + "\n")
    rep.append("### Stop-safely demonstrations (S1)\n\n" + md_table(pd.DataFrame([
        ("token_limit=200", d_tok["status"], d_tok["llm_calls"], d_tok["tokens_used"], len(d_tok["steps"])),
        ("time_limit_seconds=0", d_time["status"], d_time["llm_calls"], d_time["tokens_used"], len(d_time["steps"]))],
        columns=["demo", "status", "model calls", "tokens used", "tools executed"])) +
        f"\n\nToken-limit hand-off text: {d_tok['final_answer']}")

    # ---- crash and resume (uses a separate ledger so the real ledger has no duplicate S1 refund)
    LEDGER = os.path.join(OUT, "refund_ledger_demo.jsonl")
    try:
        run_agent(s1.message, "S1", run_id="S1-crash-demo", crash_after_tools=2)
        raise AssertionError("crash was not raised")
    except SimulatedCrash as e:
        crash_msg = str(e)
    ck = json.load(open(os.path.join(STATE, "S1-crash-demo.json")))
    before = list(ck["tools_executed"])
    res = resume_agent("S1-crash-demo")
    ck2 = json.load(open(os.path.join(STATE, "S1-crash-demo.json")))
    tools = ck2["tools_executed"]
    assert tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"], tools
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) <= 0.01
    demo.write(json.dumps({"demo": "crash_and_resume", "scenario_id": "S1", "status": res["status"], "final_answer": res["final_answer"],
                           "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
    demo.close()
    rep.append("### Crash and resume (S1)\n\n" + f"Simulated crash: `{crash_msg}`. Tools executed before the crash: {', '.join(before)}. "
               f"After `resume_agent('S1-crash-demo')`: status **{res['status']}**, refund INR {res['refund_inr']:.2f}, {res['llm_calls']} model calls in total.\n\n"
               "Tools executed across the crash and the resume, in order: " + " -> ".join(tools) + f" ({len(tools)} executions, none repeated).")
    write_report("part_d.md", rep)
    print("Part D done")
