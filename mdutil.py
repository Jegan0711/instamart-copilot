"""Tiny helpers to write Markdown report fragments (used to build README.md)."""
import os
import pandas as pd

def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)

def md_table(df, index=False):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)

def write_report(name, parts):
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", name), "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")
