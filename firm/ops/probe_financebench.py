"""Probe FinanceBench evidence structure — what text fields exist?"""
from datasets import load_dataset  # type: ignore[import-untyped]
from collections import defaultdict

ds = load_dataset("PatronusAI/financebench")
print("Total rows:", len(ds["train"]))

per_doc = defaultdict(list)
for r in ds["train"]:
    per_doc[r["doc_name"]].append(r)

print(f"Unique docs: {len(per_doc)}")

# Pick first doc to inspect
first_doc = next(iter(per_doc))
rows = per_doc[first_doc]
print(f"\nFirst doc: {first_doc} (has {len(rows)} Q&A rows)")
print(f"Doc link: {rows[0]['doc_link']}")

evidence_total = 0
for i, row in enumerate(rows):
    ev = row["evidence"]
    print(f"\n--- Row {i} ---")
    print(f"Question: {row['question'][:100]}")
    print(f"Evidence count: {len(ev)}")
    for j, e in enumerate(ev[:2]):
        keys = list(e.keys()) if isinstance(e, dict) else "NOT DICT"
        text_preview = str(e.get("evidence_text", ""))[:150] if isinstance(e, dict) else ""
        print(f"  evidence[{j}] keys={keys}")
        print(f"  evidence[{j}] text preview: {text_preview!r}")
    evidence_total += len(ev)

print(f"\nTotal evidence items for this doc: {evidence_total}")
