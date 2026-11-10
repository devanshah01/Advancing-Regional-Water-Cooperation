"""Question-answering evaluation: run a query set under several conditions and summarize.

Conditions (all use the SAME code as the app):
  - RAG at each k in --ks, with each prompt style in --styles
  - a no-retrieval baseline (plain Llama 3.2) unless --no-baseline

Real run (needs Ollama running with llama3.2 and nomic-embed-text pulled):
    python evaluate.py --docs ./knowledge_base --queries test_queries.csv

Dry run (no Ollama; fake models; only checks the plumbing, numbers are meaningless):
    python evaluate.py --docs ./knowledge_base --queries test_queries.csv --dry-run

Writes results.csv (one row per query x condition x repeat) and results_summary.csv.
Then have two raters fill correct_r1/_r2, grounded_r1/_r2 (1 or 0) and error_type, and run analyze_ratings.py.

Retrieval vs generation: context_recall = share of the expected facts (context_keywords, else
expected_keywords) found in the retrieved chunks; *_when_context_complete rows isolate generation
failures (the right text was retrieved but the model refused or missed it).

error_type taxonomy: retrieval_miss | hallucination | wrong_number | arithmetic_error |
                     false_refusal | missed_refusal | incomplete | other
"""
import argparse
import csv
import re
import statistics

import rag_core as rc


# ------------------------------------------------------------------------ utils
def read_queries(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]
    for r in rows:
        r["in_scope"] = str(r.get("in_scope", "1")).strip().lower() in ("1", "true", "yes", "y")
        r["expected_keywords_list"] = [k.strip().lower() for k in (r.get("expected_keywords") or "").split(";") if k.strip()]
        ctx = (r.get("context_keywords") or "").strip() or (r.get("expected_keywords") or "")
        r["context_keywords_list"] = [k.strip().lower() for k in ctx.split(";") if k.strip()]
    return rows


def _norm(text):
    """Lowercase and drop thousands separators so '182,631' matches '182631'."""
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())


def keyword_coverage(answer, keywords):
    """Share of expected keywords found in the answer. A keyword may list alternatives
    separated by '|' (any one counts), e.g. 'downstream|tail-end'."""
    if not keywords:
        return ""
    a = _norm(answer)
    found = 0
    for k in keywords:
        if any(alt.strip() and _norm(alt).strip() in a for alt in k.split("|")):
            found += 1
    return round(found / len(keywords), 3)


def _numbers(text):
    out = set()
    for m in re.findall(r"\d[\d,]*\.?\d*", _norm(text)):
        m = m.rstrip(".,")
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if v > 10:  # ignore list numbers and tiny counts
            out.add(m.replace(",", ""))
    return out


def numbers_not_in_context(answer, query, chunks):
    """Count of numbers in the answer that appear neither in the retrieved chunks nor in the question.
    A hallucination/arithmetic PROXY: correctly computed results also show up here, so inspect them."""
    support = _numbers(query)
    for c in chunks:
        support |= _numbers(c["text"])
    return len(_numbers(answer) - support)


def mean_or_blank(values):
    values = [v for v in values if v != "" and v is not None]
    return round(statistics.mean(values), 3) if values else ""


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=rc.DOCS_FOLDER, help="folder with your .txt/.pdf knowledge base")
    ap.add_argument("--queries", default="test_queries.csv")
    ap.add_argument("--db", default="./eval_chroma_db", help="separate from the app's chroma_db")
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--styles", default="basic,structured")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1, help="runs per condition (use 3 with --temperature 0.7 for a consistency check)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=rc.CHUNK_SIZE, help="characters per chunk (paper default 500)")
    ap.add_argument("--overlap", type=int, default=rc.CHUNK_OVERLAP, help="overlap in characters (paper default 50)")
    ap.add_argument("--no-clean", action="store_true", help="do not strip web-page boilerplate from PDFs")
    ap.add_argument("--chunking", choices=["fixed", "table"], default="fixed", help="fixed = 500/50 baseline; table = keep tables whole (H1)")
    ap.add_argument("--chunk-header", action="store_true", help="prefix each chunk with its source file name before embedding")
    ap.add_argument("--dry-run", action="store_true", help="fake models; plumbing test only")
    args = ap.parse_args()

    if args.dry_run:
        rc.DRY_RUN = True
        print("*** DRY RUN: fake model outputs. Do NOT report these numbers. ***")
    embed_fn, generate_fn = rc.default_embed_fn(), rc.default_generate_fn()
    out_prefix = args.out + ("_DRYRUN" if args.dry_run else "")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    queries = read_queries(args.queries)
    if not queries:
        raise SystemExit(f"No queries found in {args.queries}")

    col, status, info = rc.build_index(args.docs, args.db, embed_fn=embed_fn, chunk_size=args.chunk_size,
                                       overlap=args.overlap, rebuild=args.rebuild or args.dry_run,
                                       clean=not args.no_clean, header=args.chunk_header, chunking=args.chunking)
    if col is None:
        raise SystemExit(f"No .txt/.pdf documents found in {args.docs}")
    print(f"Index ({status}): {info['documents']} documents, {info['chunks']} chunks "
          f"(chunk {args.chunk_size}/{args.overlap}, cleaning {'off' if args.no_clean else 'on'}, "
          f"chunk header {'on' if args.chunk_header else 'off'}, chunking {args.chunking})")
    for name, n in info["sources"]:
        print(f"  {name}: {n} chunks")

    conditions = [(k, s) for s in styles for k in ks]
    rows = []
    for q in queries:
        q_emb = embed_fn(q["query"])  # embed once, reuse across k values
        runs = [(f"k={k}|{s}", k, s) for k, s in conditions]
        if not args.no_baseline:
            runs.append(("baseline(no retrieval)", 0, "none"))
        for label, k, style in runs:
            for rep in range(args.repeats):
                out = rc.answer_query(q["query"], col, k=k, style=style, embed_fn=embed_fn,
                                      generate_fn=generate_fn, temperature=args.temperature,
                                      seed=args.seed, query_embedding=q_emb)
                refused = rc.FALLBACK.lower().rstrip(".") in out["answer"].lower()
                exp_srcs = [x.strip() for x in (q.get("expected_source") or "").split(";") if x.strip()]
                rows.append({
                    "id": q.get("id", ""), "category": q.get("category", ""), "query": q["query"],
                    "in_scope": int(q["in_scope"]), "ground_truth": q.get("ground_truth", ""),
                    "condition": label, "k": k, "style": style, "repeat": rep + 1,
                    "retrieved_sources": ";".join(out["sources"]),
                    "source_hit": ("" if not exp_srcs or style == "none" else int(any(x in out["sources"] for x in exp_srcs))),
                    "context_recall": ("" if style == "none" else keyword_coverage(
                        " ".join(c["text"] for c in out["chunks"]), q["context_keywords_list"])),
                    "keyword_coverage": keyword_coverage(out["answer"], q["expected_keywords_list"]),
                    "refused": int(refused),
                    "numbers_not_in_context": ("" if style == "none" else numbers_not_in_context(out["answer"], q["query"], out["chunks"])),
                    "seconds": out["seconds"],
                    "answer": out["answer"].replace("\r", " ").strip(),
                    "retrieved_chunks": " ||| ".join(c["text"].replace("\n", " ") for c in out["chunks"]),
                    "correct_r1": "", "correct_r2": "", "grounded_r1": "", "grounded_r2": "",
                    "error_type": "", "notes": "",
                })
        print(f"  done: {q.get('id', '')} {q['query'][:60]}")

    with open(out_prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary per condition
    summary = []
    for label in dict.fromkeys(r["condition"] for r in rows):
        sub = [r for r in rows if r["condition"] == label]
        ins = [r for r in sub if r["in_scope"]]
        outs = [r for r in sub if not r["in_scope"]]
        consistency = ""
        if args.repeats > 1:
            groups = {}
            for r in sub:
                groups.setdefault(r["id"], []).append((r["refused"], r["keyword_coverage"]))
            consistency = round(statistics.mean(1.0 if len(set(v)) == 1 else 0.0 for v in groups.values()), 3)
        summary.append({
            "condition": label,
            "n_in_scope": len(ins), "n_out_of_scope": len(outs),
            "source_hit_rate": mean_or_blank([r["source_hit"] for r in ins]),
            "context_recall": mean_or_blank([r["context_recall"] for r in ins]),
            "keyword_coverage": mean_or_blank([r["keyword_coverage"] for r in ins]),
            "refusal_rate_when_context_complete": mean_or_blank([r["refused"] for r in ins if r["context_recall"] == 1.0]),
            "keyword_coverage_when_context_complete": mean_or_blank([r["keyword_coverage"] for r in ins if r["context_recall"] == 1.0]),
            "numbers_not_in_context": mean_or_blank([r["numbers_not_in_context"] for r in ins]),
            "in_scope_refusal_rate": mean_or_blank([r["refused"] for r in ins]),
            "out_of_scope_refusal_rate": mean_or_blank([r["refused"] for r in outs]),
            "run_to_run_consistency": consistency,
            "mean_seconds": round(statistics.mean(r["seconds"] for r in sub), 2),
        })
    with open(out_prefix + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n=== SUMMARY ===")
    cols = list(summary[0].keys())
    print(" | ".join(cols))
    for s in summary:
        print(" | ".join(str(s[c]) for c in cols))
    print(f"\nWrote {out_prefix}.csv. Two raters should now fill correct_r1/_r2 and grounded_r1/_r2 (1 or 0) "
          f"and error_type; the automatic columns are only proxies. Then run analyze_ratings.py.")


if __name__ == "__main__":
    main()
