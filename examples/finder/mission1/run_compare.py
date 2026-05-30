"""Mission 1 — vector vs graph vs hybrid 비교 하니스 (real FinDER).

각 케이스: 원문(text)을 LanceDB(vector) + Neo4j(graph, seocho client)에 인덱싱 →
질문을 vector/graph/hybrid 3모드로 답변 → 정답과 비교(contains-match + LLM judge) → 집계.

T1(01_vector_vs_graph_rag.ipynb)의 검색 로직을 배치용으로 옮긴 것.

컨테이너 실행:
  docker compose -f docker-compose.tutorials.yml exec tutorials-jupyter \
    python /workspace/examples/finder/mission1/run_compare.py --per-cat 2 --ontology A
"""
import os, sys, json, time, re, argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT))

from seocho.benchmarking import load_finder_cases, normalize_answer
from seocho.store.vector import create_vector_store
from seocho.store.llm import create_llm_backend
from seocho import Ontology, Seocho
from seocho.store.graph import Neo4jGraphStore

DATASET = ROOT / "examples/finder/datasets/finder_real_3cat.json"
OUTDIR = ROOT / ".seocho/mission1"
ONTOLOGY_FILES = {
    "A": ROOT / "examples/datasets/fibo_base.jsonld",  # FIBO base (T1과 동일)
}

LLM_SPEC = os.environ.get("SEOCHO_LLM", "openai/gpt-4o-mini")
LLM_PROVIDER, LLM_MODEL = (LLM_SPEC.split("/", 1) + [""])[:2]
if not LLM_MODEL:
    LLM_PROVIDER, LLM_MODEL = "openai", LLM_SPEC

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PW = os.environ["NEO4J_PASSWORD"]

ANSWER_SYSTEM = ("Answer the question using only the supplied evidence. Be concise. "
                 "If the evidence does not contain the answer, say so.")
JUDGE_SYSTEM = ("You are a strict grader. Judge ONLY factual correctness of CANDIDATE "
                "relative to GOLD (ignore style). Numbers match within rounding; wrong "
                "scale/sign = mismatch. 'no data'/refusal/fabrication = incorrect. "
                'Output STRICT JSON only: {"verdict":"correct|partial|incorrect","score":1.0|0.5|0.0}.')

STOPWORDS = {"what", "where", "who", "when", "how", "the", "a", "an", "is", "was",
             "were", "of", "in", "to", "for", "on", "and", "did", "during", "by", "as"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.&\-]+")


def graph_context(graph_store, workspace_id, question, *, seed_limit=3, hop_limit=5):
    tokens = [t for t in TOKEN_RE.findall(question) if t.lower() not in STOPWORDS]
    facts, seen = [], set()
    for tok in tokens:
        seeds = graph_store.query(
            "MATCH (n) WHERE n._workspace_id = $workspace_id "
            "AND toLower(n.name) CONTAINS toLower($kw) "
            "RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, properties(n) AS props "
            "LIMIT $seed_limit",
            params={"workspace_id": workspace_id, "kw": tok, "seed_limit": seed_limit},
        )
        for seed in seeds:
            if seed["id"] in seen:
                continue
            seen.add(seed["id"])
            facts.append(f"{seed['label']}({seed['name']}) properties={seed['props']}")
            hops = graph_store.query(
                "MATCH (n {id: $id, _workspace_id: $workspace_id})-[r]-(m) "
                "RETURN m.id AS neighbor_id, labels(m)[0] AS neighbor_label, m.name AS neighbor_name, "
                "type(r) AS edge_type, CASE WHEN startNode(r)=n THEN 'out' ELSE 'in' END AS direction "
                "LIMIT $hop_limit",
                params={"id": seed["id"], "workspace_id": workspace_id, "hop_limit": hop_limit},
            )
            for hop in hops:
                arrow = "->" if hop["direction"] == "out" else "<-"
                facts.append(f"{seed['name']} {arrow}[{hop['edge_type']}]{arrow} "
                             f"{hop['neighbor_label']}({hop['neighbor_name']})")
    return "\n".join(facts)


def ans_vector(vector_store, llm, question, k=3):
    hits = vector_store.search(question, limit=k)
    ctx = "\n\n".join(f"[{h.id}] {h.text}" for h in hits)
    if not ctx:
        return "(no vector evidence)"
    r = llm.complete(system=ANSWER_SYSTEM, user=f"Context:\n{ctx}\n\nQuestion: {question}")
    return r.text.strip()


def ans_graph(graph_store, workspace_id, llm, question):
    ctx = graph_context(graph_store, workspace_id, question)
    if not ctx:
        return "(no graph evidence)"
    r = llm.complete(system=ANSWER_SYSTEM, user=f"Graph evidence:\n{ctx}\n\nQuestion: {question}")
    return r.text.strip()


def ans_hybrid(vector_store, graph_store, workspace_id, llm, question, k=3):
    hits = vector_store.search(question, limit=k)
    vec = "\n\n".join(f"[{h.id}] {h.text}" for h in hits)
    g = graph_context(graph_store, workspace_id, question)
    r = llm.complete(system=ANSWER_SYSTEM,
                     user=f"Passages:\n{vec}\n\nGraph evidence:\n{g}\n\nQuestion: {question}")
    return r.text.strip()


def contains_match(answer, expected):
    return normalize_answer(expected) in normalize_answer(answer)


def judge(llm, question, gold, cand):
    try:
        txt = llm.complete(system=JUDGE_SYSTEM,
                           user=f"QUESTION: {question}\nGOLD: {gold}\nCANDIDATE: {cand}").text
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        d = json.loads(m.group(0)) if m else {}
        return float(d.get("score", 0.0)), str(d.get("verdict", "?"))
    except Exception as e:
        return 0.0, f"judge_err:{type(e).__name__}"


def pick(cases, per_cat):
    if not per_cat:
        return cases
    by = defaultdict(list)
    for c in cases:
        by[c.category].append(c)
    out = []
    for cat in by:
        out.extend(by[cat][:per_cat])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=0, help="카테고리당 케이스 수(0=전부)")
    ap.add_argument("--ontology", default="A", choices=list(ONTOLOGY_FILES))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = pick(load_finder_cases(DATASET), args.per_cat)
    print(f"[setup] {len(cases)} cases | LLM={LLM_PROVIDER}/{LLM_MODEL} | ontology={args.ontology}")

    llm = create_llm_backend(provider=LLM_PROVIDER, model=LLM_MODEL)

    # 1) vector index
    vector_store = create_vector_store(kind="lancedb", uri=str(OUTDIR / "vec.lance"),
                                       table_name=f"m1_{args.ontology}")
    vector_store.add_batch([{"id": c.case_id, "text": c.text,
                             "metadata": {"category": c.category}} for c in cases])
    print(f"[vector] indexed {vector_store.count()} rows")

    # 2) graph index
    ont = Ontology.from_jsonld(str(ONTOLOGY_FILES[args.ontology]))
    gs = Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PW)
    workspace = f"m1_{args.ontology}"
    gs.execute_write("MATCH (n) WHERE n._workspace_id=$w DETACH DELETE n", params={"w": workspace})
    gs.ensure_constraints(ont)
    client = Seocho(ontology=ont, graph_store=gs, llm=llm, workspace_id=workspace)
    client.default_database = "neo4j"
    t0 = time.perf_counter()
    for i, c in enumerate(cases, 1):
        try:
            client.add(c.text, database="neo4j", category=c.category)
        except Exception as e:
            print(f"  [index err] {c.case_id}: {e}")
        if i % 10 == 0:
            print(f"  indexed {i}/{len(cases)} into graph...")
    print(f"[graph] indexed in {time.perf_counter()-t0:.0f}s")

    # 3) compare + judge
    rows = []
    for i, c in enumerate(cases, 1):
        va = ans_vector(vector_store, llm, c.question, args.k)
        ga = ans_graph(gs, workspace, llm, c.question)
        ha = ans_hybrid(vector_store, gs, workspace, llm, c.question, args.k)
        vs, vv = judge(llm, c.question, c.expected_answer, va)
        gsc, gv = judge(llm, c.question, c.expected_answer, ga)
        hs, hv = judge(llm, c.question, c.expected_answer, ha)
        rows.append({
            "case_id": c.case_id, "category": c.category, "reasoning_type": c.reasoning_type,
            "question": c.question, "expected": c.expected_answer,
            "vector": {"answer": va, "contains": contains_match(va, c.expected_answer), "judge": vs, "verdict": vv},
            "graph": {"answer": ga, "contains": contains_match(ga, c.expected_answer), "judge": gsc, "verdict": gv},
            "hybrid": {"answer": ha, "contains": contains_match(ha, c.expected_answer), "judge": hs, "verdict": hv},
        })
        if i % 5 == 0:
            print(f"  compared {i}/{len(cases)}")

    # 4) aggregate
    def agg(mode):
        n = len(rows)
        cont = sum(1 for r in rows if r[mode]["contains"]) / n
        jsc = sum(r[mode]["judge"] for r in rows) / n
        return {"contains_rate": round(cont, 3), "avg_judge": round(jsc, 3)}

    summary = {m: agg(m) for m in ("vector", "graph", "hybrid")}
    # wins by judge score
    wins = {"vector": 0, "graph": 0, "tie": 0}
    for r in rows:
        v, g = r["vector"]["judge"], r["graph"]["judge"]
        if g > v: wins["graph"] += 1
        elif v > g: wins["vector"] += 1
        else: wins["tie"] += 1
    # per-category avg judge
    by_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("vector", "graph", "hybrid"):
            by_cat[r["category"]][m].append(r[m]["judge"])
    cat_summary = {cat: {m: round(sum(v)/len(v), 3) for m, v in d.items()} for cat, d in by_cat.items()}

    out = {"n": len(rows), "ontology": args.ontology, "model": f"{LLM_PROVIDER}/{LLM_MODEL}",
           "summary": summary, "graph_vs_vector_wins": wins, "by_category": cat_summary, "rows": rows}
    fp = OUTDIR / f"compare_{args.ontology}_n{len(rows)}.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n" + "=" * 56)
    print(f"  RESULT (n={len(rows)}, ontology={args.ontology})")
    print("=" * 56)
    print(f"{'mode':<8}{'contains':>10}{'avg_judge':>12}")
    for m in ("vector", "graph", "hybrid"):
        print(f"{m:<8}{summary[m]['contains_rate']:>10}{summary[m]['avg_judge']:>12}")
    print(f"\nGraph vs Vector (judge): graph {wins['graph']} / vector {wins['vector']} / tie {wins['tie']}")
    print("\nby category (avg_judge):")
    for cat, d in cat_summary.items():
        print(f"  {cat:<18} v={d['vector']} g={d['graph']} h={d['hybrid']}")
    print(f"\nsaved -> {fp}")


if __name__ == "__main__":
    main()
