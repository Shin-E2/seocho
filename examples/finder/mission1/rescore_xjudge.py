"""Cross-vendor judge 재채점 — self-preference 편향 제거 (멘토 필수).

생성=OpenAI(gpt-4o-mini) 답변을 다른 벤더 judge(grok)로 재채점한다.
재인덱싱/재생성 없이 저장된 answer만 재평가 (djskej no-model rescoring 방식).
입력  : .seocho/mission1/compare_{arm}_n60.json (run_compare.py 산출)
출력  : .seocho/mission1/compare_{arm}_n60_xjudge.json (judge_x/verdict_x 추가)

실행 (컨테이너, XAI_API_KEY 주입):
  docker compose -f docker-compose.tutorials.yml exec -e XAI_API_KEY=$XAI_API_KEY -T \
    tutorials-jupyter python /workspace/examples/finder/mission1/rescore_xjudge.py --arm large
"""
import os, sys, json, re, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT))
from seocho.store.llm import create_llm_backend

OUTDIR = ROOT / ".seocho/mission1"

# 멘토 judge 프롬프트 (financial QA, gold 대비 factual correctness, 숫자 정규화 규칙)
JUDGE_SYSTEM = (
    "You are a strict evaluator for financial QA. You receive a QUESTION, a GOLD "
    "answer (ground truth), and a CANDIDATE answer. Judge ONLY factual correctness "
    "of CANDIDATE relative to GOLD - ignore style, verbosity, formatting.\n"
    "Rules:\n"
    "- GOLD is ground truth; judge CANDIDATE against it.\n"
    "- Weigh: (1) final answer/conclusion, (2) key figures with units & period, "
    "(3) direction/trend (increase/decrease) when asked.\n"
    "- Numbers match if equal after removing thousand separators and within normal "
    "rounding (54.4% ~ 54%). Wrong scale (thousands vs millions) or sign = mismatch.\n"
    "- CANDIDATE that says 'no data'/'not in context'/refuses, or fabricates figures "
    "not in GOLD, is INCORRECT.\n"
    "- partial when core figures are right but the final answer is incomplete or a "
    "secondary part is wrong.\n"
    'Output STRICT JSON only: {"verdict":"correct|partial|incorrect",'
    '"score":1.0|0.5|0.0,"matched":[],"missing_or_wrong":[],"rationale":"1-2 sentences"}'
)


def judge(llm, q, gold, cand):
    try:
        txt = llm.complete(
            system=JUDGE_SYSTEM,
            user=f"QUESTION:\n{q}\n\nGOLD ANSWER (ground truth):\n{gold}\n\nCANDIDATE ANSWER:\n{cand}",
        ).text
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        d = json.loads(m.group(0)) if m else {}
        return float(d.get("score", 0.0)), str(d.get("verdict", "?"))
    except Exception as e:
        return 0.0, f"err:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["non", "small", "medium", "large"])
    ap.add_argument("--judge-provider", default="grok")
    ap.add_argument("--judge-model", default="grok-4.20-reasoning")
    args = ap.parse_args()

    fp = OUTDIR / f"compare_{args.arm}_n60.json"
    data = json.loads(fp.read_text())
    rows = data["rows"]
    jl = create_llm_backend(provider=args.judge_provider, model=args.judge_model)
    print(f"[rescore] arm={args.arm} {len(rows)} rows | gen=openai/gpt-4o-mini judge={args.judge_provider}/{args.judge_model}")

    for i, r in enumerate(rows, 1):
        for m in ("vector", "graph", "hybrid"):
            s, v = judge(jl, r["question"], r["expected"], r[m]["answer"])
            r[m]["judge_x"] = s
            r[m]["verdict_x"] = v
        if i % 10 == 0:
            print(f"  rescored {i}/{len(rows)}")

    def avg(m):
        return round(sum(r[m]["judge_x"] for r in rows) / len(rows), 3)

    by_qt = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("vector", "graph", "hybrid"):
            by_qt[r["question_type"]][m].append(r[m]["judge_x"])
    qt = {q: {"n": len(d["vector"]), **{m: round(sum(v) / len(v), 3) for m, v in d.items()}}
          for q, d in by_qt.items()}

    out = {**data, "judge_cross": f"{args.judge_provider}/{args.judge_model}",
           "summary_xjudge": {m: avg(m) for m in ("vector", "graph", "hybrid")},
           "by_question_type_xjudge": qt, "rows": rows}
    op = OUTDIR / f"compare_{args.arm}_n60_xjudge.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[done] x-judge avg: vector={avg('vector')} graph={avg('graph')} hybrid={avg('hybrid')} -> {op.name}")


if __name__ == "__main__":
    main()
