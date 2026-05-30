"""Mission 1 — real FinDER → seocho cases JSON (3개 카테고리 필터).

FinDER(Linq-AI-Research/FinDER) train parquet 컬럼:
  _id, text(=질문), reasoning, category, references(=원문 출처 리스트), answer(=정답), type(=추론유형)

seocho cases 형식(load_finder_cases가 읽는 형식):
  {id, category, reasoning_type, text(=인덱싱할 원문), question(=질의), expected_answer(=정답)}

컨테이너에서 실행:
  docker compose -f docker-compose.tutorials.yml exec -e HF_TOKEN tutorials-jupyter \
    python /workspace/examples/finder/mission1/build_dataset.py --per-cat 20
"""
import os, json, ast, argparse
from pathlib import Path
from collections import Counter

from huggingface_hub import hf_hub_download
import pandas as pd

# 실제 parquet 카테고리 표기에 관계없이 소문자 매칭
TARGET_CATS = {"financials", "company overview", "footnotes"}
OUT = Path("/workspace/examples/finder/datasets/finder_real_3cat.json")


def parse_refs(v):
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            x = ast.literal_eval(v)
            return [str(i) for i in x] if isinstance(x, (list, tuple)) else [v]
        except Exception:
            return [v]
    return [str(v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=20, help="카테고리당 케이스 수")
    args = ap.parse_args()

    p = hf_hub_download(
        "Linq-AI-Research/FinDER",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    df = pd.read_parquet(p)
    print("loaded:", df.shape)
    print("all categories:", df["category"].value_counts().to_dict())

    df["_cat_l"] = df["category"].astype(str).str.lower().str.strip()
    cases = []
    for cat_l in sorted(TARGET_CATS):
        sub = df[df["_cat_l"] == cat_l].head(args.per_cat)
        for _, r in sub.iterrows():
            refs = parse_refs(r["references"])
            text = "\n\n".join(refs).strip()
            answer = str(r["answer"]).strip()
            if not text or not answer:
                continue
            cases.append({
                "id": str(r["_id"]),
                "category": str(r["category"]),
                "reasoning_type": str(r["type"]),
                "text": text,
                "question": str(r["text"]),
                "expected_answer": answer,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"\nwrote {len(cases)} cases -> {OUT}")
    print("by category:", dict(Counter(c["category"] for c in cases)))
    print("by reasoning_type:", dict(Counter(c["reasoning_type"] for c in cases)))


if __name__ == "__main__":
    main()
