# OSSCA 2026 Mission Context (fork-local, mentee work)

> Fork-local note for the OSSCA 2026 GraphRAG mentee experiment. NOT part of the upstream seocho contract.
> Mentor (정이태 / tteon) experiment-design directives, Discord 2026-05-30.
> Mentor (strong): "Understand mentor intent + data context BEFORE running. Keep experimenter ethics. Rigorous verification (vector vs graph is a fact-based comparison)."

## Experiment design (must follow)

1. **Ontology = 4 size arms**: `non / small / medium / large` (FIBO sizing). Too much ontology = noise, too little = over-constrained. Compare across sizes.
   - Mentor/team prior signal: graph-only **medium 0.46 > large 0.38 > small/non 0.00** (Goldilocks: medium optimal).
2. **Slices**: 6 slices (S1–S6) × 10 samples = ~60. Sample only; no full run yet.
   - Vector baseline (team, 60/60): overall overlap 0.316 — S1 .41 / S2 .29 / S3 .38 / **S4 .11 (weakest)** / S5 .36 / S6 .34. Persisted in LanceDB `finder_vector_0530`.
3. **Models** (4 keys valid): OpenAI · xAI(grok) · DeepSeek · Moonshot(kimi). **Vector embedding = OpenAI (fixed).** My assignment = **OpenAI**.
4. **Opik logging (mandatory)** for every run. Workspace = `seocho`. Project name = `name-date-model` (e.g. `shinhyeji-0531-openai`). Tags: `model:`, `dataset_index:{slice}/{case_id}`, `prompt_hash:`, `ontology_hash:`.
5. **Extraction prompt — 3 required slots**: (a) `{{ontology}}` wired in, (b) instruction that the model is a knowledge-graph engineer, (c) `{{text}}` raw-data slot.
6. **Judge = cross-vendor** (avoid self-preference): generator ≠ judge (e.g. grok generates → OpenAI judges). temp=0, fixed prompt, reproducible. JSON parse fail = incorrect (no silent skip).
   - **3 metrics** (cheap→expensive, deterministic→semantic): `overlap` → `token_f1` → `judge`.
   - judge score quantized 0/0.5/1; verdict correct/partial/incorrect + matched + missing_or_wrong + rationale. Optional 2-judge panel (gpt+deepseek majority).
   - judge rules: factual vs gold, strip thousand separators + rounding (54.4%≈54%), weight units/period/direction, wrong scale|sign = mismatch, no-data/refusal/fabrication = incorrect.
7. **Retrieval modes**: vector / graph / vector&graph (hybrid).

## Known bias (⚠️ GitHub data = pre-bias-discovery)

- **Anchor nodes (LegalEntity/Company) missing**: metric nodes exist with values, but Company nodes / `REPORTED_METRIC` edges are not extracted → `MATCH (c)-[r]-(m)` traversal returns **0 rows**. Structural traversal itself is fragile (extraction doesn't always create company–metric edges).
- provenance is shared (same data → vector embedding & graph extraction), so arms can be tagged and compared fairly.
- Mentor's fix direction: ontology + intent + vector + topic hybrid (surface evidence via vector+topic even when structural edges are incomplete).
- ⚠️ This repo's `examples/finder/mission1/run_compare.py:graph_context` also uses `MATCH (n)-[r]-(m)` — same 0-row risk. My earlier "graph weak = numeric extraction loss" diagnosis may be conflated with this anchor bug → must re-examine.

## Reference

- Papers: RAG vs GraphRAG (arXiv 2502.11371), FinDER (2504.15800).
- Mentee repo: djskej1688/FinancialGraphRAG (KG-enhanced retrieval + financial numerical QA eval).
- `graphrag_eval_R8_R13_synthesis_report.pdf` (post-bias report), `프롬프트 모음.zip`, `message.txt` (system prompt: "Classifying Queries into Fact-Based and Reasoning-Based Categories").

Full Korean SSOT: vault `01_Projects/OSSCA_2026/멘토_실험설계_지시.md`.
