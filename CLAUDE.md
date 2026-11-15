# CLAUDE.md — TopoChunker

## Environment & commands

Plain-`pip` project (README says `pip install -r requirements.txt`), **but there
is no `requirements.txt`, no `pyproject.toml`, no `uv.lock`, and no `.venv`**.
On this box use uv: `uv venv --python 3.12 && uv pip install ...`. Deps are not
pinned anywhere — inferred from imports:

- `langchain-core`, `langchain-openai`, `langgraph` (agent loops + `@tool`)
- `pydantic` (core/types.py)
- `pymupdf` (`fitz`), `transformers`, `torch`, `numpy` (processing)
- Visual path only: `mineru` (MinerU, `backend='vlm-transformers'`)

Run (per README):
```bash
python demo_full_pipeline.py /abs/path/to.pdf --provider deepseek --model deepseek-chat --temp 0.1
```
Needs `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`. Outputs to `outputs/<stem>/`:
`topology_sir.json`, `topology_graph.mmd`, `final_knowledge_base.jsonl`.

**No test suite, no linter, no CI.**

## What this does

Multi-stage PDF → RAG-chunk pipeline. An **Inspector** agent probes a PDF and
routes it to one of three markdown extractors; a **SIRBuilder** parses that
markdown into a hierarchical tree; a **Refiner** agent walks the tree and emits
context-enriched JSONL chunks (title + optional pronoun-resolution supplement +
verbatim body). Both agents are LangGraph `StateGraph` tool-calling loops over a
DeepSeek/OpenAI chat model. Much of the code (prompts, comments) is in Chinese.

## Architecture

1. `demo_full_pipeline.py` — entrypoint. Builds one shared `ChatOpenAI` via
   `agents/llm_factory.create_llm`, runs the 4 phases, writes artifacts.
2. **Phase 1 — `agents/inspector.py::InspectorAgent`**: LangGraph loop calling
   three `@tool`s in `agents/tools.py`, backed by `agents/probes.py`
   (`MetaReader` scan/image/table detection, `LayoutProbe` complexity,
   `RegexTester` structure score). Emits a `core/types.Blueprint` with
   `selected_path ∈ {STRUCTURAL_RULE, SEMANTIC_FLOW, VISUAL_TOPOLOGY}`.
3. **Phase 2 — `processing/`**: `StructuralRuleProcessor` (pure PyMuPDF + regex
   heading hierarchy → markdown), `SemanticFlowProcessor` (wraps
   `LongformerSemanticSplitter`, a token-classification splitter), or
   `VisualTopologyProcessor` (MinerU VLM). All subclass `processing/base.py`.
4. **Phase 3 — `core/sir.py::SIRBuilder`**: stack-based markdown → `SIRNode`
   tree; computes `token_count`/`total_tree_token`, `path_str`, atomicity.
5. **Phase 4 — `agents/refiner.py::RefinerAgent`**: recursive divide-and-conquer
   over the tree; per node a LangGraph loop uses tools in
   `agents/refiner_tools.py` (`Semantic_Slicer`=Longformer split, `SIR_Query`=
   lineage/sibling lookup, `Chunk_Enhancer`=LLM title). ~half the 682 lines are
   manual `tool_calls`/`ToolMessage` pairing repair around the LangChain loop.

### Landmines (repo is NOT self-contained — will not run as-is)

- **`utils/config_loader.py` is missing.** `agents/llm_factory.py:6` does
  `from utils.config_loader import load_llm_config`; `utils/` doesn't exist and
  is `.gitignore`d. This import runs at startup → **hard crash on every path.**
- **`LongformerSemanticSplitter` is required for ALL paths**, not just SEMANTIC:
  `demo_full_pipeline.py:89` instantiates it unconditionally for the Refiner.
  Its default `model_path="/root/liushifu/TopoChunker/models/longformer"`
  (`processing/semantic_processor.py:24`) is **another user's home dir**; the
  model isn't vendored (`models/` is gitignored). Loads onto CUDA.
- **VISUAL_TOPOLOGY needs MinerU** at `utils/MinerU/` (`processing/visual_processor.py:7`)
  — not vendored → `do_parse=None` → returns an error string.
- Missing per README: `requirements.txt`, `config/` + `config/llm_config.example.json`,
  `Evaluation/`. `.gitignore` excludes `*.txt *.json utils/ demo/ models/ Evaluation/ figure/`
  — this is *why* they're absent; they were intentionally not committed.
- `=0.1.0` — empty junk file at repo root (botched `pip/uv` invocation), committed.
- Dead code: `core/pruner.py`, `processing/llm_semantic_chunker.py` — imported nowhere.

### Only fully-local-capable path

`StructuralRuleProcessor` is pure PyMuPDF+regex (no model, no API). But end-to-end
still needs the LLM (both agents) and the Longformer (Refiner's slicer).
