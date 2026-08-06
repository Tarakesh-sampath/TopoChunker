# TopoChunker

[![arXiv](https://img.shields.io/badge/arXiv-2603.18409-B31B1B.svg)](https://arxiv.org/abs/2603.18409)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Topology-aware agentic document chunking for retrieval-augmented generation (RAG).

TopoChunker routes each PDF by layout complexity, maps it onto a Structured Intermediate Representation (SIR), and assembles self-contained, retrieval-ready chunks.

## Pipeline

```
PDF → Inspector → Processor → SIR Graph → Refiner → JSONL Chunks
```

| Stage | Component | Role |
|-------|-----------|------|
| Perception | Inspector Agent | Assess layout / scan quality and select a routing path |
| Execution | Processor | `STRUCTURAL_RULE` · `SEMANTIC_FLOW` · `VISUAL_TOPOLOGY` |
| Modeling | SIR Builder | Construct a hierarchical topology graph from markdown |
| Assembly | Refiner Agent | Slice, enrich, and emit final chunks (`SIR_Query`, `Semantic_Slicer`, `Chunk_Enhancer`) |

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+** and an API key for DeepSeek or OpenAI.

## Local Models

The `SEMANTIC_FLOW` route loads a fine-tuned Longformer segmenter for semantic boundary detection.

| Item | Value |
|------|-------|
| Hugging Face | [`Theoreticallyhugo/longformer-sep_tok_full_labels`](https://huggingface.co/Theoreticallyhugo/longformer-sep_tok_full_labels) (`checkpoint-3159`) |
| Local path | `models/longformer` |

Download:

```bash
# huggingface-cli
huggingface-cli download Theoreticallyhugo/longformer-sep_tok_full_labels 
```

## Configuration

Set credentials via environment variables (recommended):

```bash
export DEEPSEEK_API_KEY="your_key"
# optional
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Or copy `config/llm_config.example.json` to `config/llm_config.json` and fill in your keys.

## Usage

```bash
python demo_full_pipeline.py /path/to/document.pdf
```

Optional arguments:

```bash
python demo_full_pipeline.py /path/to/document.pdf \
  --provider deepseek \
  --model deepseek-chat \
  --temp 0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `deepseek` | `deepseek` or `openai` |
| `--model` | `deepseek-chat` | Model name |
| `--temp` | `0.1` | Sampling temperature |

## Outputs

Results are written to `outputs/<pdf_stem>/`:

| File | Description |
|------|-------------|
| `final_knowledge_base.jsonl` | Final retrieval chunks |
| `topology_sir.json` | Serialized SIR tree |
| `topology_graph.mmd` | Mermaid visualization of the topology |

## Project Layout

```
TopoChunker/
├── demo_full_pipeline.py   # End-to-end entry point
├── agents/                 # Inspector & Refiner agents
├── processing/             # Structural / semantic / visual processors
├── core/                   # SIR models and routing types
├── config/                 # LLM config template
├── Evaluation/             # Evaluation scripts
└── requirements.txt
```

## Citation

```bibtex
@article{liu2026topochunker,
  title   = {TopoChunker: Topology-Aware Agentic Document Chunking Framework},
  author  = {Liu, Xiaoyu},
  journal = {arXiv preprint arXiv:2603.18409},
  year    = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
