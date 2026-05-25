# TopoChunker

TopoChunker is a multi-stage PDF-to-knowledge pipeline for producing topology-aware, retrieval-friendly chunks.

## Overview

The pipeline has four phases:

1. **Perception (Inspector Agent)**  
   Analyze PDF characteristics (scan state, layout complexity, structural regularity) and choose a routing path.
2. **Execution (Processor Routing)**  
   Route the document to one of:
   - `STRUCTURAL_RULE`
   - `SEMANTIC_FLOW`
   - `VISUAL_TOPOLOGY`
3. **Modeling (SIR Construction)**  
   Build a hierarchical SIR graph from generated markdown.
4. **Assembly (Refiner Agent)**  
   Apply the refinement toolkit:
   - `SIR_Query`
   - `Semantic_Slicer`
   - `Chunk_Enhancer`
   to produce final JSONL chunks for RAG.

## Repository Structure

Key directories/files:

- `demo_full_pipeline.py`: End-to-end entrypoint
- `agents/`: Inspector + Refiner agents and tools
- `processing/`: Structural / semantic / visual processing modules
- `core/`: Core data models (`SIR`, `Blueprint`, routing types)
- `config/llm_config.example.json`: Local API config example
- `requirements.txt`: Python dependencies
- `outputs/`: Runtime outputs (created automatically)
- `Evaluation/`: Evaluation scripts and experiment artifacts

## Requirements

- Python 3.10+ (recommended)
- API key for at least one provider:
  - DeepSeek (`DEEPSEEK_API_KEY`) or
  - OpenAI (`OPENAI_API_KEY`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

You can configure credentials in either way:

1. **Environment variables** (recommended for quick run)
2. **Local config file** `config/llm_config.json` (copy from `config/llm_config.example.json`)

Example env setup:

```bash
export DEEPSEEK_API_KEY="your_key"
# optional
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

## Quick Start

Run the full pipeline on a PDF:

```bash
python demo_full_pipeline.py "/absolute/path/to/your.pdf"
```

Optional arguments:

```bash
python demo_full_pipeline.py "/absolute/path/to/your.pdf" \
  --provider deepseek \
  --model deepseek-chat \
  --temp 0.1
```

Supported providers:
- `--provider deepseek` (default)
- `--provider openai`

## Outputs

For input `example.pdf`, outputs are written to:

`outputs/example/`

Main artifacts:

- `final_knowledge_base.jsonl` (final chunked corpus for retrieval)
- `topology_sir.json` (serialized SIR tree)
- `topology_graph.mmd` (Mermaid graph for visualization)

## Notes

- Large nodes are refined with `Semantic_Slicer` and `Chunk_Enhancer`.
- Context disambiguation relies on `SIR_Query` lineage + sibling evidence.
- The pipeline logs each phase in stdout for traceability.

## Troubleshooting

- **File not found**: use an absolute path for `pdf_path`.
- **API key error**: ensure provider key is available in env/config.
- **PyMuPDF error**: reinstall `pymupdf` from `requirements.txt`.
- **Model loading issues (semantic splitter)**: verify local model files/dependencies under `processing/semantic_processor.py`.

## License

Add your license information here.
