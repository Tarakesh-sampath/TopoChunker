"""Demo for LangGraph-based InspectorAgent.

Usage:
    # Use local model (default)
    python demo_inspector.py <path_to_pdf>
    
    # Use OpenAI API
    OPENAI_API_KEY=your-key python demo_inspector.py <path_to_pdf> --openai

    # Specify custom model path
    python demo_inspector.py <path_to_pdf> --model-path /path/to/model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agents.inspector import InspectorAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoChunker InspectorAgent Demo")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API instead of local model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/liushifu/TopoChunker/models/Qwen3-8B",
        help="Path to local model directory (default: models/qwen3)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (if using OpenAI)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum ReAct loop iterations",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for local model (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output files (default: outputs)",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Determine model type
    if args.openai:
        model_type = "openai"
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Error: OPENAI_API_KEY not set.")
            print("Set it with: export OPENAI_API_KEY='your-key'")
            sys.exit(1)
    else:
        model_type = "local"

    print(f"\n🤖 LangGraph InspectorAgent")
    print(f"📄 Analyzing: {pdf_path}")
    print(f"🔧 Model: {model_type.upper()}")
    if model_type == "local":
        model_path = args.model_path or "models/qwen3"
        print(f"📦 Model path: {model_path}")
    else:
        print(f"🌐 OpenAI model: {args.model_name}")
    print("=" * 70)

    # Initialize agent
    agent = InspectorAgent(
        model_type=model_type,
        model_path=args.model_path,
        model_name=args.model_name,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        device_map=args.device
    )

    # Run inspection
    blueprint = agent.inspect(str(pdf_path))

    # Generate output filename based on input PDF
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_stem = pdf_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output_file = output_dir / f"{pdf_stem}_blueprint_{timestamp}.json"
    reasoning_output_file = output_dir / f"{pdf_stem}_reasoning_{timestamp}.txt"

    # Save Blueprint as JSON
    blueprint_dict = blueprint.model_dump()
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(blueprint_dict, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Blueprint saved to: {json_output_file}")

    # Save reasoning chain to text file
    if isinstance(blueprint.execution_plan, dict):
        reasoning_chain = blueprint.execution_plan.get("reasoning_chain")
        if reasoning_chain:
            with open(reasoning_output_file, "w", encoding="utf-8") as f:
                f.write(f"Document: {pdf_path}\n")
                f.write(f"Analysis Time: {timestamp}\n")
                f.write("=" * 70 + "\n\n")
                f.write("REASONING CHAIN\n")
                f.write("=" * 70 + "\n\n")
                for i, thought in enumerate(reasoning_chain, 1):
                    f.write(f"Step {i}:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"{thought}\n\n")
                f.write("=" * 70 + "\n\n")
                f.write("FINAL DECISION\n")
                f.write("=" * 70 + "\n")
                f.write(f"Selected Path: {blueprint.selected_path.value}\n")
                f.write(f"Confidence: {blueprint.confidence_score:.2f}\n")
                f.write(f"Risk Score: {blueprint.risk_assessment.score:.2f}\n")
                f.write(f"Reasoning: {blueprint.risk_assessment.reason}\n")
            print(f"💾 Reasoning chain saved to: {reasoning_output_file}")

    # Display results to console
    print("\n✅ Blueprint Generated:")
    print(json.dumps(blueprint_dict, ensure_ascii=False, indent=2))

    # Show reasoning chain if available
    if isinstance(blueprint.execution_plan, dict):
        reasoning_chain = blueprint.execution_plan.get("reasoning_chain")
        if reasoning_chain:
            print("\n🧠 Reasoning Chain:")
            for i, thought in enumerate(reasoning_chain, 1):
                print(f"\n  Step {i}:")
                print(f"  {thought}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
