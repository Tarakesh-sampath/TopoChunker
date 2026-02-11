import argparse
import os
from agents.llm_factory import create_llm
from processing.semantic_processor import SemanticFlowProcessor

def main():
    parser = argparse.ArgumentParser(description="TopoChunker SemanticFlowProcessor Demo")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI instead of local model")
    args = parser.parse_args()

    # 1. Create LLM
    model_type = "openai" if args.openai else "local"
    # Note: Using small qwen3 local path as defined in your factory
    llm = create_llm(model_type=model_type)

    # 2. Initialize Processor
    processor = SemanticFlowProcessor(llm_client=llm)

    # 3. Run Processing
    print(f"\n🚀 Processing document: {args.pdf_path}")
    markdown_output = processor.process(args.pdf_path)

    # 4. Show Result
    print("\n" + "="*50)
    print("FINAL STRUCTURED MARKDOWN")
    print("="*50 + "\n")
    print(markdown_output)
    
    # Save to file
    output_file = "semantic_output.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"\n💾 Result saved to {output_file}")

if __name__ == "__main__":
    main()
