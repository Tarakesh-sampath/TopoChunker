import argparse
import sys
import datetime
import json
import uuid
from pathlib import Path

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.inspector import InspectorAgent
from agents.llm_factory import create_llm
from core.types import RoutingPath
from core.sir import SIRBuilder, SIRNode
from agents.refiner import RefinerAgent
from processing.semantic_processor import LongformerSemanticSplitter
from processing.structural_processor import StructuralRuleProcessor
from processing.semantic_processor import SemanticFlowProcessor
from processing.visual_processor import VisualTopologyProcessor

def main() -> None:
    parser = argparse.ArgumentParser(description="TopoChunker API-Only Pipeline")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name (deepseek-chat or gpt-4o-mini)")
    parser.add_argument("--provider", type=str, choices=["deepseek", "openai"], default="deepseek", help="API Provider")
    parser.add_argument("--temp", type=float, default=0.1, help="LLM Temperature")
    
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'='*70}\n🚀 TopoChunker API Pipeline Starting for: {pdf_path.name}\n{'='*70}")

    # 1. Initialize shared LLM
    llm_instance = create_llm(model_type=args.provider, model_name=args.model, temperature=args.temp)

    # 2. Phase 1: Perception (Inspector Agent)
    print(f"\n🧐 [Phase 1] Perception: Running Inspector Agent...")
    inspector = InspectorAgent(llm=llm_instance)
    blueprint = inspector.inspect(str(pdf_path))

    print(f"\n✅ Inspector Decision: {blueprint.selected_path.value}")
    print(f"   Reason: {blueprint.risk_assessment.reason}")

    # 3. Phase 2: Execution (Markdown Generation)
    print(f"\n🛠️ [Phase 2] Execution: Routing to {blueprint.selected_path.value}...")
    
    if blueprint.selected_path == RoutingPath.STRUCTURAL_RULE:
        processor = StructuralRuleProcessor()
    elif blueprint.selected_path == RoutingPath.SEMANTIC_FLOW:
        processor = SemanticFlowProcessor(llm_client=llm_instance)
    else:
        processor = VisualTopologyProcessor()

    final_markdown = processor.process(str(pdf_path))

    # 4. Phase 3: Modeling (SIR Construction)
    print(f"\n🌳 [Phase 3] Modeling: Building SIR Graph...")
    sir_builder = SIRBuilder(doc_name=pdf_path.name)
    sir_root = sir_builder.build(final_markdown)
    
    # Save SIR artifacts
    output_folder = Path("outputs") / pdf_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)
    
    with open(output_folder / "topology_sir.json", "w", encoding="utf-8") as f:
        json.dump(sir_root.to_dict(), f, ensure_ascii=False, indent=2)
    
    with open(output_folder / "topology_graph.mmd", "w", encoding="utf-8") as f:
        f.write(sir_root.to_mermaid())
    print(f"💾 SIR Graph saved to {output_folder}/topology_sir.json")

    # 5. Phase 4: Assembly Line (Refiner Agent)
    print(f"\n🏭 [Phase 4] Assembly Line: Generating Final Knowledge Base...")
    
    # Register all nodes for Tool lookup
    node_registry = {}
    def register(n):
        node_registry[n.node_id] = n
        for child in n.children: register(child)
    register(sir_root)

    # Initialize Refiner with the core logic we restored
    refiner = RefinerAgent(
        llm=llm_instance, 
        splitter_func=LongformerSemanticSplitter().split,
        node_registry=node_registry
    )
    
    final_jsonl_path = output_folder / "final_knowledge_base.jsonl"
    all_chunks_count = 0

    with open(final_jsonl_path, "w", encoding="utf-8") as f_out:
        # 调度策略：收集顶级 H1 章节
        tasks = [child for child in sir_root.children if child.level == 1]
        
        # 如果没有 H1，就把 Root 作为一个任务
        if not tasks:
            tasks = [sir_root]

        for i, task_node in enumerate(tasks):
            print(f"   🚀 Processing Section {i+1}/{len(tasks)}: {task_node.content[:30]}...")
            
            # Agent 进行智能决策和切分
            refined_chunks = refiner.refine(task_node)
            
            # 将切出来的片段（如果是列表）进行封装
            for chunk_text in refined_chunks:
                # 【核心修改】使用括号包裹路径信息，使其与正文区分更明显
                context_prefix = f"【文档层级：{task_node.path_str}】\n\n"
                full_content = context_prefix + chunk_text
                
                chunk_obj = {
                    "id": str(uuid.uuid4()),
                    "text": full_content,  # 这里将 path 注入正文
                    "metadata": {
                        "source": pdf_path.name,
                        "path": task_node.path_str,
                        "token_count": len(full_content),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                }
                f_out.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                all_chunks_count += 1

    print(f"\n{'='*70}\n🎉 Finished! Total {all_chunks_count} chunks generated.\nSaved in: {final_jsonl_path}\n{'='*70}")

if __name__ == "__main__":
    main()
