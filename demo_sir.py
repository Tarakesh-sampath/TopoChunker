from core.sir import SIRBuilder
import json
from pathlib import Path

def main():
    markdown_text = """
# 规章制度
## 第1章 总则
本规定旨在加强安全管理。
## 第2章 职责
### 2.1 销售部
负责销售安全。
| 部门 | 职责 |
| --- | --- |
| 销售部 | 客户开发 |
### 2.2 技术部
负责技术支持。
```python
print("Hello Safety")
```
"""

    print("🚀 Starting SIR Construction Demo...")
    
    # 1. Initialize Builder with Document Metadata
    builder = SIRBuilder(doc_name="path1.pdf")
    
    # 2. Build SIR Graph
    sir_root = builder.build(markdown_text)
    
    # 3. Print Tree Structure
    print("\n🌳 SIR Tree Structure:")
    def print_node(node, indent=0):
        print("  " * indent + str(node))
        for child in node.children:
            print_node(child, indent + 1)
            
    print_node(sir_root)
    
    # 4. Check specific node attributes
    print("\n🔍 Detail Check:")
    for child in sir_root.children:
        print(f"\nTop-level: {child.content}")
        print(f"Path: {child.path_str}")
        print(f"Tree Tokens: {child.total_tree_token}")
        for subchild in child.children:
            print(f"  - Sub: {subchild.content} (Type: {subchild.node_type}, Atomic: {subchild.is_atomic})")
            if subchild.children:
                for leaf in subchild.children:
                    print(f"    - Leaf: {leaf.content[:20]}.. (Path: {leaf.path_str})")

    # 5. Export to JSON & Mermaid
    output_dir = Path("outputs") / "demo_sir"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_dir / "sir_demo.json", "w", encoding="utf-8") as f:
        json.dump(sir_root.to_dict(), f, ensure_ascii=False, indent=2)
        
    # Save Mermaid
    with open(output_dir / "sir_graph.mmd", "w", encoding="utf-8") as f:
        f.write(sir_root.to_mermaid())
        
    print(f"\n💾 SIR Graph exported to {output_dir}/sir_demo.json")
    print(f"🎨 Visual Graph (Mermaid) saved to {output_dir}/sir_graph.mmd")

if __name__ == "__main__":
    main()
