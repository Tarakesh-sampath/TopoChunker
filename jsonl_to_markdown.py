import json
import os
import re
from pathlib import Path

def convert_context_to_markdown(context_text):
    """
    将原始 context 文本转化为带有层级的 Markdown。
    针对 Passage X: 和 常见标题进行增强。
    """
    # 1. 处理 "Passage 1:" 这种作为一级标题
    text = re.sub(r'^(Passage \d+:)', r'# \1', context_text, flags=re.MULTILINE)
    
    # 2. 处理常见的段落标题（如 Biography, Early life 等）作为二级标题
    headers = ["Biography", "Early life", "Career", "Death", "Legacy", "References", "Background"]
    for h in headers:
        text = re.sub(f'^({h})$', r'## \1', text, flags=re.MULTILINE)
    
    return text

def jsonl_to_markdowns(input_jsonl, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            obj_id = data.get("_id", f"idx_{count}")
            context = data.get("context", "")
            
            md_content = convert_context_to_markdown(context)
            
            # 保存 md 文件
            md_path = os.path.join(output_dir, f"{obj_id}.md")
            with open(md_path, 'w', encoding='utf-8') as f_md:
                f_md.write(md_content)
            
            count += 1
            
    print(f"✅ Successfully converted {count} lines to .md files in: {output_dir}")

if __name__ == "__main__":
    INPUT_JSONL = "/root/liushifu/TopoChunker/test.jsonl"
    OUTPUT_MD_DIR = "/root/liushifu/TopoChunker/outputs/jsonl_markdowns"
    
    jsonl_to_markdowns(INPUT_JSONL, OUTPUT_MD_DIR)
