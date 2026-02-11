import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from .base import BaseTopologyProcessor

class StructuralRuleProcessor(BaseTopologyProcessor):
    """
    Path 1: Structural Rule Processor.
    Uses dynamic hierarchy mapping to ensure correct Markdown nesting.
    """
    
    def __init__(self):
        # 1. 定义所有可能的结构模式（按逻辑优先级排序）
        # 增加了对英文 Chapter, Article, Section 的支持
        self.hierarchy_defs = [
            ("CHAPTER", r"^\s*(?:第[一二三四五六七八九十百]+章|Chapter\s+\d+)"),
            ("SECTION", r"^\s*(?:第[一二三四五六七八九十百]+节|Section\s+\d+)"),
            ("ARTICLE", r"^\s*(?:第[一二三四五六七八九十百]+条|Article\s+\d+)"),
            ("SUB_CHINESE", r"^\s*[（(][一二三四五六七八九十百]+[）)]"),
            ("NUMERIC_DOT", r"^\s*\d+[\.、\s]"),
            ("NUMERIC_PAREN", r"^\s*[（(]\d+[）)]"),
            ("NESTED_DOT", r"^\s*\d+(?:\.\d+)+"),
        ]
        # 扁平化的模式列表，用于快速匹配
        self.all_patterns = [p for _, p in self.hierarchy_defs]

    def _get_active_hierarchy(self, lines: List[str]) -> Dict[str, int]:
        """
        预扫描：检测文档中实际出现了哪些层级，并按优先级分配 Markdown Level (1-N)。
        """
        detected_types = []
        for type_name, pattern in self.hierarchy_defs:
            # 使用 IGNORECASE 提高鲁棒性
            if any(re.match(pattern, line, re.IGNORECASE) for line in lines):
                detected_types.append(type_name)
        
        # 建立映射：例如 {'CHAPTER': 1, 'ARTICLE': 2, 'NUMERIC_DOT': 3}
        mapping = {type_name: i + 1 for i, type_name in enumerate(detected_types)}
        return mapping

    def process(self, file_path: str) -> str:
        pdf_name = Path(file_path).stem
        output_dir = Path("outputs") / pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 提取并清洗文本行
        raw_lines = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    # 针对英文 PDF，使用更加结构化的文本提取方式
                    text = page.get_text("text")
                    
                    cleaned_text = ""
                    for line in text.splitlines():
                        if not line.strip(): continue
                        
                        # 清理异常空格（针对某些 PDF 提取出“每 个 字 带 空 格”的情况）
                        if line.count(' ') > len(line) / 2 and len(line) > 5:
                            line = line.replace(' ', '')
                        cleaned_text += line + "\n"
                    
                    page_lines = [l.strip() for l in cleaned_text.splitlines() if l.strip()]
                    raw_lines.extend(page_lines)
        except Exception as e:
            return f"Error: {e}"

        # 智能合并行内换行
        document_lines = []
        if raw_lines:
            current_line = raw_lines[0]
            for i in range(1, len(raw_lines)):
                next_line = raw_lines[i]
                
                # 检查下一行是否是新标题
                is_next_header = any(re.match(p, next_line, re.IGNORECASE) for p in self.all_patterns)
                
                # 如果当前行以标点结尾，或者是下一行是标题，则停止合并
                # 针对英文增加了 .!?: 等标点的支持
                if re.search(r'[。！？；.!?;:：]$', current_line) or is_next_header:
                    document_lines.append(current_line)
                    current_line = next_line
                else:
                    # 否则合并（针对跨行文本）
                    # 如果是英文，合并时需要加个空格
                    if re.search(r'[a-zA-Z0-9]$', current_line) and re.match(r'[a-zA-Z0-9]', next_line):
                        current_line += " " + next_line
                    else:
                        current_line += next_line
            document_lines.append(current_line)

        # 2. 动态构建层级映射表
        hierarchy_mapping = self._get_active_hierarchy(document_lines)
        print(f"📊 Detected Hierarchy: {hierarchy_mapping}")

        # 3. 执行转换
        chunks = []
        current_chunk = []
        
        for line in document_lines:
            # 识别当前行属于哪个类型
            matched_type = None
            for type_name, pattern in self.hierarchy_defs:
                if re.match(pattern, line, re.IGNORECASE):
                    matched_type = type_name
                    break
            
            if matched_type and matched_type in hierarchy_mapping:
                # 这是一个标题
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                
                level = hierarchy_mapping[matched_type]
                prefix = "#" * level
                current_chunk = [f"{prefix} {line}"]
            else:
                # 这是一个普通正文行
                current_chunk.append(line)
                
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        final_markdown = "\n\n".join(chunks)
        
        # 保存结果
        with open(output_dir / f"{pdf_name}_final.md", "w", encoding="utf-8") as f:
            f.write(final_markdown)

        return final_markdown
