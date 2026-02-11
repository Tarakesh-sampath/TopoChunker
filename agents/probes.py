"""Perception probes for fast, low-cost document sensing."""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

# Lazy import for PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


class MetaReader:
    """Quick metadata and scanned-PDF check."""

    @staticmethod
    def analyze(file_path: str, verbose: bool = False) -> Dict[str, object]:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")
        try:
            if verbose:
                print(f"\n[工具调用] analyze_metadata(file_path='{file_path}')")
            
            with fitz.open(file_path) as doc:
                meta = doc.metadata or {}
                page_count = doc.page_count
                author = meta.get("author")
                
                if verbose:
                    print(f"  [元数据] 页数: {page_count}")
                    print(f"  [元数据] 作者: {author}")
                    print(f"  [元数据] 开始扫描检测...")
                
                is_scanned = MetaReader._is_scanned_pdf(doc, verbose=verbose)
                
                # Count visual elements (Images and Tables)
                total_images = 0
                total_tables = 0
                for page in doc:
                    # Count images
                    total_images += len(page.get_images())
                    # Count tables using PyMuPDF's built-in finder
                    tables = page.find_tables()
                    total_tables += len(tables.tables)
                
                result = {
                    "page_count": page_count,
                    "author": author,
                    "is_scanned": is_scanned,
                    "image_count": total_images,
                    "table_count": total_tables,
                    "metadata": meta,
                }
                
                if verbose:
                    print(f"  [结果] is_scanned = {is_scanned}")
                    print(f"  [工具返回] {result}\n")
                
                return result
        except Exception as exc:
            if verbose:
                print(f"  [错误] {exc}")
            return {"error": str(exc), "is_scanned": True}

    @staticmethod
    def _is_scanned_pdf(doc: fitz.Document, verbose: bool = False) -> bool:
        """Check if PDF is scanned (no text layer).
        
        Uses a more accurate approach:
        - Check multiple pages (not just first page)
        - Check text blocks existence (more reliable)
        - Use stricter thresholds to avoid false positives
        
        Args:
            doc: PyMuPDF document
            verbose: If True, print detailed logging
        """
        if doc.page_count == 0:
            if verbose:
                print("  [扫描检测] 文档页数为0，判定为扫描件")
            return True
        
        try:
            # Check multiple pages for better accuracy
            pages_to_check = min(3, doc.page_count)
            total_text_length = 0
            total_text_blocks = 0
            
            if verbose:
                print(f"  [扫描检测] 开始检查文档，总页数: {doc.page_count}，检查前 {pages_to_check} 页")
            
            page_details = []
            for page_idx in range(pages_to_check):
                page = doc.load_page(page_idx)
                
                # Method 1: Get plain text
                text = page.get_text("text").strip()
                text_length = len(text)
                total_text_length += text_length
                
                # Method 2: Check text blocks (more reliable)
                text_dict = page.get_text("dict")
                text_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]
                block_count = len(text_blocks)
                total_text_blocks += block_count
                
                page_details.append({
                    "page": page_idx + 1,
                    "text_length": text_length,
                    "block_count": block_count,
                })
                
                if verbose:
                    print(f"    - 第 {page_idx + 1} 页: 文本长度={text_length} 字符, 文本块={block_count} 个")
            
            # Calculate averages
            avg_text_length = total_text_length / pages_to_check
            avg_text_blocks = total_text_blocks / pages_to_check
            
            if verbose:
                print(f"  [扫描检测] 统计结果:")
                print(f"    - 平均文本长度: {avg_text_length:.1f} 字符/页")
                print(f"    - 平均文本块数: {avg_text_blocks:.1f} 个/页")
                print(f"    - 判定阈值: 文本长度 < 30 且 文本块 < 2")
            
            # Conservative check: Only mark as scanned if:
            # 1. Average text length is very low (< 30 chars per page)
            # 2. AND very few text blocks (< 2 blocks per page)
            # This prevents false positives for documents with sparse but valid text
            if avg_text_length < 30 and avg_text_blocks < 2:
                if verbose:
                    print(f"  [扫描检测] ✅ 判定为扫描件 (平均文本长度 {avg_text_length:.1f} < 30 且 文本块 {avg_text_blocks:.1f} < 2)")
                return True
            
            # Additional check: If no text blocks at all across checked pages, likely scanned
            if total_text_blocks == 0:
                if verbose:
                    print(f"  [扫描检测] ✅ 判定为扫描件 (未发现任何文本块)")
                return True
            
            # Otherwise, assume it has text layer (not scanned)
            if verbose:
                print(f"  [扫描检测] ❌ 判定为非扫描件 (有文本层)")
            return False
            
        except Exception as exc:
            # If check fails, assume not scanned (safer default)
            if verbose:
                print(f"  [扫描检测] ⚠️  检查异常: {exc}，默认判定为非扫描件")
            return False


class LayoutProbe:
    """Estimate layout complexity using sparse page sampling."""

    @staticmethod
    def probe(doc: fitz.Document) -> float:
        try:
            sample_pages = LayoutProbe._sample_middle_pages(doc.page_count, max_pages=3)
            if not sample_pages:
                return 0.0
            scores = []
            for page_index in sample_pages:
                page = doc.load_page(page_index)
                blocks = page.get_text("dict").get("blocks", [])
                scores.append(LayoutProbe._page_complexity(blocks))
            return sum(scores) / len(scores)
        except Exception:
            return 1.0

    @staticmethod
    def _sample_middle_pages(page_count: int, max_pages: int) -> List[int]:
        if page_count == 0:
            return []
        start = max(0, page_count // 3)
        end = min(page_count, (page_count * 2) // 3 + 1)
        candidates = list(range(start, end))
        if not candidates:
            candidates = list(range(page_count))
        k = min(max_pages, len(candidates))
        return sorted(random.sample(candidates, k=k))

    @staticmethod
    def _page_complexity(blocks: List[dict]) -> float:
        if not blocks:
            return 0.0
        x_coords = []
        small_blocks = 0
        for block in blocks:
            if block.get("type") != 0:
                continue
            x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
            width = max(0.0, x1 - x0)
            height = max(0.0, y1 - y0)
            area = width * height
            x_coords.append(x0)
            if area < 1500:
                small_blocks += 1
        if not x_coords:
            return 0.0
        unique_columns = len({round(x / 10) for x in x_coords})
        column_score = min(1.0, unique_columns / 6.0)
        density_score = min(1.0, small_blocks / max(1, len(x_coords)))
        return min(1.0, 0.6 * column_score + 0.4 * density_score)


class RegexTester:
    """Check rule adherence using structural regex patterns."""

    _PATTERNS = [
        re.compile(r"^\s*\d+[\.、\s]", re.IGNORECASE), # 1. or 1、 or 1 (with space)
        re.compile(r"^\s*\d+\.\d+", re.IGNORECASE),    # 1.1
        re.compile(r"^\s*[IVXLCDM]+\.", re.IGNORECASE), # Roman numerals I. II.
        re.compile(r"^\s*Chapter\s+\d+", re.IGNORECASE),
        re.compile(r"^\s*Article\s+\d+", re.IGNORECASE),
        re.compile(r"^\s*第[一二三四五六七八九十百]+[章节条]", re.IGNORECASE), # Chinese
        re.compile(r"^\s*[（(][一二三四五六七八九十百]+[）)]", re.IGNORECASE), # (一)
    ]

    @staticmethod
    def test_adherence(doc: fitz.Document, verbose: bool = False) -> float:
        try:
            if verbose:
                print(f"\n[工具调用] test_rule_adherence")
            
            max_pages = min(5, doc.page_count)
            if max_pages == 0:
                return 0.0
            
            total_lines = 0
            matches = 0
            header_samples = []
            
            for page_index in range(max_pages):
                page = doc.load_page(page_index)
                lines = page.get_text("text").splitlines()
                for line in lines:
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                    total_lines += 1
                    for pattern in RegexTester._PATTERNS:
                        if pattern.match(line_clean):
                            matches += 1
                            if len(header_samples) < 5:
                                header_samples.append(line_clean)
                            break
            
            if total_lines == 0:
                return 0.0
            
            # AGGRESSIVE SCORING:
            # If we find 3+ headers in the first 5 pages, it's structured.
            # If we find a sequence, even better.
            base_ratio = matches / total_lines
            
            if matches >= 3:
                # High confidence for structured documents
                adherence_score = max(base_ratio * 2, 0.85) 
            elif matches >= 1:
                adherence_score = max(base_ratio * 1.5, 0.4)
            else:
                adherence_score = base_ratio

            if verbose:
                print(f"  [规则测试] 匹配到 {matches} 个潜在标题。示例: {header_samples}")
                print(f"  [规则测试] 最终得分: {adherence_score:.3f}")
            
            return min(1.0, adherence_score)
        except Exception as exc:
            if verbose:
                print(f"  [规则测试] ⚠️  异常: {exc}，返回遵循度 0.0\n")
            return 0.0
