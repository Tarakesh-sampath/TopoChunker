import re
from typing import Any, Dict, List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from core.sir import SIRNode

CHUNK_ENHANCER_SYSTEM_PROMPT = """You are Chunk_Enhancer for semantic signature generation.

Given one raw segment, produce:
1) A concise thematic title with 3-8 words.
2) Optional Context Supplement only if needed to resolve ambiguous pronouns/references.

Rules:
- Keep title short, concrete, and searchable.
- Do NOT copy the first sentence as title.
- Do NOT include generic placeholders like "Document Root" or "Segment".
- Return strict JSON object only:
  {"title":"...", "context_supplement":"... or null"}
"""


def get_refiner_tools(node_registry: Dict[str, SIRNode], splitter_func, llm=None):
    """
    Factory function to create tools for the Refiner Agent.
    Captures registry and splitter in closures to avoid 'self' binding issues.
    """

    @tool("Semantic_Slicer")
    def semantic_slicer(text: str) -> List[str]:
        """使用 LLM 语义切分器，将长文本切分为语义完整的段落。
        
        该方法使用滑动窗口和 LLM 判决的方式，识别场景转换、话题变更或时间跳跃等逻辑断点。
        """
        try:
            return splitter_func(text)
        except Exception as e:
            return [f"Split error: {str(e)}", text]

    @tool("SIR_Query")
    def SIR_Query(
        node_id: str,
        include_children: bool = False,
        include_lineage: bool = True,
        include_entity_definitions: bool = True
    ) -> Dict[str, Any]:
        """统一的 SIR 拓扑查询工具：同级回溯 + parent 祖先链追溯（Topological Lineage）。
        
        典型用途：
        1) 获取前一个兄弟节点内容，解决跨 chunk 代词指代；
        2) 沿 parent 指针向上提取宏观标题路径（章节脉络）；
        3) 从祖先与前序兄弟中抽取疑似实体定义，便于补全语义上下文。
        
        Args:
            node_id: 当前节点 ID
            include_children: 是否展开前一个兄弟节点的子树内容
            include_lineage: 是否返回沿 parent 链的宏观标题
            include_entity_definitions: 是否返回疑似实体定义候选
        
        Returns:
            包含 previous sibling、lineage 与 definition candidates 的字典
        """
        node = node_registry.get(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        def collect_full_text(curr: SIRNode, max_depth: int = 50, current_depth: int = 0) -> str:
            if current_depth >= max_depth:
                return "[... content truncated due to depth limit ...]"
            texts = [curr.content]
            for child in curr.children:
                texts.append(collect_full_text(child, max_depth, current_depth + 1))
            return "\n".join(texts)

        def gather_lineage(curr: SIRNode) -> List[Dict[str, Any]]:
            lineage: List[Dict[str, Any]] = []
            p = curr.parent
            while p is not None:
                lineage.append({
                    "node_id": p.node_id,
                    "node_type": p.node_type,
                    "level": p.level,
                    "title": p.content if p.node_type == "heading" else "",
                    "path": p.path_str
                })
                p = p.parent
            # 从根到近端，更符合宏观脉络阅读顺序
            return list(reversed(lineage))

        def extract_definition_candidates(text: str, source: str, source_node: SIRNode) -> List[Dict[str, Any]]:
            candidates: List[Dict[str, Any]] = []
            if not text:
                return candidates

            patterns = [
                r"([A-Za-z][A-Za-z0-9_\-\s]{1,40})\s+(is|are|refers to|means)\s+(.{5,120})",
                r"([A-Za-z][A-Za-z0-9_\-\s]{1,40})\s*[:：]\s*(.{5,120})",
                r"(“[^”]{1,40}”|\"[^\"]{1,40}\")\s*(是|指|定义为)\s*(.{5,120})",
                r"(第[一二三四五六七八九十百]+[章节条].{0,30})"
            ]

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for line in lines[:30]:
                for p in patterns:
                    m = re.search(p, line, flags=re.IGNORECASE)
                    if m:
                        candidates.append({
                            "source": source,
                            "source_node_id": source_node.node_id,
                            "path": source_node.path_str,
                            "evidence": line[:220]
                        })
                        break
            return candidates

        # previous sibling context
        prev_info: Dict[str, Any]
        if not node.prev_sibling:
            prev_info = {
                "exists": False,
                "error": "No previous sibling node found",
                "is_first_sibling": True
            }
            prev_text = ""
            prev_node = None
        else:
            prev_node = node.prev_sibling
            prev_text = collect_full_text(prev_node) if include_children else prev_node.content
            prev_info = {
                "exists": True,
                "node_id": prev_node.node_id,
                "node_type": prev_node.node_type,
                "level": prev_node.level,
                "path": prev_node.path_str,
                "content": prev_text,
                "token_count": prev_node.token_count,
                "total_subtree_tokens": prev_node.total_tree_token,
                "has_children": len(prev_node.children) > 0
            }

        lineage = gather_lineage(node) if include_lineage else []

        definition_candidates: List[Dict[str, Any]] = []
        if include_entity_definitions:
            # 1) 前序兄弟的定义线索
            if prev_node is not None:
                definition_candidates.extend(
                    extract_definition_candidates(prev_text, "previous_sibling", prev_node)
                )
            # 2) 沿 parent 链的宏观定义线索（Topological Lineage）
            p = node.parent
            hops = 0
            while p is not None and hops < 8:
                definition_candidates.extend(
                    extract_definition_candidates(p.content, "lineage_parent", p)
                )
                p = p.parent
                hops += 1

        return {
            "query_node_id": node.node_id,
            "query_node_path": node.path_str,
            "previous_sibling": prev_info,
            "lineage": lineage,
            "definition_candidates": definition_candidates[:20]
        }

    @tool("Chunk_Enhancer")
    def chunk_enhancer(text: str, query: str = "") -> Dict[str, Any]:
        """为原始 chunk 生成语义签名：标题（3-8词）和可选上下文补充。"""
        if not text or not text.strip():
            return {"error": "Empty text input", "title": "", "context_supplement": None}

        if llm is None:
            # 无 LLM 时提供确定性降级，保证工具存在且可调用
            first_line = text.strip().splitlines()[0][:60]
            fallback_title = " ".join(first_line.split()[:8]).strip() or "Untitled Segment"
            return {
                "title": fallback_title,
                "context_supplement": None
            }

        try:
            user_prompt = (
                f"Query hint: {query}\n\n"
                "Raw segment:\n"
                f"{text}\n\n"
                "Return strict JSON object only."
            )
            resp = llm.invoke([
                SystemMessage(content=CHUNK_ENHANCER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            content = (resp.content or "").strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return {
                    "title": "Untitled Segment",
                    "context_supplement": None,
                    "warning": "Chunk_Enhancer returned non-JSON; fallback applied"
                }
            import json
            parsed = json.loads(match.group(0))
            title = str(parsed.get("title", "")).strip()
            if not title:
                title = "Untitled Segment"
            words = title.split()
            if len(words) > 8:
                title = " ".join(words[:8])
            if len(words) < 3:
                title = (title + " Key Topic Summary").strip()
            return {
                "title": title,
                "context_supplement": parsed.get("context_supplement")
            }
        except Exception as e:
            return {
                "title": "Untitled Segment",
                "context_supplement": None,
                "error": str(e)
            }

    return [
        SIR_Query,
        semantic_slicer,
        chunk_enhancer
    ]
