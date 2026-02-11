import json
from typing import Any, Dict, List
from langchain_core.tools import tool
from core.sir import SIRNode

def get_refiner_tools(node_registry: Dict[str, SIRNode], splitter_func):
    """
    Factory function to create tools for the Refiner Agent.
    Captures registry and splitter in closures to avoid 'self' binding issues.
    """

    @tool
    def analyze_node_meta(node_id: str) -> Dict[str, Any]:
        """获取特定节点的统计信息（Token数、子节点数量、层级等）。"""
        node = node_registry.get(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "level": node.level,
            "own_tokens": node.token_count,
            "total_subtree_tokens": node.total_tree_token,
            "child_count": len(node.children),
            "is_atomic": node.is_atomic,
            "path": node.path_str
        }

    @tool
    def fetch_children(node_id: str) -> List[Dict[str, Any]]:
        """获取直接子节点列表及其完整内容（用于按标题拆分章节）。"""
        node = node_registry.get(node_id)
        if not node:
            return []
        return [
            {
                "node_id": child.node_id,
                "node_type": child.node_type,
                "level": child.level,
                "content": child.content,
                "tokens": child.total_tree_token
            }
            for child in node.children
        ]

    @tool
    def semantic_split_content(text: str) -> List[str]:
        """使用 LLM 语义切分器，将长文本切分为语义完整的段落。
        
        该方法使用滑动窗口和 LLM 判决的方式，识别场景转换、话题变更或时间跳跃等逻辑断点。
        """
        try:
            return splitter_func(text)
        except Exception as e:
            return [f"Split error: {str(e)}", text]

    @tool
    def preview_node_content(node_id: str) -> str:
        """获取该节点及其下属所有子节点的完整文本内容（用于阅读整个章节）。"""
        node = node_registry.get(node_id)
        if not node:
            return "Node not found"
        
        # 递归收集所有文本
        def collect_text(curr):
            texts = [curr.content]
            for child in curr.children:
                texts.append(collect_text(child))
            return "\n".join(texts)
            
        return collect_text(node)

    @tool
    def search_knowledge_in_sir(query: str) -> List[Dict[str, Any]]:
        """在整个文档 SIR 树中搜索关键词的定义或上下文。当发现代词（如"该数据"）指代不明时，使用此工具查找。"""
        results = []
        try:
            # 遍历整个注册表寻找匹配内容
            for _, node in node_registry.items():
                if query in node.content:
                    results.append({
                        "node_id": node.node_id,
                        "path": node.path_str,
                        "content": node.content[:300] # 返回足够长度供 Agent 判断
                    })
            return results[:5] 
        except Exception as e:
            return [{"error": str(e)}]

    @tool
    def get_previous_node_content(node_id: str, include_children: bool = False) -> Dict[str, Any]:
        """获取当前节点的前一个兄弟节点的内容。
        
        当需要理解上下文或解决代词指代问题时，可以使用此工具获取前一个chunk的内容。
        
        Args:
            node_id: 当前节点的ID
            include_children: 是否包含前一个节点的所有子节点内容（默认False，只返回节点本身内容）
        
        Returns:
            包含前一个节点信息的字典，如果不存在则返回错误信息
        """
        node = node_registry.get(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}
        
        if not node.prev_sibling:
            return {
                "error": "No previous sibling node found",
                "current_node_id": node_id,
                "is_first_sibling": True
            }
        
        prev_node = node.prev_sibling
        
        # 如果包含子节点，递归收集所有内容（添加深度限制防止栈溢出）
        if include_children:
            def collect_full_text(curr, max_depth=50, current_depth=0):
                if current_depth >= max_depth:
                    return "[... content truncated due to depth limit ...]"
                texts = [curr.content]
                for child in curr.children:
                    texts.append(collect_full_text(child, max_depth, current_depth + 1))
                return "\n".join(texts)
            content = collect_full_text(prev_node)
        else:
            content = prev_node.content
        
        return {
            "previous_node_id": prev_node.node_id,
            "node_type": prev_node.node_type,
            "level": prev_node.level,
            "path": prev_node.path_str,
            "content": content,
            "token_count": prev_node.token_count,
            "total_subtree_tokens": prev_node.total_tree_token,
            "has_children": len(prev_node.children) > 0
        }

    @tool
    def get_previous_nodes(node_id: str, count: int = 1) -> List[Dict[str, Any]]:
        """获取当前节点的前N个兄弟节点的内容。
        
        当需要查看多个前序chunk的上下文时，可以使用此工具。
        
        Args:
            node_id: 当前节点的ID
            count: 要获取的前序节点数量（默认1，最大建议不超过5）
        
        Returns:
            前序节点信息列表，按从近到远的顺序排列
        """
        node = node_registry.get(node_id)
        if not node:
            return [{"error": f"Node {node_id} not found"}]
        
        # 参数验证：确保count是正整数
        if count < 1:
            count = 1
        # 限制最大数量，避免返回过多内容
        count = min(count, 10)
        
        results = []
        current = node.prev_sibling
        retrieved = 0
        visited_ids = set()  # 防止循环引用
        
        while current and retrieved < count:
            # 循环检测：如果遇到已访问的节点，说明存在循环，停止遍历
            if current.node_id in visited_ids:
                results.append({
                    "error": "Circular reference detected in sibling chain",
                    "node_id": current.node_id
                })
                break
            
            visited_ids.add(current.node_id)
            results.append({
                "node_id": current.node_id,
                "node_type": current.node_type,
                "level": current.level,
                "path": current.path_str,
                "content": current.content,
                "token_count": current.token_count,
                "total_subtree_tokens": current.total_tree_token
            })
            current = current.prev_sibling
            retrieved += 1
        
        if not results:
            return [{
                "error": "No previous sibling nodes found",
                "current_node_id": node_id
            }]
        
        return results

    return [
        analyze_node_meta, 
        fetch_children, 
        semantic_split_content, 
        preview_node_content, 
        search_knowledge_in_sir,
        get_previous_node_content,
        get_previous_nodes
    ]
