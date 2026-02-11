from typing import List, Dict
from core.sir import SIRNode

class TopologicalPruner:
    """
    RMSP (Recursive Markdown Semantic Pruning) Algorithm.
    Uses token budgets to deterministically split the SIR tree into raw chunks.
    """
    def __init__(self, target_tokens: int = 800, max_tokens: int = 1200):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    def prune(self, node: SIRNode) -> List[List[SIRNode]]:
        """
        Recursively groups nodes into 'raw chunks'.
        Each chunk is a list of nodes that fit the token budget.
        """
        raw_chunks = []
        current_batch = []
        current_tokens = 0

        def traverse(curr: SIRNode):
            nonlocal current_tokens, current_batch
            
            # Case 1: The entire subtree fits perfectly
            if curr.total_tree_token <= self.max_tokens:
                if current_tokens + curr.total_tree_token > self.max_tokens and current_batch:
                    raw_chunks.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                # Add the whole branch as a unit
                current_batch.append(curr)
                current_tokens += curr.total_tree_token
                return

            # Case 2: Node itself is a heading, but subtree is too big -> Drill down
            if curr.node_type == 'heading':
                # Add the heading itself first (to maintain context)
                if current_tokens + curr.token_count > self.max_tokens and current_batch:
                    raw_chunks.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                current_batch.append(curr)
                current_tokens += curr.token_count
                
                # Then process children
                for child in curr.children:
                    traverse(child)
            
            # Case 3: Atomic or large text node -> Must include but it might exceed limit
            else:
                if current_tokens + curr.token_count > self.max_tokens and current_batch:
                    raw_chunks.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                current_batch.append(curr)
                current_tokens += curr.token_count

        traverse(node)
        if current_batch:
            raw_chunks.append(current_batch)
            
        return raw_chunks

    @staticmethod
    def nodes_to_text(nodes: List[SIRNode]) -> str:
        """Converts a group of nodes back to text, maintaining path context."""
        if not nodes: return ""
        # Use the first node's path as a breadcrumb
        breadcrumb = f"Context: {nodes[0].path_str}\n\n"
        return breadcrumb + "\n".join([n.content for n in nodes])
