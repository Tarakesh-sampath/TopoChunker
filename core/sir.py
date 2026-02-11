import uuid
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class SIRNode:
    """
    Structured Intermediate Representation (SIR) Node.
    A rich object representing a document element in a topological graph.
    """
    # --- Basic Attributes ---
    node_id: str              # Unique ID (UUID)
    node_type: str            # 'root', 'heading', 'text', 'table', 'code', 'image'
    content: str              # Original content of the node
    level: int                # Heading level (H1=1, H2=2, Text=99)
    
    # --- Topological Relations (Graph Edges) ---
    parent: Optional['SIRNode'] = field(default=None, repr=False)
    children: List['SIRNode'] = field(default_factory=list, repr=False)
    prev_sibling: Optional['SIRNode'] = field(default=None, repr=False)
    next_sibling: Optional['SIRNode'] = field(default=None, repr=False)
    
    # --- Enhanced Attributes ---
    path_str: str = ""        # Breadcrumb path: "path1.pdf > Chapter 1 > Section 1.1"
    token_count: int = 0      # Node's own token count
    total_tree_token: int = 0 # sum(tokens of self + descendants)
    is_atomic: bool = False   # True if node shouldn't be split (table/image/code)
    metadata: Dict[str, Any] = field(default_factory=dict) # [New] Extra metadata (doc_name, page, etc.)
    
    def __repr__(self):
        content_preview = (self.content[:30] + "..") if len(self.content) > 30 else self.content
        return f"<{self.node_type.upper()} L{self.level} id={self.node_id[:8]} tokens={self.token_count} total={self.total_tree_token}> {content_preview}"

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Convert node to a serializable dictionary."""
        d = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": self.content,
            "level": self.level,
            "path_str": self.path_str,
            "token_count": self.token_count,
            "total_tree_token": self.total_tree_token,
            "is_atomic": self.is_atomic,
            "metadata": self.metadata,
            "parent_id": self.parent.node_id if self.parent else None,
        }
        if include_children:
            d["children"] = [child.to_dict() for child in self.children]
        return d

    def to_mermaid(self) -> str:
        """Generate Mermaid.js diagram code for the graph."""
        lines = ["graph TD"]
        
        def traverse(node):
            # Escaping quotes for safety
            safe_content = node.content.replace('"', "'")[:20]
            node_label = f'{node.node_id[:8]}["[{node.node_type.upper()}] {safe_content}"]'
            lines.append(f"    {node_label}")
            
            for child in node.children:
                lines.append(f"    {node.node_id[:8]} --> {child.node_id[:8]}")
                traverse(child)
                
        traverse(self)
        return "\n".join(lines)

class SIRBuilder:
    """
    Builds a SIR graph from Markdown text using a Stack-Based State Machine.
    """
    def __init__(self, tokenizer=None, doc_name: str = "Unknown", doc_id: str = ""):
        """
        Args:
            tokenizer: An object with an .encode() method.
            doc_name: Filename of the source document.
            doc_id: Unique identifier for the document.
        """
        self.tokenizer = tokenizer
        self.doc_name = doc_name
        self.doc_id = doc_id or str(uuid.uuid4())
        
        # Initialize root with metadata
        self.root = SIRNode(
            node_id="root", 
            node_type="root", 
            content="Document Root", 
            level=0,
            path_str=doc_name,
            metadata={"filename": doc_name, "doc_id": self.doc_id}
        )
        self.stack = [self.root] 
        self.last_node = None

    def build(self, markdown_text: str) -> SIRNode:
        """
        Parses Markdown text into a SIR tree.
        """
        # Split into logical lines/blocks
        lines = markdown_text.splitlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 1. Parsing type and level
            level, node_type, content = self._parse_markdown_line(line)

            # 2. Stack Operation: Find the correct parent
            # Backtrack until the top of the stack is a valid parent (lower level)
            while self.stack[-1].level >= level and len(self.stack) > 1:
                self.stack.pop()
            
            parent = self.stack[-1]

            # 3. Instantiate Node
            node = SIRNode(
                node_id=str(uuid.uuid4()),
                node_type=node_type,
                content=content,
                level=level,
                parent=parent,
                metadata={"filename": self.doc_name, "doc_id": self.doc_id}
            )
            
            # --- Calculation (Deterministic Decorator) ---
            node.token_count = self._count_tokens(content)
            node.is_atomic = self._check_atomicity(content, node_type)
            
            # Path inheritance: Always starts with filename
            if node_type == 'heading':
                node.path_str = f"{parent.path_str} > {content}"
            else:
                node.path_str = parent.path_str

            # 4. Linkage
            parent.children.append(node)
            
            # Sibling linkage
            if self.last_node:
                # Strictly speaking, siblings are nodes with the same parent
                # but we keep a linear sequence for context flow
                self.last_node.next_sibling = node
                node.prev_sibling = self.last_node
            
            self.last_node = node

            # 5. Push headings to stack as potential parents
            if node_type == 'heading':
                self.stack.append(node)
                
        # 6. Post-processing: Bottom-up token aggregation
        self._aggregate_tokens(self.root)
        
        return self.root

    def get_node_registry(self) -> Dict[str, SIRNode]:
        """
        Returns a flat dictionary mapping node_id to SIRNode for all nodes in the tree.
        """
        registry = {}
        
        def traverse(node):
            registry[node.node_id] = node
            for child in node.children:
                traverse(child)
                
        traverse(self.root)
        return registry

    def _parse_markdown_line(self, line: str):
        # 1. Heading check
        heading_match = re.match(r'^(#+)\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            return level, 'heading', heading_match.group(2)
        
        # 2. Table check
        if line.startswith('|') and line.endswith('|'):
            return 99, 'table', line
            
        # 3. Code block start/end (simplified)
        if line.startswith('```'):
            return 99, 'code', line
            
        # Default: text
        return 99, 'text', line

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                # Handle different tokenizer API styles
                if hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(text)
                    return len(tokens)
                elif callable(self.tokenizer):
                    return len(self.tokenizer(text))
            except:
                pass
        
        # Fallback: Simple heuristic (CJK characters + whitespace split)
        # Roughly 1 token per 1.5 chars for mixed text
        return len(text.split()) + len(re.findall(r'[\u4e00-\u9fff]', text))

    def _check_atomicity(self, content: str, node_type: str) -> bool:
        if node_type in ['table', 'code', 'image']:
            return True
        if re.search(r'!\[.*?\]\(.*?\)', content): # Image tag
            return True
        if len(content) < 50:
            return False
        return False

    def _aggregate_tokens(self, node: SIRNode) -> int:
        node.total_tree_token = node.token_count
        for child in node.children:
            node.total_tree_token += self._aggregate_tokens(child)
        return node.total_tree_token
