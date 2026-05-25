import json
import re
from typing import Annotated, Dict, List, Literal, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from core.sir import SIRNode
from .refiner_tools import get_refiner_tools

class RefinerState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    root_node_id: str
    final_chunks: List[str]
    iteration_count: int
    validation_errors: List[str]
    query: str  # Added query to state
    consecutive_tool_errors: int  # 连续 tool_calls 错误计数

class RefinerAgent:
    def __init__(self, llm: BaseChatModel, splitter_func, node_registry: Dict[str, SIRNode], verbose: bool = True):
        self.base_llm = llm
        self.splitter_func = splitter_func
        self.node_registry = node_registry
        self.verbose = verbose
        
        # --- 优化 A: 预初始化工具并绑定 LLM，避免每轮重复开销 ---
        self.tools = get_refiner_tools(self.node_registry, self.splitter_func, llm=self.base_llm)
        self.llm_with_tools = self.base_llm.bind_tools(self.tools)
        
        self.workflow = self._build_workflow()
    
    def _clean_message_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        清理消息历史，确保 tool_calls 和 ToolMessage 严格配对。
        
        核心算法：
        1. 扫描所有 ToolMessage，收集哪些 tool_call_id 有响应
        2. 对于每个有 tool_calls 的 AIMessage：
           - 如果所有 tool_call_id 都有响应 → 保留原样
           - 如果有任何 tool_call_id 没有响应 → 转换为纯文本 AIMessage（剥离 tool_calls）
        3. ToolMessage 只保留父 AIMessage 被保留的
        
        Returns:
            清理后的、tool_calls 配对正确的消息列表
        """
        # Step 1: 收集所有 ToolMessage 的 tool_call_id
        response_ids = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tid = getattr(msg, "tool_call_id", None)
                if tid:
                    response_ids.add(tid)
        
        # Step 2: 确定哪些 AIMessage 的 tool_calls 是完全匹配的
        fully_matched_tc_ids = set()
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                all_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                if all_ids and all_ids.issubset(response_ids):
                    fully_matched_tc_ids.update(all_ids)
        
        # Step 3: 重建消息列表
        cleaned = []
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                all_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                if all_ids and all_ids.issubset(response_ids):
                    # 完全匹配 → 保留原样
                    cleaned.append(msg)
                else:
                    # 未完全匹配 → 剥离 tool_calls，仅保留 content
                    if msg.content:
                        cleaned.append(AIMessage(content=msg.content))
                    # 如果既没有 content 也没有匹配的 tool_calls，直接丢弃
            elif isinstance(msg, ToolMessage):
                tid = getattr(msg, "tool_call_id", None)
                if tid and tid in fully_matched_tc_ids:
                    # 父 AIMessage 被保留了 → 保留此 ToolMessage
                    cleaned.append(msg)
                # 否则：孤儿 ToolMessage，丢弃
            else:
                cleaned.append(msg)
        
        return cleaned
    
    def _extract_initial_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        从消息历史中提取初始消息（SystemMessage + 第一个 HumanMessage）。
        用于在连续错误时重建干净的对话。
        """
        initial = []
        for msg in messages:
            if isinstance(msg, (SystemMessage, HumanMessage)):
                initial.append(msg)
                if isinstance(msg, HumanMessage):
                    break  # 只取第一个 HumanMessage
        return initial

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(RefinerState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue, {
            "continue": "tools",
            "end": END
        })
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def _agent_node(self, state: RefinerState) -> Dict:
        iteration = state.get("iteration_count", 0) + 1
        new_messages_to_save = []
        current_history = list(state.get("messages", []))
        
        if self.verbose:
            print(f"🤖 [Agent Loop] Iteration {iteration}...")
        
        root_node = self.node_registry[state["root_node_id"]]

        if not current_history:
            # --- 优化 B: 初始信息注入 ---
            def _get_full_text(node):
                text_parts = []
                # 跳过 "Document Root" 占位符，只收集实际内容
                if node.content.strip() not in ["Document Root", "Root", "document root", "root"]:
                    text_parts.append(node.content)
                for child in node.children:
                    text_parts.append(_get_full_text(child))
                return "\n".join(filter(None, text_parts))

            full_node_text = _get_full_text(root_node)
            # 只有当节点 ≤ 2000 tokens 时，才注入完整内容
            # 对于 > 2000 tokens 的节点，不注入内容，要求先调用 Semantic_Slicer 切分
            if root_node.total_tree_token > 2000:
                node_content_brief = f"(This node has {root_node.total_tree_token} tokens. You MUST call `Semantic_Slicer` tool FIRST to split it.)"
            else:
                node_content_brief = full_node_text

            # --- 强化版 System Prompt (同步自 PDF 处理核心逻辑) ---
            system_prompt = f"""You are an expert **Document Contextualization Agent**.
Your task is to organize document sections into **Self-Contained** chunks optimized for RAG, **WITHOUT altering the original text content**.

### CORE PRINCIPLE:
**PRESERVE THE ORIGINAL TEXT.** You must NOT rewrite, paraphrase, or replace words in the document content body. Your value is added strictly through **Metadata Headers** that provide context.

---

### 1. Mandatory Contextualization Strategies

**Strategy 1: Title Generation (MANDATORY)**
* **Goal**: Create a clear, searchable title for the chunk.
* **Format**: Add the title as a Markdown heading at the very beginning: `# {{Generated Title}}`
* **Rules**:
    * The title must be concise (3-8 words).
    * It must capture the main topic or theme.
    * **ABSOLUTELY FORBIDDEN**: DO NOT use the first sentence (or part of it) as the title.
    * **ABSOLUTELY FORBIDDEN**: DO NOT use generic placeholders like "Document Root", "Segment 1", or "Root".

**Strategy 2: Contextual Disambiguation (The "Decoder Ring" - CONDITIONAL)**
* **Goal**: Explain ambiguous pronouns (`he`, `she`, `it`, `they`, `him`, `her`, `them`) and specific references (`the plan`, `that house`, `the system`) **WITHOUT altering the original text**.
* **Action**: ONLY create a `Context Supplement` field when needed.
* **Format** (only when needed):
    ```text
    "Context Supplement": "{{Explanation of who/what pronouns refer to}}"
    ```
    **CRITICAL**: 
    * **If NO ambiguous references need resolution**: DO NOT include "Context Supplement" field at all.
    * **If ambiguous references are found BUT cannot be resolved**: DO NOT include "Context Supplement" field.
    * **ONLY include "Context Supplement"** when a pronoun/reference is found in current chunk AND its referent is found in previous chunk(s).

---

### **STRICT 3-STEP RESOLUTION PROCESS:**

#### **STEP 1: SCAN (扫描)**
* Scan the current chunk text for:
    * **Pronouns**: `he`, `she`, `it`, `they`, `him`, `her`, `them`, `his`, `hers`, `its`, `their`
    * **Specific references**: `the plan`, `that house`, `this case`, `the system`, `the money`, etc.
* **Example**: Text "Tom handed him the money." → Found: `him`, `the money`

#### **STEP 2: INTERNAL CHECK (自检)**
* **FIRST**: Check if the referent is defined WITHIN the current chunk text.
* **Rule**: If the chunk contains both the pronoun AND its referent, NO Context Supplement field is needed (omit it entirely).
* **Examples**:
    * ✅ **Clear**: "Tom handed Jim the money. Then Tom handed him the book." → `him` refers to `Jim` (found in same chunk) → **DO NOT include "Context Supplement" field**
    * ❌ **Unclear**: "Tom handed him the money." → `him` is undefined → Proceed to STEP 3

#### **STEP 3: RESOLVE (回溯)**
* **ONLY trigger if**: A pronoun/reference cannot be resolved within the current chunk.
* **Process**:
    1. Call `SIR_Query(node_id="<current_node_id>", include_lineage=true, include_entity_definitions=true)` to get previous sibling + topological lineage context. Replace `<current_node_id>` with the actual node ID you're processing.
    2. Search the 1st previous chunk content for the referent.
    3. **If found**: Include "Context Supplement" field mapping the pronoun to the referent (e.g., `"Context Supplement": "'Him' refers to Jim."`). STOP.
    4. If unresolved, re-check `SIR_Query` output (`lineage` and `definition_candidates`) and only add Context Supplement when evidence exists.
    5. **If still NOT found**: **DO NOT include "Context Supplement" field at all** (omit it entirely).
* Use only `SIR_Query` for backtracking and lineage evidence.

---

### **STRICT "STOPWORD" CONTROL (停用词控制):**

**ABSOLUTELY FORBIDDEN in Context Supplement:**
* ❌ Global document metadata: "This text is from 'Adventures of Huckleberry Finn'", "The author is Mark Twain"
* ❌ Generic placeholders: "This passage", "The document", "The text"
* ❌ Redundant information: If chunk already says "Tom Sawyer said...", DO NOT write "'Tom' refers to Tom Sawyer"
* ❌ Scene-setting that doesn't resolve pronouns: "This takes place on the Mississippi River" (unless it resolves a specific reference)

**ALLOWED in Context Supplement:**
* ✅ Direct pronoun-to-entity mappings: "'Him' refers to Jim.", "'The money' refers to the treasure found in the cave."
* ✅ Only when the referent is NOT in the current chunk but IS in a previous chunk

---

### **EXAMPLES:**

**Example 1: Clear (no supplement field needed)**
```
Text: "Tom Sawyer and Huck Finn found treasure. Tom gave Huck his share."
→ All referents clear → **DO NOT include "Context Supplement" field**

Output format:
# Tom and Huck's Treasure Discovery

Tom Sawyer and Huck Finn found treasure. Tom gave Huck his share.
```

**Example 2: Unclear pronoun (needs supplement)**
```
Current chunk: "Tom handed him the money."
Previous chunk: "Jim was waiting by the river."
→ `him` refers to `Jim` (found in previous chunk)
→ **Include "Context Supplement" field**

Output format:
# Tom Hands Money to Jim

"Context Supplement": "'Him' refers to Jim."

Tom handed him the money.
```

**Example 3: Unresolved (no supplement field)**
```
Current chunk: "He ran away."
Previous chunk 1: "The weather was sunny."
Previous chunk 2: "Birds were singing."
→ `He` cannot be resolved after checking 2 previous chunks
→ **DO NOT include "Context Supplement" field**

Output format:
# Someone Runs Away

He ran away.
```

**Example 4: Redundant (forbidden)**
```
Current chunk: "Tom Sawyer said hello."
→ DO NOT write: "'Tom' refers to Tom Sawyer" (redundant!)
→ **DO NOT include "Context Supplement" field**

Output format:
# Tom's Greeting

Tom Sawyer said hello.
```

**Strategy 3: The Content (Verbatim Preservation)**
* **Action**: Output the original text segment **exactly as it is**.
* **Constraint**:
    * **DO NOT** replace pronouns in the text (e.g., keep "he", do not change to "Judge Keady").
    * **DO NOT** fix grammar or modernize language.
    * **DO NOT** merge segments if the input was split.

---

### 2. Operational Rules (STRICT ENFORCEMENT)

**Case A: Small Nodes (< 2000 tokens)**
1.  Generate a concise Title (Strategy 1).
2.  Conditionally create a Context Supplement if needed (Strategy 2) - **ONLY if pronouns/references cannot be resolved in current chunk BUT are found in previous chunk(s)**.
3.  Output the **original content verbatim** (Strategy 3).
4.  Return as **ONE string** in the list.

**Case B: Large Nodes (> 2000 tokens)**
* **CRITICAL STEP**: You **MUST** call the `Semantic_Slicer` tool FIRST.
* **Process**: For EACH resulting segment returned by the tool:
    1.  Call `Chunk_Enhancer` to generate a unique semantic signature title.
    2.  Conditionally create a Context Supplement specific to *that segment* - **ONLY if needed** (follow 3-step resolution process).
    3.  Output the **original content** of that segment.
* **Output**: A list of multiple strings (one per segment). **DO NOT merge them back.**
* **Warning**: DO NOT try to process >2000 token nodes without splitting. It will fail.

---

### 3. Output Format (Strict JSON)

Wrap results in `<final_chunks>` tags as a **valid JSON list of strings**.

**Format A: When Context Supplement is needed**
```text
# {{Title}}

"Context Supplement": "{{Contextual Explanation}}"

{{Original Content Verbatim}}
```

**Format B: When Context Supplement is NOT needed (default)**
```text
# {{Title}}

{{Original Content Verbatim}}
```

**CRITICAL FORMAT RULES:**
* **Default state**: **DO NOT include "Context Supplement" field** - assume chunk is clear
* **ONLY include "Context Supplement" field** when:
    * A pronoun/reference is found in current chunk
    * The referent is NOT in current chunk
    * The referent IS found in previous chunk(s) (max 2)
* **If unresolved after 2 previous chunks**: **DO NOT include "Context Supplement" field**
* **Format**: Use double quotes and proper JSON escaping
"""
            # 根据节点大小和内容类型，给出不同的指令
            # 检查是否是占位符标题
            is_placeholder_title = root_node.content.strip() == "Document Root" or root_node.content.strip().lower() in ["root", "document root"]
            
            # 构建工具使用说明
            tool_usage_guide = f"""
**CRITICAL TOOL USAGE FOR PRONOUN RESOLUTION:**

Current Node ID: `{root_node.node_id}`

**3-Step Resolution Process:**

**STEP 1: SCAN** - Identify pronouns/references in current chunk
**STEP 2: INTERNAL CHECK** - Check if referent exists in current chunk
  - If YES → DO NOT include "Context Supplement" field (all clear)
  - If NO → Proceed to STEP 3

**STEP 3: RESOLVE** (only if referent NOT in current chunk):
  1. Call `SIR_Query(node_id="{root_node.node_id}", include_lineage=true, include_entity_definitions=true)` to get previous sibling + lineage context
  2. Search `previous_sibling.content` and `definition_candidates` for referent
  3. If found → Include "Context Supplement" field with mapping (e.g., "'Him' refers to Jim.") → STOP
  4. If NOT found → Re-check `lineage` and `definition_candidates` from SIR_Query
  5. If still NOT found → DO NOT include "Context Supplement" field at all

**CRITICAL REMINDERS**: 
- Default: DO NOT include "Context Supplement" field (assume chunk is clear)
- Only include "Context Supplement" field when pronoun CANNOT be resolved in current chunk BUT IS found in previous chunk(s)
- Use only `SIR_Query`, no other backtracking tools
- DO NOT write redundant information already in the chunk
- DO NOT write global document metadata (author, book title, etc.)
"""
            
            if root_node.total_tree_token > 2000:
                task_instruction = f"CRITICAL: This node has {root_node.total_tree_token} tokens (> 2000). You MUST:\n1. Call `Semantic_Slicer` tool FIRST to split it into multiple segments.\n2. Output MULTIPLE chunks (one per segment), NOT a single merged chunk.\n3. For EACH segment: (a) Call `Chunk_Enhancer` to generate title, (b) Follow the 3-step resolution process for Context Supplement (scan → internal check → resolve with SIR_Query), (c) Output the original content VERBATIM without any changes.\n\n{tool_usage_guide}\n\n[Full Node Content to Process]:\n{node_content_brief}"
            elif is_placeholder_title:
                # 特殊处理：如果是占位符标题，强制要求生成新标题
                # CRITICAL: 内容中可能包含 "Document Root" 占位符，必须忽略它，只处理实际内容
                task_instruction = f"SPECIAL CASE: This node has a placeholder title ('{root_node.content}').\n\nCRITICAL INSTRUCTIONS:\n1. IGNORE any placeholder text like 'Document Root', 'Root', etc. in the content\n2. Generate a simple, clear title based on the ACTUAL CONTENT (Strategy 1)\n3. Follow the 3-step resolution process for Context Supplement:\n   - SCAN for pronouns/references\n   - INTERNAL CHECK if referent exists in current chunk\n   - RESOLVE by calling tools if needed (max 2 previous chunks)\n4. Output the original content VERBATIM without any changes (Strategy 3)\n5. Output format: <final_chunks>[\"# Title\\n\\n\\\"Context Supplement\\\": \\\"...\\\"\\n\\nOriginal content verbatim\"]</final_chunks>\n\n{tool_usage_guide}\n\n[Full Node Content to Process]:\n{node_content_brief}\n\nREMEMBER: Do NOT include 'Document Root' or any placeholder text in your output! Do NOT modify the original text!"
            else:
                task_instruction = f"This node has {root_node.total_tree_token} tokens (≤ 2000). Do NOT split. You MUST:\n1. Call `Chunk_Enhancer` to generate a concise title based on content (Strategy 1).\n2. Follow the 3-step resolution process for Context Supplement:\n   - SCAN for pronouns/references in current chunk\n   - INTERNAL CHECK if referent exists in current chunk\n   - RESOLVE by calling `SIR_Query` (previous sibling + lineage)\n3. Output the original content VERBATIM without any changes (Strategy 3).\n4. Output as ONE chunk.\n\n{tool_usage_guide}\n\n[Full Node Content to Process]:\n{node_content_brief}"
            
            init_msgs = [
                SystemMessage(content=system_prompt), 
                HumanMessage(content=f"{task_instruction}\n\nTask: Process the content now. Output ONLY the <final_chunks> tags with a valid JSON list of strings. Each string MUST:\n1. Start with a title: `# Title` (concise summary, 3-8 words, NOT the first sentence)\n2. Conditionally include `\"Context Supplement\": \"...\"` field:\n   - **ONLY include** when pronoun is NOT in current chunk BUT IS found in previous chunk(s)\n   - **DO NOT include** if all pronouns are clear in current chunk\n   - **DO NOT include** if pronoun cannot be resolved after checking 2 previous chunks\n3. Include the ORIGINAL content VERBATIM without any modifications\n4. Wrap everything in <final_chunks>...</final_chunks> tags as a JSON list.\n\n**CRITICAL REMINDERS:**\n- **Default**: DO NOT include \"Context Supplement\" field at all (omit it entirely)\n- **Only include** \"Context Supplement\" when pronoun is NOT in current chunk BUT IS found in previous chunk(s)\n- DO NOT write redundant information (e.g., if chunk says \"Tom Sawyer\", don't write \"'Tom' refers to Tom Sawyer\")\n- DO NOT write global document metadata\n- Maximum backtracking: 2 previous chunks only\n\nExample (with resolved pronoun - include Context Supplement): <final_chunks>[\"# Tom Hands Money to Jim\\n\\n\\\"Context Supplement\\\": \\\"'Him' refers to Jim.\\\"\\n\\nTom handed him the money.\"]</final_chunks>\n\nExample (all clear - NO Context Supplement field): <final_chunks>[\"# Tom and Jim's Conversation\\n\\nTom Sawyer and Jim talked about the plan. Tom gave Jim the money.\"]</final_chunks>\n\nExample (unresolved - NO Context Supplement field): <final_chunks>[\"# Someone Runs Away\\n\\nHe ran away.\"]</final_chunks>")
            ]
            new_messages_to_save.extend(init_msgs)
            current_history.extend(init_msgs)

        consecutive_tool_errors = state.get("consecutive_tool_errors", 0)
        
        # --- 根据连续错误次数选择策略 ---
        if consecutive_tool_errors >= 2:
            # 连续错误超过 2 次：彻底重建对话，只用初始消息，不绑定工具
            if self.verbose:
                print(f"🔄 [Recovery] 连续 {consecutive_tool_errors} 次 tool_calls 错误，重建对话，禁用工具")
            initial_msgs = self._extract_initial_messages(current_history)
            initial_msgs.append(HumanMessage(
                content="IMPORTANT: Do NOT use any tools. Output <final_chunks> directly based on the content provided above. "
                        "Generate a title and include the original content verbatim."
            ))
            try:
                response = self.base_llm.invoke(initial_msgs)  # 不绑定工具
            except Exception as e:
                if self.verbose:
                    print(f"🧠 [Agent Thought] Recovery also failed: {e}")
                response = AIMessage(content="<final_chunks>[\"Error: Unable to process this node.\"]</final_chunks>")
            
            new_messages_to_save.append(response)
            if response.content and self.verbose:
                print(f"\n🧠 [Agent Thought] {response.content[:150]}...")
            return {
                "messages": new_messages_to_save, 
                "iteration_count": iteration, 
                "validation_errors": [],
                "consecutive_tool_errors": consecutive_tool_errors + 1
            }
        
        # --- 正常流程 ---
        target_llm = self.base_llm.bind_tools(self.tools)
        
        if state.get("validation_errors"):
            last_error = state["validation_errors"][-1]
            error_msg = HumanMessage(content=f"Previous output format was invalid: {last_error}. Please fix and output <final_chunks> again.")
            new_messages_to_save.append(error_msg)
            current_history.append(error_msg)
        
        try:
            # 清理消息历史：确保 tool_calls 和 ToolMessage 严格配对
            cleaned_history = self._clean_message_history(current_history)
            response = target_llm.invoke(cleaned_history)
            # 成功调用 → 重置连续错误计数
            new_messages_to_save.append(response)
            if response.content and self.verbose:
                print(f"\n🧠 [Agent Thought] {response.content[:150]}...")
            return {
                "messages": new_messages_to_save, 
                "iteration_count": iteration, 
                "validation_errors": [],
                "consecutive_tool_errors": 0  # 重置
            }
        except Exception as e:
            error_msg_str = str(e)
            if self.verbose:
                print(f"🧠 [Agent Thought] Error in LLM invocation: {error_msg_str}")
            
            is_tool_error = any(kw in error_msg_str.lower() for kw in [
                "tool_calls", "tool messages", "insufficient tool messages",
                "tool_call_id"
            ])
            
            if is_tool_error:
                if self.verbose:
                    print(f"🧠 [Agent Thought] Tool calls 配对错误（第 {consecutive_tool_errors + 1} 次），尝试恢复...")
                
                # 用初始消息 + 请求直接输出
                initial_msgs = self._extract_initial_messages(current_history)
                initial_msgs.append(HumanMessage(
                    content="Please output <final_chunks> directly. Generate a title and include the original content."
                ))
                try:
                    response = self.base_llm.invoke(initial_msgs)  # 不绑定工具
                except Exception as e2:
                    response = AIMessage(content=f"Error: {error_msg_str}")
                
                new_messages_to_save.append(response)
                if response.content and self.verbose:
                    print(f"\n🧠 [Agent Thought] {response.content[:150]}...")
                return {
                    "messages": new_messages_to_save, 
                    "iteration_count": iteration, 
                    "validation_errors": [],
                    "consecutive_tool_errors": consecutive_tool_errors + 1  # 递增
                }
            else:
                # 非 tool 错误
                response = AIMessage(content=f"Error: {error_msg_str}. Please output <final_chunks> directly.")
                new_messages_to_save.append(response)
                return {
                    "messages": new_messages_to_save, 
                    "iteration_count": iteration, 
                    "validation_errors": [],
                    "consecutive_tool_errors": 0
                }

    def _tool_node(self, state: RefinerState) -> Dict:
        last_msg = state["messages"][-1]
        
        # 检查是否有 tool_calls
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            if self.verbose:
                print("⚠️ [Tool Node] No tool_calls found in last message, skipping tool execution")
            return {"messages": []}
        
        tools_map = {t.name: t for t in self.tools}
        results = []
        
        for tool_call in last_msg.tool_calls:
            try:
                if self.verbose:
                    print(f"🛠️ [Refiner Tool] {tool_call['name']}")
                if tool_call["name"] not in tools_map:
                    content = f"Error: Tool '{tool_call['name']}' not found."
                else:
                    res = tools_map[tool_call["name"]].invoke(tool_call["args"])
                    content = json.dumps(res, ensure_ascii=False)
            except Exception as e:
                content = f"Tool Execution Error: {str(e)}"
                if self.verbose:
                    print(f"⚠️ [Tool Error] {str(e)}")
            
            results.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
        
        return {"messages": results}

    def _should_continue(self, state: RefinerState) -> Literal["continue", "end"]:
        last_msg = state["messages"][-1]
        iteration = state.get("iteration_count", 0)
        consecutive_tool_errors = state.get("consecutive_tool_errors", 0)
        
        # 1. 硬性迭代上限
        if iteration > 20:
            if self.verbose:
                print(f"⚠️ 迭代次数过多 ({iteration})，强制终止")
            return "end"
        
        # 2. 连续 tool_calls 错误过多 → 直接终止
        if consecutive_tool_errors >= 3:
            if self.verbose:
                print(f"⚠️ 连续 {consecutive_tool_errors} 次 tool_calls 错误，强制终止")
            return "end"
        
        # 3. 检查最后一条消息是否包含 error 信息（避免错误循环）
        content = last_msg.content or ""
        if content.startswith("Error:") and "<final_chunks>" not in content:
            if iteration > 5:
                if self.verbose:
                    print(f"⚠️ 检测到错误响应且迭代已超过 5 次，终止")
                return "end"
        
        # 4. 检查是否有工具调用
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            if isinstance(last_msg, AIMessage):
                # 统计历史工具调用次数
                tool_call_count = sum(
                    1 for msg in state.get("messages", []) 
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls
                )
                if tool_call_count > 30:
                    if self.verbose:
                        print(f"⚠️ 工具调用次数过多 ({tool_call_count})，强制终止")
                    return "end"
                return "continue"
            else:
                return "end"
            
        # 5. 检查是否有 <final_chunks> 输出
        if "<final_chunks>" in content:
            match = re.search(r"<final_chunks>(.*?)</final_chunks>", content, re.DOTALL)
            if not match:
                state["validation_errors"].append("未找到完整的 <final_chunks> 闭合标签。")
                return "continue"
            
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                if not isinstance(parsed, list):
                    state["validation_errors"].append("标签内部必须是一个 JSON 列表格式。")
                    return "continue"
                return "end"
            except json.JSONDecodeError:
                if len(state.get("validation_errors", [])) >= 3:
                    if self.verbose:
                        print("⚠️ 连续 3 次格式校验失败，强制终止，进入启发式修复。")
                    return "end"
                
                if iteration < 8:
                    state["validation_errors"].append("标签内部不是合法的 JSON 格式。请确保使用双引号。")
                    return "continue"
                return "end"
                
        return "continue"

    def refine(self, root_node: SIRNode, query: str = "") -> List[str]:
        """递归精炼：将大任务拆解为小任务"""
        
        # --- 策略 1: 结构化递归 (Python 驱动) ---
        # 优化：只有当节点较大且包含子章节（Heading）时，才进行结构化拆分。
        # 如果子节点全是 Text 类型，则视为一个整体，交给 Agent 进行语义切分，避免过度碎片化。
        has_heading_children = any(c.node_type == 'heading' for c in root_node.children)
        if root_node.total_tree_token > 2000 and has_heading_children:
            if self.verbose:
                print(f"分而治之 📂: Section '{root_node.path_str}' is large ({root_node.total_tree_token} tokens) and has sub-headings. Splitting by children.")
            all_sub_chunks = []
            for child in root_node.children:
                all_sub_chunks.extend(self.refine(child, query=query))
            return all_sub_chunks

        # --- 优化策略: 快车道 (Fast Path) ---
        # 对于超小的叶子节点（< 30 tokens），直接返回，不再触发 Agent。
        # 理由：这类节点通常是标题或简单短句，代词替换意义不大，且节省成本。
        if root_node.total_tree_token < 30 and not root_node.children:
            if self.verbose:
                print(f"快车道 ⚡: Node '{root_node.node_id}' is very small ({root_node.total_tree_token} tokens). Skipping Agent.")
            return [root_node.content]

        # --- 策略 2: 原子节点精炼 (Agent 驱动) ---
        if self.verbose:
            print(f"智能精炼 🧠: Processing node '{root_node.node_id}' ({root_node.total_tree_token} tokens)")
        
        initial_state = {
            "messages": [],
            "root_node_id": root_node.node_id,
            "final_chunks": [],
            "iteration_count": 0,
            "validation_errors": [],
            "query": query,
            "consecutive_tool_errors": 0
        }
        final_state = self.workflow.invoke(initial_state)
        last_content = final_state["messages"][-1].content
        
        # 调试：打印 Agent 返回的内容（前2000字符）
        if self.verbose:
            print(f"  📝 Agent returned content (first 2000 chars):\n{last_content[:2000]}")
            print(f"  📝 Agent returned content length: {len(last_content)} chars")
            print(f"  📝 Contains <final_chunks> tag: {'<final_chunks>' in last_content}")
        
        # --- 解析逻辑 (软着陆) ---
        match = re.search(r"<final_chunks>(.*?)</final_chunks>", last_content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"⚠️ Agent 返回格式非标准 JSON，尝试启发式修复...")
                lines = [l.strip() for l in json_str.split('\n') if l.strip()]
                cleaned_lines = [re.sub(r'^(\d+[\.、\s]+|[-*+]\s+)', '', l) for l in lines]
                if cleaned_lines:
                    return cleaned_lines
        
        # --- 最后的兜底逻辑 ---
        # 如果 Agent 没有返回正确格式，尝试从返回内容中提取
        if self.verbose:
            print(f"  ⚠️ WARNING: Agent did not return proper <final_chunks> format!")
            print(f"  📝 Agent returned (first 1000 chars): {last_content[:1000]}")
        
        # 尝试从 Agent 返回的内容中提取 chunk（即使没有正确的标签）
        # 检查是否包含标题（# 开头）或内容
        if last_content and len(last_content.strip()) > 0:
            # 如果内容包含 # 开头的行，可能是标题
            lines = last_content.split('\n')
            potential_chunks = []
            current_chunk = []
            
            for line in lines:
                stripped = line.strip()
                # 跳过 "Document Root" 占位符
                if stripped in ["Document Root", "Root", "document root", "root"]:
                    continue
                if stripped.startswith('#'):
                    # 如果遇到新标题，保存之前的 chunk
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk).strip()
                        # 再次过滤掉占位符
                        if chunk_text and "Document Root" not in chunk_text:
                            potential_chunks.append(chunk_text)
                    current_chunk = [line]
                elif stripped:
                    current_chunk.append(line)
            
            # 保存最后一个 chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                # 再次过滤掉占位符
                if chunk_text and "Document Root" not in chunk_text:
                    potential_chunks.append(chunk_text)
            
            if potential_chunks:
                if self.verbose:
                    print(f"  ✅ Extracted {len(potential_chunks)} chunks from Agent response (without proper tags)")
                return potential_chunks
        
        # 如果无法提取，说明 Agent 完全失败了
        # 这种情况下，不应该返回原始内容（因为会跳过所有处理）
        # 而是应该返回空列表，让调用方知道处理失败
        if self.verbose:
            print(f"  ❌ CRITICAL: Agent failed completely. Cannot extract any chunks.")
            print(f"  💡 This usually means:")
            print(f"     1. LLM API error or timeout")
            print(f"     2. LLM did not understand the instructions")
            print(f"     3. System Prompt or task_instruction needs improvement")
        return []
