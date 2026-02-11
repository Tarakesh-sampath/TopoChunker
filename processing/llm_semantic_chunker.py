"""
LLM-based Semantic Chunker (Lumber Sentence Style)

复刻 LumberChunker 的核心逻辑，改为句子版：
1. 预处理：将文本拆成带编号的句子 [S 0], [S 1], ...
2. 积攒：从当前位置往后凑够 threshold tokens
3. 判决：带编号的句子发给 LLM，让 LLM 返回切分点的句子 ID
4. 切分：根据 LLM 返回的 ID 切下前半部分为 Chunk
5. 移动指针：从切分点继续下一轮

关键设计（复刻 LumberChunker 动态长度）：
- NO_SPLIT 时不立即归档，而是继续积攒更多句子，直到 max_tokens 才强制切
- 这样保证了 chunk 长度是动态的，而非固定长度
"""

import re
from typing import List, Union
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from agents.llm_factory import create_llm

# === 复刻 LumberChunker 的 Prompt (改为句子版) ===
SYSTEM_PROMPT = """
You will receive a text comprised of numbered sentences (e.g., [S 0], [S 1]...).

**Task:**
Identify the **FIRST sentence ID** (starting from [S 1], do **NOT** pick [S 0]) where the topic, scene, or time shifts **drastically** compared to the sentences before it.

**Rules:**
1. **NOT THE FIRST ONE:** Do not select [S 0]. The first chunk must contain at least the first sentence.
2. **PREFER LONGER CHUNKS:** If a shift occurs very early in the text window (e.g., within the first 300 words), IGNORE it unless it is a drastic scene change (e.g., a completely new chapter or a major time jump). We prefer chunks that are substantial in length.
3. **MERGE IF POSSIBLE:** If the text flows reasonably well, even with minor topic drifts or dialogue shifts, keep it together. Only split on **major, unmistakable** boundaries.
4. **MAJOR SHIFTS ONLY:** Look for clear, drastic boundaries (e.g., "Chapter X", "The next day...", "Meanwhile, in another city...", a complete change of characters or setting). Minor transitions do NOT warrant a split.
5. **GROUP HEADERS:** If you see a Chapter Title, the shift starts AT the title. Select the ID of the title.

**Output Format:**
- Return ONLY the ID of the shift point (e.g., "S 12").
- If no significant shift occurs in this window, return "NO_SPLIT".
"""


class LLMSemanticChunker:
    """
    LLM-based semantic chunker using LumberChunker sentence-style approach.
    
    核心改进：
    - 将文本拆为编号句子，LLM 直接返回句子 ID，避免"找不到切分句"的问题
    - 以句子索引控制流程，逻辑清晰，不会死循环也不会越界
    - NO_SPLIT 时继续积攒，直到 max_tokens 才强制切分（复刻 LumberChunker 动态长度）
    """
    
    def __init__(
        self, 
        model_type: str = "deepseek",
        model_name: str = "deepseek-chat",
        target_chunk_tokens: int = 1000,    # 阈值：攒够多少 token 问一次 LLM
        max_chunk_tokens: int = 2500,       # 硬上限：NO_SPLIT 时继续积攒直到此值才强制切
        max_llm_window_tokens: int = 3000,  # 保留兼容性
        llm: BaseChatModel = None
    ):
        self.llm = llm or create_llm(model_type=model_type, model_name=model_name, temperature=0.0)
        self.threshold = target_chunk_tokens
        self.max_tokens = max_chunk_tokens

    def _get_token_count(self, text: str) -> int:
        """简单的 token 估算（约 4 字符/token）"""
        return len(text) // 4

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本拆分成句子级别的小单元。
        使用更鲁棒的正则：先按换行拆，再按句末标点+空格拆。
        支持：省略号(... / …)、数字开头(1. 2.)、大写字母、引号、括号。
        """
        lines = text.split('\n')
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # (?<=[.!?…]) 后行断言：前一个字符是句末标点（含省略号字符 …）
            # \s+ 匹配一个或多个空白
            # (?=[A-Z0-9"'"\u201c\u201d(\[]) 前行断言：下一个字符是大写字母、数字、引号、括号
            parts = re.split(r'(?<=[.!?…])\s+(?=[A-Z0-9"\'"\u201c\u201d(\[])', line)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
        return sentences

    def _call_llm(self, text: str) -> str:
        """调用 LLM 判断切分点，返回 "S XX" 或 "NO_SPLIT" """
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Here is the text:\n{text}")
            ]
            resp = self.llm.invoke(messages)
            return resp.content.strip()
        except Exception as e:
            print(f"⚠️ LLM Error: {e}")
            return "NO_SPLIT"

    def split(self, text_input: Union[str, List[str]]) -> List[str]:
        """
        使用 LumberChunker 句子版方式将文本切分为语义 chunks。
        
        流程：
        1. 全文拆为句子列表
        2. 逐句累积 buffer，达到 threshold 时问 LLM
        3. LLM 找到切分点 → 切下 Chunk，移动指针
        4. LLM 说 NO_SPLIT → 继续积攒，直到 max_tokens 才强制切
        
        Args:
            text_input: 输入文本（字符串或段落列表）
            
        Returns:
            切分后的 chunk 列表
        """
        # ========== 1. 预处理：统一转为字符串 ==========
        if isinstance(text_input, str):
            text = text_input.strip()
        elif isinstance(text_input, list):
            text = "\n\n".join(p.strip() for p in text_input if p.strip())
        else:
            text = str(text_input).strip()

        if not text:
            print("⚠️ 输入文本为空")
            return []

        # ========== 2. 拆分为句子 ==========
        all_sentences = self._split_into_sentences(text)
        
        if not all_sentences:
            print("⚠️ 无法从文本中提取句子")
            return [text]
        
        total_tokens = self._get_token_count(text)
        print(f"🚀 开始 Lumber 模式切分 (句子版) | 总句数: {len(all_sentences)} | "
              f"总 tokens: ~{total_tokens} | 触发阈值: {self.threshold} tokens | "
              f"硬上限: {self.max_tokens} tokens")
        
        final_chunks = []
        
        # chunk_start: 当前 chunk 起始句子索引
        # cursor: 扫描指针，逐句前移
        chunk_start = 0
        cursor = 0
        buffer_tokens = 0            # 从 chunk_start 到 cursor 的累积 token 数
        next_ask_tokens = self.threshold  # 下次问 LLM 的 token 门槛
        
        while cursor < len(all_sentences):
            # ========== 3. 逐句积攒 ==========
            buffer_tokens += self._get_token_count(all_sentences[cursor])
            cursor += 1
            
            # 还没到门槛，继续积攒
            if buffer_tokens < next_ask_tokens and cursor < len(all_sentences):
                continue
            
            # 到达文末或到达门槛 —— 需要做决策
            buffer_sentences = all_sentences[chunk_start : cursor]
            
            # 如果是文末且量太少（< 50% threshold），直接归档不问 LLM
            if cursor >= len(all_sentences) and buffer_tokens < self.threshold * 0.5:
                chunk_text = " ".join(buffer_sentences)
                if chunk_text.strip():
                    final_chunks.append(chunk_text.strip())
                    print(f"📝 最后剩余归档为 Chunk {len(final_chunks)} "
                          f"({self._get_token_count(chunk_text)} tokens)")
                break
            
            # ========== 4. 构建带 ID 的文本给 LLM ==========
            numbered_text = "\n".join([f"[S {i}] {s}" for i, s in enumerate(buffer_sentences)])
            
            # ========== 5. 询问 LLM ==========
            decision = self._call_llm(numbered_text)
            
            if "NO_SPLIT" in decision:
                # ===== NO_SPLIT 逻辑（复刻 LumberChunker 动态长度）=====
                if buffer_tokens >= self.max_tokens or cursor >= len(all_sentences):
                    # 已到硬上限或文末：强制归档整个 buffer
                    chunk_text = " ".join(buffer_sentences)
                    if chunk_text.strip():
                        final_chunks.append(chunk_text.strip())
                        reason = "硬上限" if buffer_tokens >= self.max_tokens else "文末"
                        print(f"⚠️ NO_SPLIT 且达到{reason} ({buffer_tokens} tokens) -> "
                              f"强制归档 Chunk {len(final_chunks)} ({len(buffer_sentences)} 句)")
                    # 重置
                    chunk_start = cursor
                    buffer_tokens = 0
                    next_ask_tokens = self.threshold
                else:
                    # 还有空间 → 继续积攒，下次多攒一些再问
                    increment = max(self.threshold // 2, 300)
                    next_ask_tokens = buffer_tokens + increment
                    print(f"💡 NO_SPLIT ({buffer_tokens} tokens)，继续积攒到 {next_ask_tokens} tokens...")
                    # 不切分，不移动 chunk_start，继续外层 while 循环
            else:
                # ===== 有切分点 =====
                match = re.search(r"S\s*(\d+)", decision)
                if match:
                    split_offset = int(match.group(1))
                    print(f"✂️  LLM 决定在局部 ID [S {split_offset}] 处切分")
                else:
                    # 解析失败 → 归档整个窗口
                    split_offset = len(buffer_sentences)
                    print(f"⚠️ 无法解析 LLM 输出 '{decision[:80]}'，将窗口全部归档")
                
                # 边界检查
                if split_offset <= 0:
                    split_offset = 1  # 至少切一句，防止死循环
                if split_offset > len(buffer_sentences):
                    split_offset = len(buffer_sentences)
                
                # 生成 Chunk：chunk_start 到 chunk_start + split_offset
                chunk_sents = all_sentences[chunk_start : chunk_start + split_offset]
                chunk_text = " ".join(chunk_sents)
                if chunk_text.strip():
                    final_chunks.append(chunk_text.strip())
                    print(f"✅ 生成 Chunk {len(final_chunks)}: "
                          f"{self._get_token_count(chunk_text)} tokens, "
                          f"{split_offset} 句")
                
                # 移动 chunk_start 到切分点
                chunk_start += split_offset
                # 重要：cursor 回退到 chunk_start，重新从新 chunk 开始积攒
                cursor = chunk_start
                buffer_tokens = 0
                next_ask_tokens = self.threshold
        
        # ========== 处理最后可能遗漏的 buffer ==========
        if chunk_start < len(all_sentences):
            remaining_sents = all_sentences[chunk_start:]
            chunk_text = " ".join(remaining_sents)
            if chunk_text.strip():
                final_chunks.append(chunk_text.strip())
                print(f"📝 保存最后一个 Chunk {len(final_chunks)} "
                      f"({self._get_token_count(chunk_text)} tokens)")
        
        # ========== 安全检查 ==========
        if not final_chunks:
            print(f"⚠️ 未生成任何 chunks，返回原始文本")
            final_chunks = [text]
        
        print(f"✅ Lumber 句子版切分完成，共生成 {len(final_chunks)} 个 chunks")
        return final_chunks


# ================= 测试 =================
if __name__ == "__main__":
    # 模拟：标题 + 正文 + 新标题 + 正文
    text = "Chapter 1: The Beginning. "
    text += "This is sentence one. " * 50  # 50句
    text += "Chapter 2: The Middle. "  # 应该是切分点
    text += "This is sentence two. " * 60
    
    splitter = LLMSemanticChunker(target_chunk_tokens=500)  # 设小点方便测
    chunks = splitter.split(text)
    
    for i, c in enumerate(chunks):
        print(f"\nChunk {i+1}:\n{c[:100]}...")
