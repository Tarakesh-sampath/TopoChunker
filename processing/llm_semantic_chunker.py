"""
LLM-based Semantic Chunker (ID-Based Deterministic Pipeline)

四阶段流水线（Four-Stage Pipeline）：

    [Stage 1] 边界标定（Regex Segmentation）
        ├─ 用正则在中英文标点处切句
        └─ 为每条句子分配全局 ID  →  [S0] [S1] [S2] ...

    [Stage 2] LLM 决策（Boundary Proposal）
        ├─ 在带 ID 的窗口内询问 LLM
        └─ LLM 仅返回主题切换的 ID（可单个或多个），或 "NO_SPLIT"

    [Stage 3] 确定性校验（Deterministic Validation）
        ├─ 解析合法性     —— 必须能从输出中抽取出整数 ID
        ├─ 范围合法性     —— 0 < id < 窗口句子数
        ├─ 最小占位检查   —— 切下的前缀不少于 min_chunk_tokens
        ├─ 最大占位检查   —— 切下的前缀不超过 max_chunk_tokens（超出则自动回退）
        └─ 重复/单调性检查 —— 多 ID 时去重、排序、过滤过近边界

    [Stage 4] 执行切分（Execute）
        └─ 按已校验 ID 切分；NO_SPLIT 或全部不通过 → 继续积攒/到达硬上限强切

LumberChunker 风格：NO_SPLIT 时不立即归档，而是继续往后积攒，直到 max_tokens 才强制切。
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import create_llm

# =====================================================================
# Prompt：要求 LLM 仅输出 ID（支持单/多 ID），或 NO_SPLIT
# =====================================================================
SYSTEM_PROMPT = """\
You will receive a text comprised of numbered sentences in the form [S0] [S1] [S2] ...

**Task:**
Identify the **sentence ID(s)** where the topic, scene, or time shifts **drastically** compared
to the sentences before it. Each ID you return marks the FIRST sentence of a new chunk.

**Rules:**
1. **NEVER pick S0.** The first chunk must contain at least the first sentence.
2. **Major shifts only.** Look for clear, drastic boundaries (e.g., "Chapter X", "The next day...",
   "Meanwhile, in another city...", a complete change of characters, setting, or topic).
   Minor transitions, dialogue beats, or small drifts do NOT warrant a split.
3. **Prefer longer chunks.** If a possible shift occurs very early in the window, IGNORE it
   unless it is unmistakable (e.g., a brand-new chapter title).
4. **Group headers.** If you see a Chapter / Section title, the shift starts AT the title;
   return the ID of the title itself.

**Output Format (STRICT):**
- If shift(s) found: return ONLY the IDs, comma-separated, e.g.  `S12`  or  `S12, S37, S58`
- If no significant shift: return exactly  `NO_SPLIT`
- Do NOT include any other text, explanation, punctuation, or markdown.
"""


# =====================================================================
# 校验结果数据结构
# =====================================================================
@dataclass
class BoundaryDecision:
    """Stage 3 确定性校验的输出。"""
    is_valid: bool
    boundary_ids: List[int] = field(default_factory=list)  # 已校验的、按升序排列的切分点
    raw_response: str = ""
    reasons: List[str] = field(default_factory=list)       # 拒绝/调整的解释（用于日志）


# =====================================================================
# 主类
# =====================================================================
class LLMSemanticChunker:
    """
    ID-based deterministic semantic chunker.

    设计要点：
      - Stage 1/3/4 全程确定性，无随机性；
      - Stage 2 是唯一可能"出错"的环节，所有结果都必须经过 Stage 3 校验；
      - 单 ID / 多 ID / NO_SPLIT 三种 LLM 输出形式统一处理。
    """

    def __init__(
        self,
        model_type: str = "deepseek",
        model_name: str = "deepseek-chat",
        target_chunk_tokens: int = 1000,
        max_chunk_tokens: int = 2500,
        min_chunk_tokens: int = 200,
        llm: Optional[BaseChatModel] = None,
    ):
        self.llm = llm or create_llm(model_type=model_type, model_name=model_name, temperature=0.0)
        self.threshold = target_chunk_tokens
        self.max_tokens = max_chunk_tokens
        self.min_tokens = min_chunk_tokens

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _get_token_count(text: str) -> int:
        """简单的 token 估算（约 4 字符/token）。"""
        return max(1, len(text) // 4)

    # ==================================================================
    # Stage 1 — 句子边界标定
    # ==================================================================
    @staticmethod
    def _segment_sentences(text: str) -> List[str]:
        """
        基于正则切句，支持中英文标点。
        - 英文：. ! ? … 之后接大写字母 / 数字 / 引号 / 括号
        - 中文：。！？…… 之后直接切（中文句子不强制后面接空白）
        """
        text = text.strip()
        if not text:
            return []

        sentences: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 中英文混合切句
            #   分隔点：
            #     英文 — 句末标点(.!?…)后跟空白，再接大写字母/数字/引号/括号/中文字符
            #     中文 — 句末标点(。！？……)后可有可无空白
            pattern = (
                r"(?<=[.!?…])\s+(?=[A-Z0-9\"'\u201c\u201d(\[\u4e00-\u9fff])"
                r"|"
                r"(?<=[。！？])\s*"
            )
            parts = re.split(pattern, line)
            for part in parts:
                if part is None:
                    continue
                part = part.strip()
                if part:
                    sentences.append(part)

        return sentences

    @staticmethod
    def _format_numbered_window(sentences: List[str]) -> str:
        """把窗口内的句子标上 [S0] [S1] ... 给 LLM。"""
        return "\n".join(f"[S{i}] {s}" for i, s in enumerate(sentences))

    # ==================================================================
    # Stage 2 — LLM 决策
    # ==================================================================
    def _query_llm(self, numbered_text: str) -> str:
        """调用 LLM，原样返回响应字符串（不做解析/校验）。"""
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Here is the text:\n{numbered_text}"),
            ]
            resp = self.llm.invoke(messages)
            return (resp.content or "").strip()
        except Exception as e:
            print(f"⚠️ LLM Error: {e}")
            return "NO_SPLIT"

    # ==================================================================
    # Stage 3 — 确定性校验
    # ==================================================================
    def _validate_boundary(
        self,
        raw_response: str,
        buffer_sentences: List[str],
    ) -> BoundaryDecision:
        """
        对 LLM 的原始输出进行严格的、与 LLM 解耦的确定性校验。

        校验顺序：
          1. NO_SPLIT 短路
          2. 解析所有形如 `S\\d+` 的 ID（支持单 ID / 多 ID）
          3. 范围检查：0 < id < n
          4. 去重 + 升序
          5. 过滤过近的边界（前缀至少 min_chunk_tokens）
          6. 拆出超过 max_chunk_tokens 的部分时，自动回退到不超限的最大 ID
          7. 至少保留一个有效边界，否则 is_valid=False
        """
        decision = BoundaryDecision(is_valid=False, raw_response=raw_response)
        n = len(buffer_sentences)

        # --- 1. NO_SPLIT 短路 ---
        if not raw_response or "NO_SPLIT" in raw_response.upper():
            decision.reasons.append("LLM returned NO_SPLIT (or empty)")
            return decision

        # --- 2. 抽取 ID（支持 "S12", "S 12", "S12, S37" 等形式）---
        raw_ids = [int(m) for m in re.findall(r"S\s*(\d+)", raw_response, flags=re.IGNORECASE)]
        if not raw_ids:
            decision.reasons.append(f"Cannot parse any S<id> from response: '{raw_response[:80]}'")
            return decision

        # --- 3+4. 范围 / 去重 / 升序 ---
        seen = set()
        candidates: List[int] = []
        for cid in raw_ids:
            if cid <= 0:
                decision.reasons.append(f"Drop S{cid}: cannot split at the very start")
                continue
            if cid >= n:
                decision.reasons.append(f"Drop S{cid}: out of window range [0, {n})")
                continue
            if cid in seen:
                continue
            seen.add(cid)
            candidates.append(cid)
        candidates.sort()

        if not candidates:
            decision.reasons.append("All proposed IDs failed range check")
            return decision

        # --- 5. 最小占位（每段不少于 min_chunk_tokens）---
        token_counts = [self._get_token_count(s) for s in buffer_sentences]

        accepted: List[int] = []
        prev_id = 0
        for cid in candidates:
            seg_tokens = sum(token_counts[prev_id:cid])
            if seg_tokens < self.min_tokens:
                decision.reasons.append(
                    f"Drop S{cid}: prefix from S{prev_id} only {seg_tokens} tokens "
                    f"(< min {self.min_tokens})"
                )
                continue
            accepted.append(cid)
            prev_id = cid

        if not accepted:
            decision.reasons.append("All boundaries below min_chunk_tokens")
            return decision

        # --- 6. 最大占位（如果某段 > max_tokens，强制在 max 处插入额外切点）---
        # 注意：这里只做"自动回退"——把 LLM 给出的过大段在硬上限处切开，
        # 不改变 LLM 已选定的有效边界，只追加额外保护性边界。
        final_ids: List[int] = []
        prev_id = 0

        def _force_split_in_range(start: int, end: int) -> List[int]:
            """在 [start, end) 范围内，每超过 max_tokens 就强制切一刀。返回追加的边界列表。"""
            extra: List[int] = []
            accum = 0
            for i in range(start, end):
                accum += token_counts[i]
                if accum > self.max_tokens:
                    cut = max(start + 1, i)  # 至少切走 1 句，防死循环
                    if cut < end and cut not in seen:
                        extra.append(cut)
                        seen.add(cut)
                        accum = 0
                        # 调整起点，避免重复触发
                        start = cut
            return extra

        for cid in accepted:
            forced = _force_split_in_range(prev_id, cid)
            for f in forced:
                if f > prev_id and f < cid:
                    final_ids.append(f)
                    decision.reasons.append(f"Auto-add S{f}: enforce max_chunk_tokens budget")
            final_ids.append(cid)
            prev_id = cid

        # 处理最后一段（prev_id .. n）也可能超长，但通常由外层 max_tokens 兜底，这里不再追加。
        final_ids = sorted(set(final_ids))

        if not final_ids:
            decision.reasons.append("No boundary survived all checks")
            return decision

        decision.is_valid = True
        decision.boundary_ids = final_ids
        return decision

    # ==================================================================
    # Stage 4 — 执行切分（主控循环）
    # ==================================================================
    def split(self, text_input: Union[str, List[str]]) -> List[str]:
        """
        主入口：将原文切为语义 chunks。
        """
        # ----- 输入归一化 -----
        if isinstance(text_input, list):
            text = "\n\n".join(p.strip() for p in text_input if p and p.strip())
        else:
            text = (text_input or "").strip()

        if not text:
            print("⚠️ 输入文本为空")
            return []

        # ----- Stage 1：全文切句 -----
        all_sentences = self._segment_sentences(text)
        if not all_sentences:
            print("⚠️ 无法从文本中提取句子")
            return [text]

        total_tokens = self._get_token_count(text)
        print(
            f"🚀 ID-based 切分开始 | 总句数: {len(all_sentences)} | "
            f"总 tokens ≈ {total_tokens} | 阈值: {self.threshold} | "
            f"硬上限: {self.max_tokens} | 最小段: {self.min_tokens}"
        )

        final_chunks: List[str] = []

        chunk_start = 0                       # 当前 chunk 起点（句子索引）
        cursor = 0                            # 扫描指针
        buffer_tokens = 0                     # chunk_start..cursor 累积 token
        next_ask_tokens = self.threshold      # 下次询问 LLM 的门槛

        while cursor < len(all_sentences):
            # 逐句积攒
            buffer_tokens += self._get_token_count(all_sentences[cursor])
            cursor += 1

            # 还没到询问门槛 → 继续积攒
            if buffer_tokens < next_ask_tokens and cursor < len(all_sentences):
                continue

            buffer_sentences = all_sentences[chunk_start:cursor]

            # 文末且量太少 → 直接归档不问 LLM
            if cursor >= len(all_sentences) and buffer_tokens < self.threshold * 0.5:
                tail_text = " ".join(buffer_sentences).strip()
                if tail_text:
                    final_chunks.append(tail_text)
                    print(
                        f"📝 文末归档 Chunk {len(final_chunks)} "
                        f"(~{self._get_token_count(tail_text)} tokens)"
                    )
                break

            # ----- Stage 2：LLM 决策（局部 ID）-----
            numbered_text = self._format_numbered_window(buffer_sentences)
            raw_response = self._query_llm(numbered_text)

            # ----- Stage 3：确定性校验 -----
            decision = self._validate_boundary(raw_response, buffer_sentences)

            if not decision.is_valid:
                # ===== 等同 NO_SPLIT：继续积攒或在硬上限处强切 =====
                reason_brief = "; ".join(decision.reasons[:2]) if decision.reasons else "no reason"
                if buffer_tokens >= self.max_tokens or cursor >= len(all_sentences):
                    chunk_text = " ".join(buffer_sentences).strip()
                    if chunk_text:
                        final_chunks.append(chunk_text)
                        cause = "硬上限" if buffer_tokens >= self.max_tokens else "文末"
                        print(
                            f"⚠️ 校验未通过且达到{cause} ({buffer_tokens} tokens, {reason_brief}) "
                            f"-> 强制归档 Chunk {len(final_chunks)} ({len(buffer_sentences)} 句)"
                        )
                    chunk_start = cursor
                    buffer_tokens = 0
                    next_ask_tokens = self.threshold
                else:
                    increment = max(self.threshold // 2, 300)
                    next_ask_tokens = buffer_tokens + increment
                    print(
                        f"💡 校验未通过 ({reason_brief})，buffer={buffer_tokens} tokens，"
                        f"继续积攒到 {next_ask_tokens} tokens"
                    )
                continue

            # ----- Stage 4：按已校验 ID 切分 -----
            print(
                f"✂️  LLM→ '{raw_response[:60]}'  →  校验通过边界(局部): "
                f"{['S' + str(i) for i in decision.boundary_ids]}"
            )

            # 多 ID 一次切多段（前 N-1 个全部归档，最后一个作为新的 chunk_start）
            prev_local = 0
            for local_id in decision.boundary_ids:
                seg_sents = buffer_sentences[prev_local:local_id]
                seg_text = " ".join(seg_sents).strip()
                if seg_text:
                    final_chunks.append(seg_text)
                    print(
                        f"   ✅ Chunk {len(final_chunks)}: "
                        f"~{self._get_token_count(seg_text)} tokens, {len(seg_sents)} 句"
                    )
                prev_local = local_id

            # 把 chunk_start 推进到最后一个边界对应的全局位置
            new_global_start = chunk_start + decision.boundary_ids[-1]
            chunk_start = new_global_start
            cursor = chunk_start            # 重新从切分点之后开始积攒
            buffer_tokens = 0
            next_ask_tokens = self.threshold

        # ----- 收尾：剩余 buffer 归档 -----
        if chunk_start < len(all_sentences):
            tail_text = " ".join(all_sentences[chunk_start:]).strip()
            if tail_text:
                final_chunks.append(tail_text)
                print(
                    f"📝 收尾 Chunk {len(final_chunks)} "
                    f"(~{self._get_token_count(tail_text)} tokens)"
                )

        if not final_chunks:
            print("⚠️ 未生成任何 chunks，返回原始文本")
            final_chunks = [text]

        print(f"✅ 切分完成，共 {len(final_chunks)} 个 chunks")
        return final_chunks


# =====================================================================
# 测试
# =====================================================================
if __name__ == "__main__":
    sample = "Chapter 1: The Beginning. "
    sample += "This is sentence one. " * 50
    sample += "Chapter 2: The Middle. "
    sample += "This is sentence two. " * 60

    splitter = LLMSemanticChunker(target_chunk_tokens=500)
    chunks = splitter.split(sample)
    for i, c in enumerate(chunks):
        print(f"\nChunk {i + 1}:\n{c[:120]}...")
