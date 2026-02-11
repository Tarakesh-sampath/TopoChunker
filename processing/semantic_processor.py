import re
import json
import torch
import gc
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Union
from transformers import LongformerForTokenClassification, LongformerTokenizerFast
from langchain_core.language_models import BaseChatModel
from .base import BaseTopologyProcessor

class LongformerSemanticSplitter:
    """
    Revised Semantic Splitter implementing HiChunk-style Iterative Inference.
    Features:
    - Global Context Awareness
    - Strict Token-level Length Control
    - Anti-Stagnation Logic (No Token Left Behind)
    """
    
    def __init__(
        self, 
        model_path: str = "/root/liushifu/TopoChunker/models/longformer-segmenter-finetuned/checkpoint-3159",
        target_chunk_size: int = 500,
        max_chunk_size: int = 512,
        threshold: float = 0.5
    ):
        print(f"🔄 Loading Longformer from {model_path}...")
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model_path)
        self.model = LongformerForTokenClassification.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.threshold = threshold
        
        # Buffer for CLS/SEP
        raw_max_len = self.tokenizer.model_max_length
        self.model_max_length = min(raw_max_len, 4096) - 2
        
        print(f"✅ Loaded on {self.device} | Window: {self.model_max_length}")

    def split(self, text_input: Union[str, List[str]]) -> List[str]:
        # 1. Pre-process
        if isinstance(text_input, list):
            text = " ".join(text_input)
        else:
            text = text_input
        
        if not text.strip():
            return []

        # 2. Global Tokenization (CPU Tensor to save GPU memory)
        print("🔠 Tokenizing document...")
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, verbose=False)
        all_input_ids = inputs["input_ids"][0]
        
        total_tokens = len(all_input_ids)
        print(f"📄 Total tokens: {total_tokens}")

        # 3. Inference Strategy
        if total_tokens <= self.model_max_length:
            return self._process_window(all_input_ids)
        
        return self._iterative_inference(all_input_ids)

    def _iterative_inference(self, all_input_ids: torch.Tensor) -> List[str]:
        final_chunks = []
        start_idx = 0
        total_length = len(all_input_ids)

        print(f"🚀 Starting iterative inference...")

        while start_idx < total_length:
            # [Memory Optimization] Calculate window bounds
            window_end = min(start_idx + self.model_max_length, total_length)
            window_ids = all_input_ids[start_idx:window_end]

            # Model Inference
            _, split_probs = self._run_model_inference(window_ids)
            valid_probs = split_probs[1:-1] # Align with original content

            # --- Smart Split Strategy ---
            search_limit = min(self.max_chunk_size, len(valid_probs))
            
            # Find candidates within the allowed max_chunk_size
            candidates = [
                i for i, p in enumerate(valid_probs[:search_limit]) 
                if p > self.threshold
            ]

            cut_relative_idx = -1

            if candidates:
                # Filter noise (<50 tokens) unless it's the only choice
                reasonable = [c for c in candidates if c > 50]
                if reasonable:
                    cut_relative_idx = reasonable[-1] # Greedy: Longest valid chunk
                else:
                    cut_relative_idx = candidates[-1] # Fallback
            
            # --- Decision Execution ---
            if cut_relative_idx != -1:
                cut_abs_length = cut_relative_idx + 1
            else:
                # Hard Cut Fallback
                cut_abs_length = search_limit
            
            # [Fix: Dead Loop / Zero Progress Protection]
            # 如果计算出的长度为0 (极罕见)，强制截取1个token，防止死循环且不丢数据
            if cut_abs_length <= 0:
                print(f"⚠️ Zero progress at {start_idx}, forcing 1 token step.")
                cut_abs_length = 1

            # Execute Slice
            cut_abs_global = start_idx + cut_abs_length
            chunk_ids = all_input_ids[start_idx:cut_abs_global]
            
            # Decode
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if chunk_text:
                final_chunks.append(chunk_text)
                
            # [Fix: Log Position] Log AFTER update
            if len(final_chunks) % 10 == 0:
                 print(f"   -> Generated {len(final_chunks)} chunks... (Pos: {cut_abs_global}/{total_length})")

            # Update Index
            start_idx = cut_abs_global

            # [Memory] Periodic Cleanup
            if len(final_chunks) % 50 == 0:
                torch.cuda.empty_cache()

        # Post-process
        return self._post_merge_chunks(final_chunks)

    def _run_model_inference(self, window_ids: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        # Construct [CLS] + Content + [SEP]
        cls_t = torch.tensor([self.tokenizer.cls_token_id], device=self.device)
        sep_t = torch.tensor([self.tokenizer.sep_token_id], device=self.device)
        
        # Move only the current window to GPU
        input_ids = torch.cat([cls_t, window_ids.to(self.device), sep_t])

        # Standard Longformer Masking
        attention_mask = torch.ones_like(input_ids)
        global_mask = torch.zeros_like(input_ids)
        global_mask[0] = 1 

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                global_attention_mask=global_mask.unsqueeze(0)
            )
            probs = torch.softmax(outputs.logits, dim=2)[0]
            # Move to CPU immediately to free VRAM
            split_probs = probs[:, 1].cpu().numpy()

        return input_ids, split_probs

    def _process_window(self, raw_token_ids: torch.Tensor) -> List[str]:
        """Simple greedy split for short texts."""
        input_ids, split_probs = self._run_model_inference(raw_token_ids)
        chunks = []
        start = 1
        current_count = 0

        # Loop range: 1 to len-1 (Excluding CLS and SEP)
        for i in range(1, len(input_ids) - 1):
            current_count += 1
            prob = split_probs[i]
            
            should_split = (prob > self.threshold and current_count >= self.target_chunk_size) or \
                           (current_count >= self.max_chunk_size)

            if should_split:
                chunk_ids = input_ids[start:i+1]
                text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
                if text: chunks.append(text)
                start = i + 1
                current_count = 0
        
        # Trailing
        if start < len(input_ids) - 1:
            chunk_ids = input_ids[start:-1]
            text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if text: chunks.append(text)

        return self._post_merge_chunks(chunks)

    def _post_merge_chunks(self, chunks: List[str]) -> List[str]:
        """
        [Fix] Merge using accurate Token Count instead of Char Length.
        """
        if not chunks: return []
        
        merged = [chunks[0]]
        
        # Cache token counts for the last merged chunk to avoid re-tokenizing repeatedly
        current_token_count = self._count_tokens(merged[-1])

        for next_chunk in chunks[1:]:
            next_chunk_tokens = self._count_tokens(next_chunk)
            
            # Check if merging keeps us under the target (soft limit)
            if (current_token_count + next_chunk_tokens) <= self.target_chunk_size:
                merged[-1] += " " + next_chunk
                current_token_count += next_chunk_tokens # Add approx overhead for space? Negligible.
            else:
                merged.append(next_chunk)
                current_token_count = next_chunk_tokens
                
        return merged

    def _count_tokens(self, text: str) -> int:
        """Helper for accurate length check."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

class SemanticFlowProcessor(BaseTopologyProcessor):
    def __init__(self, llm_client: BaseChatModel):
        self.splitter = LongformerSemanticSplitter()

    def process(self, file_path: str) -> str:
        pdf_name = Path(file_path).stem
        output_dir = Path("outputs") / pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📖 Extracting sentences from: {file_path}")
        sentences = self._extract_sentences(file_path)
        
        if not sentences:
            return "No text content found."

        print(f"🧠 Running Chunking...")
        final_chunks = self.splitter.split(sentences)
        
        # Output
        with open(output_dir / "longformer_chunks.json", "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)

        md_content = "\n\n".join(final_chunks)
        with open(output_dir / f"{pdf_name}_final.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_content

    def _extract_sentences(self, file_path: str) -> List[str]:
        """
        [Fix] Enhanced sentence splitting and PDF extraction.
        """
        full_text_buffer = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    blocks = page.get_text("blocks")
                    # Sort: Top-to-bottom, Left-to-right
                    blocks.sort(key=lambda b: (b[1], b[0]))

                    for b in blocks:
                        if b[6] != 0: continue # Skip images
                        text = b[4].strip()
                        if not text: continue
                        
                        # Filter standalone page numbers
                        if text.isdigit() and len(text) < 4: continue

                        # Fix hyphenation and join lines
                        clean_text = text.replace("-\n", "").replace("\n", " ")
                        full_text_buffer.append(clean_text)
            
            all_text = " ".join(full_text_buffer)
            
            # [Fix] Better Regex for Sentence Splitting
            # 1. Negative lookbehind for Mr./Dr./Fig. (Basic list)
            # 2. Match .!? followed by space
            # Note: For production, consider using 'pysbd' library.
            split_pattern = r'(?<!Mr)(?<!Dr)(?<!Ms)(?<!Fig)(?<!Vol)(?<!No)(?<=[.!?])\s+'
            
            sentence_list = re.split(split_pattern, all_text)
            
            return [s.strip() for s in sentence_list if len(s.strip()) > 5]
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return []