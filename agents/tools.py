"""Tools for Inspector Agent - exposed to LLM for autonomous tool calling."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

from .probes import LayoutProbe, MetaReader, RegexTester

# Lazy import for PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

# Initialize probe instances (shared across tool calls)
_meta_reader = MetaReader()
_layout_probe = LayoutProbe()
_regex_tester = RegexTester()


@tool
def analyze_metadata(file_path: str) -> Dict[str, Any]:
    """Analyze PDF metadata and check if document is scanned.
    
    This tool extracts basic metadata (page count, author) and determines
    if the PDF is scanned (no text layer).
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with metadata including:
        - page_count: int
        - author: str or None
        - is_scanned: bool (True if scanned PDF)
        - metadata: dict (raw PDF metadata)
    """
    # Check if verbose mode is enabled (passed via context)
    verbose = getattr(analyze_metadata, "_verbose", False)
    return _meta_reader.analyze(file_path, verbose=verbose)


@tool
def probe_layout_complexity(file_path: str) -> Dict[str, Any]:
    """Estimate layout complexity of the document.
    
    This tool samples pages from the document and analyzes text block
    distribution to estimate layout complexity (multi-column, tables, etc.).
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with:
        - complexity_score: float (0.0 to 1.0, higher = more complex)
        - page_count: int (total pages)
    """
    if fitz is None:
        return {
            "complexity_score": 1.0,
            "error": "PyMuPDF not installed",
        }
    try:
        with fitz.open(file_path) as doc:
            # LayoutProbe.probe() doesn't support verbose parameter
            score = _layout_probe.probe(doc)
            return {
                "complexity_score": score,
                "page_count": doc.page_count,
            }
    except Exception as exc:
        return {
            "complexity_score": 1.0,
            "error": str(exc),
        }


@tool
def test_rule_adherence(file_path: str) -> Dict[str, Any]:
    """Check if document follows standard structural patterns.
    
    This tool tests if the document has structured headings like:
    - Numbered sections (1., 1.1, etc.)
    - Chapter/Article patterns
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with:
        - rule_adherence_score: float (0.0 to 1.0, higher = more structured)
        - page_count: int (total pages)
    """
    if fitz is None:
        return {
            "rule_adherence_score": 0.0,
            "error": "PyMuPDF not installed",
        }
    try:
        verbose = getattr(test_rule_adherence, "_verbose", False)
        with fitz.open(file_path) as doc:
            score = _regex_tester.test_adherence(doc, verbose=verbose)
            return {
                "rule_adherence_score": score,
                "page_count": doc.page_count,
            }
    except Exception as exc:
        return {
            "rule_adherence_score": 0.0,
            "error": str(exc),
        }


def get_inspector_tools() -> list:
    """Get all tools available to the Inspector Agent.
    
    Returns:
        List of Tool instances that LLM can call
    """
    return [
        analyze_metadata,
        probe_layout_complexity,
        test_rule_adherence,
    ]
