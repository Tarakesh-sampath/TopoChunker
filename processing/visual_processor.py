import sys
import os
from pathlib import Path
from .base import BaseTopologyProcessor

# Add MinerU to path for local imports
MINERU_PATH = Path(__file__).parent.parent / "utils" / "MinerU"
sys.path.append(str(MINERU_PATH))

try:
    from mineru.cli.common import read_fn
    import importlib.util
    # Dynamic import of do_parse from MinerU demo
    demo_path = MINERU_PATH / "demo" / "demo.py"
    spec = importlib.util.spec_from_file_location("mineru_demo", demo_path)
    mineru_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mineru_demo)
    do_parse = mineru_demo.do_parse
except Exception as e:
    print(f"⚠️ Warning: MinerU initialization failed: {e}")
    do_parse = None

class VisualTopologyProcessor(BaseTopologyProcessor):
    """
    Path 3: Visual Topology Path.
    Powerhouse for scanned or complex documents using VLM.
    """
    def process(self, file_path: str) -> str:
        if do_parse is None:
            return "Error: MinerU components missing."
        
        pdf_path = Path(file_path)
        pdf_name = pdf_path.stem
        
        # Create a dedicated folder for this file in outputs
        output_dir = Path("outputs") / pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # MinerU specific output sub-dir
        mineru_output_dir = output_dir / "visual_parsed"
        mineru_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🖼️ [Path 3] Running Visual (MinerU) on: {pdf_path.name}")
        
        try:
            # Execute MinerU VLM parsing
            do_parse(
                str(mineru_output_dir), 
                [pdf_path.stem], 
                [read_fn(str(pdf_path))], 
                ["ch"],
                backend='vlm-transformers'
            )
            
            # Retrieve the result
            md_file = mineru_output_dir / pdf_path.stem / f"{pdf_path.stem}.md"
            if md_file.exists():
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Save a copy in the main folder as final
                    with open(output_dir / f"{pdf_name}_final.md", "w", encoding="utf-8") as f2:
                        f2.write(content)
                    return content
            return "Error: MinerU failed to produce output."
        except Exception as e:
            return f"Error in Visual Processing: {str(e)}"
