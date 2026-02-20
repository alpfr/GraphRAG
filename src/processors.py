import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HierarchicalChunker
    DOCLING_AVAILABLE = True
except ImportError:
    logger.warning("Docling is not installed. Documents will be parsed as raw text.")
    DOCLING_AVAILABLE = False

from config.settings import settings

class DoclingProcessor:
    """Processes unstructured documents into structured JSON and chunks using Docling."""

    def __init__(self):
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()
            self.chunker = HierarchicalChunker()
        logger.info("DoclingProcessor initialized")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Processes a document (PDF, DOCX, PPTX, etc.) into a structured format
        with metadata and chunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing document: {path.name}")
        
        # Determine fallback parsing vs docling
        if DOCLING_AVAILABLE and path.suffix.lower() in ['.pdf', '.docx', '.pptx', '.html', '.md']:
            doc_result = self.converter.convert(str(path))
            document = doc_result.document
            
            # Generate chunks
            chunks_iter = self.chunker.chunk(document)
            chunks = []
            for i, chunk in enumerate(chunks_iter):
                chunks.append({
                    "chunk_id": i,
                    "text": chunk.text,
                    "metadata": {
                        "heading": chunk.meta.headings[0] if chunk.meta.headings else "",
                        "page": getattr(chunk.meta, 'page', 1)
                    }
                })
        else:
            # Fallback simple text parser
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Simple fixed-size chunking
            chunks = []
            words = text.split()
            chunk_size = settings.chunk_size
            overlap = settings.chunk_overlap
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if not chunk_words:
                    break
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": " ".join(chunk_words),
                    "metadata": {
                        "heading": "",
                        "page": 1
                    }
                })

        # Assemble the final document data
        doc_data = {
            "file_name": path.name,
            "processed_at": datetime.utcnow().isoformat(),
            "metadata": {
                "file_size": path.stat().st_size,
                "file_extension": path.suffix.lower()
            },
            "chunks": chunks
        }
        
        logger.info(f"Successfully processed {path.name} into {len(chunks)} chunks")
        return doc_data

    def save_output(self, doc_data: Dict[str, Any], output_dir: str) -> str:
        """Saves the processed document data to a JSON file."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        file_name = Path(doc_data["file_name"]).stem
        output_file = out_path / f"{file_name}_processed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2)
            
        logger.info(f"Saved processed output to {output_file}")
        return str(output_file)
