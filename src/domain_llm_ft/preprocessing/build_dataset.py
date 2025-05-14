"""Dataset building and preprocessing utilities."""

import json
import random
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .qa_gen import generate_qa_pairs
from ..utils.logging import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """Clean raw text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove multiple spaces
    text = re.sub(r" +", " ", text)
    
    # Fix common unicode issues
    text = text.replace("'", "'").replace(""", '"').replace(""", '"')
    
    return text.strip()

def segment(
    text: str,
    max_tokens: int = 1024,
    overlap: int = 128,
    split_on: List[str] = ["\n\n", "\n", ". "]
) -> List[str]:
    """Split text into overlapping segments.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per segment (approximate)
        overlap: Number of tokens to overlap between segments
        split_on: Delimiters to try splitting on, in order of preference
        
    Returns:
        List of text segments
    """
    # Simple token count approximation
    def token_count(s: str) -> int:
        return len(s.split())
    
    segments = []
    current_segment = ""
    current_tokens = 0
    
    # Try each delimiter in order
    for delimiter in split_on:
        chunks = text.split(delimiter)
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            chunk_tokens = token_count(chunk)
            
            if current_tokens + chunk_tokens <= max_tokens:
                current_segment += chunk + delimiter
                current_tokens += chunk_tokens
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = chunk + delimiter
                current_tokens = chunk_tokens
                
    if current_segment:
        segments.append(current_segment.strip())
        
    # Add overlap between segments
    final_segments = []
    for i, seg in enumerate(segments):
        if i > 0:
            # Add end of previous segment
            prev_tokens = segments[i-1].split()[-overlap:]
            seg = " ".join(prev_tokens) + " " + seg
        if i < len(segments) - 1:
            # Add start of next segment
            next_tokens = segments[i+1].split()[:overlap]
            seg = seg + " " + " ".join(next_tokens)
        final_segments.append(seg)
        
    return final_segments

def build_dataset(
    documents: Iterable[Dict[str, str]],
    output_path: Path,
    max_tokens: int = 1024,
    qa_ratio: float = 0.3,
    qa_method: str = "heuristic",
    qa_model: Optional[str] = None,
    qa_pairs_per_segment: int = 3,
) -> None:
    """Build training dataset from documents.
    
    Args:
        documents: Iterator of document dicts with text and metadata
        output_path: Path to save dataset
        max_tokens: Maximum tokens per example
        qa_ratio: Ratio of examples to convert to QA format
        qa_method: Method for QA generation ('heuristic' or 'llm')
        qa_model: Optional model name for LLM-based QA generation
        qa_pairs_per_segment: Number of QA pairs to generate per segment
    """
    logger.info(f"Building dataset, saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for doc in documents:
            text = clean_text(doc["text"])
            segments = segment(text, max_tokens=max_tokens)
            
            for seg in segments:
                if random.random() < qa_ratio:
                    # Generate QA pairs
                    qa_pairs = generate_qa_pairs(
                        seg,
                        num_pairs=qa_pairs_per_segment,
                        method=qa_method,
                        model_name=qa_model,
                    )
                    
                    for question, answer in qa_pairs:
                        qa_example = {
                            "id": str(uuid.uuid4()),
                            "question": question,
                            "answer": answer,
                            "metadata": {
                                **doc.get("metadata", {}),
                                "qa_method": qa_method,
                                "original_text": seg,
                            }
                        }
                        f.write(json.dumps(qa_example) + "\n")
                else:
                    # Basic example format
                    example = {
                        "id": str(uuid.uuid4()),
                        "text": seg,
                        "metadata": doc.get("metadata", {})
                    }
                    f.write(json.dumps(example) + "\n")
                
    logger.info("Dataset build complete")