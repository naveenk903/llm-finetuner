"""Question-Answer pair generation utilities."""

import re
from typing import List, Tuple, Optional
import random

from transformers import pipeline
from ..utils.logging import get_logger

logger = get_logger(__name__)

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract key phrases from text using simple heuristics.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Simple scoring based on indicators of importance
    scored_sentences = []
    for sent in sentences:
        score = 0
        # Presence of key phrases
        score += sum(1 for phrase in ["important", "key", "main", "significant", 
                                    "essential", "crucial"] if phrase in sent.lower())
        # Contains numbers or dates
        score += len(re.findall(r'\d+', sent))
        # Contains proper nouns (simple approximation)
        score += len(re.findall(r'[A-Z][a-z]+', sent))
        scored_sentences.append((score, sent))
    
    # Take top scoring sentences
    scored_sentences.sort(reverse=True)
    return [sent for _, sent in scored_sentences[:max_phrases]]

def generate_qa_pairs_heuristic(text: str, num_pairs: int = 3) -> List[Tuple[str, str]]:
    """Generate QA pairs using rule-based heuristics.
    
    Args:
        text: Input text
        num_pairs: Number of QA pairs to generate
        
    Returns:
        List of (question, answer) tuples
    """
    qa_pairs = []
    key_phrases = extract_key_phrases(text)
    
    for phrase in key_phrases[:num_pairs]:
        # Simple question templates
        templates = [
            ("What is", "described as"),
            ("Can you explain", "in the text"),
            ("How would you describe", "mentioned here"),
            ("What does the text say about", ""),
            ("Could you elaborate on", "discussed in this passage")
        ]
        
        template = random.choice(templates)
        question = f"{template[0]} {phrase.strip().rstrip('.')}?"
        answer = phrase + (f" {template[1]}" if template[1] else "")
        
        qa_pairs.append((question, answer))
    
    return qa_pairs

def generate_qa_pairs_llm(
    text: str,
    num_pairs: int = 3,
    model_name: str = "google/flan-t5-base",
    device: str = "cuda",
) -> List[Tuple[str, str]]:
    """Generate QA pairs using a language model.
    
    Args:
        text: Input text
        num_pairs: Number of QA pairs to generate
        model_name: Model to use for generation
        device: Device to run model on
        
    Returns:
        List of (question, answer) tuples
    """
    logger.info(f"Generating QA pairs using {model_name}")
    
    # Initialize question generation pipeline
    qg_pipeline = pipeline(
        "text2text-generation",
        model=model_name,
        device=device,
    )
    
    # Generate questions
    questions = []
    for _ in range(num_pairs):
        prompt = f"Generate a question based on this text: {text}"
        question = qg_pipeline(prompt, max_length=50)[0]["generated_text"]
        questions.append(question)
    
    # Generate answers
    qa_pipeline = pipeline(
        "question-answering",
        model=model_name,
        device=device,
    )
    
    qa_pairs = []
    for question in questions:
        answer = qa_pipeline(question=question, context=text)
        qa_pairs.append((question, answer["answer"]))
    
    return qa_pairs

def generate_qa_pairs(
    text: str,
    num_pairs: int = 3,
    method: str = "heuristic",
    model_name: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Generate QA pairs using specified method.
    
    Args:
        text: Input text
        num_pairs: Number of QA pairs to generate
        method: Generation method ('heuristic' or 'llm')
        model_name: Optional model name for LLM method
        
    Returns:
        List of (question, answer) tuples
    """
    if method == "heuristic":
        return generate_qa_pairs_heuristic(text, num_pairs)
    elif method == "llm":
        if not model_name:
            model_name = "google/flan-t5-base"
        return generate_qa_pairs_llm(text, num_pairs, model_name)
    else:
        raise ValueError(f"Unknown QA generation method: {method}") 