"""Evaluation metrics for model assessment."""

from typing import Dict, List, Optional, Union
import numpy as np
from datasets import load_metric
from transformers import PreTrainedTokenizer
import torch
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MetricCollection:
    """Collection of evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.metrics = {
            "rouge": load_metric("rouge"),
            "bleu": load_metric("bleu"),
            "bertscore": load_metric("bertscore"),
        }
        
    def compute_text_metrics(
        self,
        predictions: List[str],
        references: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dict[str, float]:
        """Compute metrics for text generation.
        
        Args:
            predictions: Model generated texts
            references: Ground truth texts
            tokenizer: Optional tokenizer for token-based metrics
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # ROUGE scores
        rouge_output = self.metrics["rouge"].compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        results.update({
            "rouge1": rouge_output["rouge1"].mid.fmeasure,
            "rouge2": rouge_output["rouge2"].mid.fmeasure,
            "rougeL": rouge_output["rougeL"].mid.fmeasure,
        })
        
        # BLEU score
        if tokenizer:
            tokenized_preds = [tokenizer.tokenize(p) for p in predictions]
            tokenized_refs = [tokenizer.tokenize(r) for r in references]
            bleu_score = self.metrics["bleu"].compute(
                predictions=tokenized_preds,
                references=[[r] for r in tokenized_refs],
            )
            results["bleu"] = bleu_score["bleu"]
        
        # BERTScore
        bertscore_output = self.metrics["bertscore"].compute(
            predictions=predictions,
            references=references,
            lang="en",
        )
        results.update({
            "bertscore_precision": np.mean(bertscore_output["precision"]),
            "bertscore_recall": np.mean(bertscore_output["recall"]),
            "bertscore_f1": np.mean(bertscore_output["f1"]),
        })
        
        return results

    def compute_qa_metrics(
        self,
        predictions: List[str],
        references: List[str],
        questions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute metrics for question answering.
        
        Args:
            predictions: Model generated answers
            references: Ground truth answers
            questions: Optional list of questions for context
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # Exact match score
        exact_matches = [pred.strip() == ref.strip() 
                        for pred, ref in zip(predictions, references)]
        results["exact_match"] = np.mean(exact_matches)
        
        # F1 score based on token overlap
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if not ref_tokens:
                continue
                
            common_tokens = pred_tokens & ref_tokens
            if not pred_tokens:
                precision = 0.0
            else:
                precision = len(common_tokens) / len(pred_tokens)
            
            recall = len(common_tokens) / len(ref_tokens)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
            
        results["f1"] = np.mean(f1_scores) if f1_scores else 0.0
        
        # Also compute text metrics
        text_metrics = self.compute_text_metrics(predictions, references)
        results.update({f"qa_{k}": v for k, v in text_metrics.items()})
        
        return results

    def compute_domain_metrics(
        self,
        predictions: List[str],
        domain_keywords: List[str],
        toxic_keywords: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute domain-specific metrics.
        
        Args:
            predictions: Model generated texts
            domain_keywords: List of domain-specific keywords/phrases
            toxic_keywords: Optional list of words/phrases to avoid
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # Domain relevance score
        domain_scores = []
        for pred in predictions:
            pred_lower = pred.lower()
            matches = sum(1 for kw in domain_keywords 
                        if kw.lower() in pred_lower)
            domain_scores.append(matches / len(domain_keywords))
        results["domain_relevance"] = np.mean(domain_scores)
        
        # Toxicity score (if keywords provided)
        if toxic_keywords:
            toxic_scores = []
            for pred in predictions:
                pred_lower = pred.lower()
                matches = sum(1 for kw in toxic_keywords 
                            if kw.lower() in pred_lower)
                toxic_scores.append(matches / len(toxic_keywords))
            results["toxicity"] = np.mean(toxic_scores)
        
        return results

def evaluate_model_outputs(
    predictions: List[str],
    references: List[str],
    task_type: str = "text",
    questions: Optional[List[str]] = None,
    domain_keywords: Optional[List[str]] = None,
    toxic_keywords: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> Dict[str, float]:
    """Evaluate model outputs using appropriate metrics.
    
    Args:
        predictions: Model generated texts
        references: Ground truth texts
        task_type: Type of task ('text' or 'qa')
        questions: Questions for QA evaluation
        domain_keywords: Domain-specific keywords
        toxic_keywords: Words/phrases to avoid
        tokenizer: Optional tokenizer for token-based metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = MetricCollection()
    results = {}
    
    # Basic text metrics
    if task_type == "text":
        results.update(
            metrics.compute_text_metrics(
                predictions=predictions,
                references=references,
                tokenizer=tokenizer,
            )
        )
    
    # QA metrics
    elif task_type == "qa":
        results.update(
            metrics.compute_qa_metrics(
                predictions=predictions,
                references=references,
                questions=questions,
            )
        )
    
    # Domain metrics (if keywords provided)
    if domain_keywords:
        results.update(
            metrics.compute_domain_metrics(
                predictions=predictions,
                domain_keywords=domain_keywords,
                toxic_keywords=toxic_keywords,
            )
        )
    
    return results 