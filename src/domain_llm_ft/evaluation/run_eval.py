"""Evaluation runner for model assessment."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import evaluate_model_outputs
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """Model evaluation coordinator."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Optional separate tokenizer path
            device: Device to run evaluation on
        """
        self.model_path = str(model_path)
        self.tokenizer_path = str(tokenizer_path) if tokenizer_path else self.model_path
        self.device = device
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate text from prompts.
        
        Args:
            prompts: Input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of sequences per prompt
            
        Returns:
            List of generated texts
        """
        outputs = []
        
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            decoded = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            outputs.extend(decoded)
            
        return outputs
        
    def evaluate_dataset(
        self,
        dataset_path: Union[str, Path],
        task_type: str = "text",
        domain_keywords: Optional[List[str]] = None,
        toxic_keywords: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            dataset_path: Path to evaluation dataset
            task_type: Type of task ('text' or 'qa')
            domain_keywords: Optional domain-specific keywords
            toxic_keywords: Optional toxic keywords to check
            output_path: Optional path to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset("json", data_files=str(dataset_path))["train"]
        
        if task_type == "qa":
            # Generate answers for questions
            questions = dataset["question"]
            predictions = self.generate(questions)
            references = dataset["answer"]
            
            results = evaluate_model_outputs(
                predictions=predictions,
                references=references,
                task_type="qa",
                questions=questions,
                domain_keywords=domain_keywords,
                toxic_keywords=toxic_keywords,
                tokenizer=self.tokenizer,
            )
            
        else:
            # Generate continuations for prompts
            prompts = dataset["text"]
            predictions = self.generate(prompts)
            references = dataset["text"]  # Using input as reference
            
            results = evaluate_model_outputs(
                predictions=predictions,
                references=references,
                task_type="text",
                domain_keywords=domain_keywords,
                toxic_keywords=toxic_keywords,
                tokenizer=self.tokenizer,
            )
        
        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(output_path / "metrics.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Save predictions
            predictions_df = pd.DataFrame({
                "input": questions if task_type == "qa" else prompts,
                "prediction": predictions,
                "reference": references,
            })
            predictions_df.to_csv(output_path / "predictions.csv", index=False)
            
            # Generate visualizations
            self._generate_report(
                results=results,
                predictions_df=predictions_df,
                output_path=output_path,
            )
            
        return results
    
    def _generate_report(
        self,
        results: Dict[str, float],
        predictions_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Generate evaluation report with visualizations.
        
        Args:
            results: Evaluation metrics
            predictions_df: DataFrame with predictions
            output_path: Path to save report
        """
        # Metrics summary plot
        metrics_df = pd.DataFrame([
            {"metric": k, "value": v}
            for k, v in results.items()
        ])
        
        fig = px.bar(
            metrics_df,
            x="metric",
            y="value",
            title="Evaluation Metrics",
        )
        fig.write_html(output_path / "metrics.html")
        
        # Length distributions
        pred_lengths = predictions_df["prediction"].str.len()
        ref_lengths = predictions_df["reference"].str.len()
        
        fig = px.histogram(
            pd.DataFrame({
                "Predictions": pred_lengths,
                "References": ref_lengths,
            }).melt(),
            x="value",
            color="variable",
            title="Output Length Distribution",
            labels={"value": "Length (chars)"},
        )
        fig.write_html(output_path / "lengths.html")
        
        # Generate HTML report
        with open(output_path / "report.html", "w") as f:
            f.write("""
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-5">
                    <h1>Model Evaluation Report</h1>
                    <hr>
                    
                    <h2>Metrics Summary</h2>
                    <iframe src="metrics.html" width="100%" height="600px"></iframe>
                    
                    <h2>Output Analysis</h2>
                    <iframe src="lengths.html" width="100%" height="600px"></iframe>
                    
                    <h2>Sample Outputs</h2>
                    <div class="table-responsive">
            """)
            
            # Add sample predictions table
            f.write(predictions_df.head(10).to_html(classes="table table-striped"))
            
            f.write("""
                    </div>
                </div>
            </body>
            </html>
            """)
            
        logger.info(f"Evaluation report saved to {output_path}/report.html") 