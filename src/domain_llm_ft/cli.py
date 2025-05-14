"""Command-line interface for domain-llm-ft."""

import typer
from pathlib import Path
from rich import print
from typing import List, Optional

from . import __version__
from .utils.logging import get_logger

app = typer.Typer(
    name="dft",
    help="Domain-specific LLM fine-tuning toolkit",
    no_args_is_help=True,
)
logger = get_logger(__name__)

@app.command()
def ingest(
    sources: List[str] = typer.Argument(
        ..., help="Source paths to ingest (comma-separated)"
    ),
    out: Path = typer.Option(
        "data/interim",
        "--out",
        "-o",
        help="Output directory for interim data",
    ),
):
    """Ingest raw data from various sources."""
    logger.info(f"Ingesting data from {sources} to {out}")
    # TODO: Implement ingestion logic
    
@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        "data/interim",
        "--in",
        "-i",
        help="Input directory with interim data",
    ),
    output_dir: Path = typer.Option(
        "data/processed",
        "--out",
        "-o",
        help="Output directory for processed data",
    ),
    qa_ratio: float = typer.Option(
        0.3,
        "--qa-ratio",
        "-q",
        help="Ratio of QA pairs to generate",
    ),
):
    """Preprocess interim data into training format."""
    logger.info(f"Preprocessing data from {input_dir} to {output_dir}")
    # TODO: Implement preprocessing logic

@app.command()
def train(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Base model to fine-tune",
    ),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to processed dataset",
    ),
    method: str = typer.Option(
        "lora",
        "--method",
        help="Fine-tuning method (lora/qlora/sft)",
    ),
):
    """Fine-tune model on processed data."""
    logger.info(f"Training {model} using {method} on {dataset}")
    # TODO: Implement training logic

@app.command()
def eval(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
    ),
    report: Optional[Path] = typer.Option(
        None,
        "--report",
        "-r",
        help="Path to save evaluation report",
    ),
):
    """Evaluate fine-tuned model."""
    logger.info(f"Evaluating checkpoint {checkpoint}")
    # TODO: Implement evaluation logic

@app.command()
def serve(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
):
    """Serve fine-tuned model via REST API."""
    logger.info(f"Serving {checkpoint} on {host}:{port}")
    # TODO: Implement serving logic

@app.command()
def version():
    """Show version information."""
    print(f"domain-llm-ft version {__version__}")

if __name__ == "__main__":
    app() 