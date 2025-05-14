"""FastAPI server for model inference."""

import json
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

from ..utils.logging import get_logger

logger = get_logger(__name__)

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling probability")
    top_k: int = Field(50, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences to stop on")

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    text: str = Field(..., description="Generated text")
    usage: dict = Field(..., description="Token usage statistics")

def create_app(
    model_path: str,
    device: str = "cuda",
    dtype: str = "float16",
    max_model_len: int = 2048,
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to run model on
        dtype: Model data type
        max_model_len: Maximum sequence length
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Domain-LLM API",
        description="API for domain-specific language model inference",
        version="0.1.0",
    )
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=dtype,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate text from prompt."""
        try:
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop_sequences,
            )
            
            outputs = model.generate(
                [request.prompt],
                sampling_params,
            )
            
            generated_text = outputs[0].outputs[0].text
            
            # Simple token counting
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(generated_text.split())
            
            return GenerateResponse(
                text=generated_text,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app

def serve(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    **kwargs,
) -> None:
    """Serve model via REST API.
    
    Args:
        model_path: Path to model checkpoint
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional arguments for create_app
    """
    app = create_app(model_path, **kwargs)
    uvicorn.run(app, host=host, port=port) 