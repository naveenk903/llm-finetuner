# Domain LLM Fine‑Tuning

A complete, production-ready framework for fine-tuning large language models with domain-specific knowledge while retaining general capabilities.

## Features

- 🚀 Supports both open-source models (Llama, Mistral) and proprietary APIs (OpenAI, Anthropic)
- 🔧 Flexible fine-tuning methods: LoRA/QLoRA and full SFT
- 📚 Robust data pipeline supporting multiple sources (PDF, HTML, Markdown, databases)
- 📊 Comprehensive evaluation harness
- 🔒 Security best-practices and compliance built-in
- 🐳 Docker-ready with GPU support
- 🔄 CI/CD integration
- 🌐 Pluggable inference (self-hosted or vendor API)

## Quick Start

```bash
# Clone & enter repo
git clone git@github.com:your-org/domain-llm-ft.git
cd domain-llm-ft

# Setup Python environment
poetry install --with dev

# Setup environment variables
cp env.example .env  # edit OPENAI_API_KEY, AWS_PROFILE, etc.

# Start GPU container
make docker-up GPU=1

# Run full pipeline
poetry run dft ingest --sources docs/,kb/#markdown --out data/interim
poetry run dft preprocess --in data/interim --out data/processed/ --qa-ratio 0.3
poetry run dft train \
    model=meta-llama/Llama-2-13b-chat-hf \
    dataset=data/processed/my_corpus.jsonl \
    method=lora
poetry run dft eval --checkpoint outputs/ckpt-latest
poetry run dft serve --checkpoint outputs/ckpt-merged.bin
```

## Project Structure

```
.
├── README.md               # Project documentation
├── pyproject.toml         # Poetry dependency & packaging
├── Makefile              # One-command tasks
├── env.example           # Template for secrets
├── docker/              # Container definitions
├── data/                # Dataset pipeline stages
├── src/                # Main package code
│   └── domain_llm_ft/  # Core modules
└── tests/              # Test suite
```

## Documentation

- [Setup Guide](docs/setup.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Training Guide](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Deployment](docs/deployment.md)
- [API Reference](docs/api.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Apache 2.0 