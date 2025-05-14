# Contributing to Domain LLM Fine-Tuning

Thank you for your interest in contributing to the Domain LLM Fine-Tuning project! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone git@github.com:your-username/domain-llm-ft.git
   cd domain-llm-ft
   ```

2. Set up the development environment:
   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install --with dev

   # Install pre-commit hooks
   poetry run pre-commit install
   ```

3. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- We use Black for code formatting with a line length of 100
- Import sorting is handled by isort
- Code quality is checked by ruff
- Security checks are performed by Bandit
- All checks are enforced by pre-commit hooks

### Testing

- Write tests for new features using pytest
- Maintain test coverage above 80%
- Run tests locally before submitting:
  ```bash
  poetry run pytest tests/
  ```

### Documentation

- Document all public functions, classes, and modules
- Follow Google style docstrings
- Update README.md if adding new features
- Add examples to docstrings where helpful

### Commits

- Use clear, descriptive commit messages
- Follow conventional commits format:
  - feat: New feature
  - fix: Bug fix
  - docs: Documentation changes
  - style: Code style changes
  - refactor: Code refactoring
  - test: Test updates
  - chore: Maintenance tasks

## Pull Request Process

1. Update documentation for any new features
2. Add or update tests as needed
3. Ensure all checks pass in CI
4. Update CHANGELOG.md following Keep a Changelog format
5. Request review from maintainers

## Running the Full Pipeline

Test your changes with the complete pipeline:

```bash
# Ingest sample data
poetry run dft ingest --sources examples/data --out data/interim

# Preprocess
poetry run dft preprocess --in data/interim --out data/processed

# Train (if you have GPU access)
poetry run dft train \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset data/processed/dataset.jsonl \
    --method lora

# Evaluate
poetry run dft eval --checkpoint outputs/ckpt-latest
```

## Getting Help

- Open an issue for bugs or feature requests
- Join our community discussions
- Tag maintainers for urgent issues

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed under the project's Apache 2.0 License. 