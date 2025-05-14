"""Unit tests for CLI module."""

from typer.testing import CliRunner
from domain_llm_ft.cli import app
from domain_llm_ft import __version__

runner = CliRunner()

def test_version():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout

def test_ingest_help():
    """Test ingest command help."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "Source paths to ingest" in result.stdout

def test_train_requires_model():
    """Test train command requires model argument."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "Missing option" in result.stdout 