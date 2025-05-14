"""Data loaders for various source formats."""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from urllib.parse import urlparse

import fitz  # PyMuPDF
import markdown
import requests
from bs4 import BeautifulSoup
from rich.progress import track

from ..utils.logging import get_logger

logger = get_logger(__name__)

class BaseLoader(ABC):
    """Base class for all data loaders."""
    
    @abstractmethod
    def load(self, path: Path) -> Iterable[Dict[str, str]]:
        """Load data from path.
        
        Args:
            path: Path to load data from
            
        Returns:
            Iterator of documents with metadata
        """
        pass

class PDFLoader(BaseLoader):
    """Load text from PDF files."""
    
    def load(self, path: Path) -> Iterable[Dict[str, str]]:
        """Extract text from PDF maintaining structure."""
        logger.info(f"Loading PDF: {path}")
        doc = fitz.open(path)
        
        for page_num in track(range(len(doc)), description=f"Processing {path.name}"):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():  # Skip empty pages
                yield {
                    "text": text,
                    "metadata": {
                        "source": str(path),
                        "page": page_num + 1,
                        "type": "pdf"
                    }
                }
        doc.close()

class MarkdownLoader(BaseLoader):
    """Load text from Markdown files."""
    
    def load(self, path: Path) -> Iterable[Dict[str, str]]:
        """Convert Markdown to text preserving structure."""
        logger.info(f"Loading Markdown: {path}")
        
        with open(path) as f:
            md_text = f.read()
            
        # Convert to HTML then extract clean text
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract text by sections (h1/h2/h3 headers)
        sections = []
        current_section = {"title": "", "content": []}
        
        for elem in soup.find_all(["h1", "h2", "h3", "p"]):
            if elem.name in ["h1", "h2", "h3"]:
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {
                    "title": elem.get_text(),
                    "content": []
                }
            else:
                current_section["content"].append(elem.get_text())
        
        if current_section["content"]:
            sections.append(current_section)
            
        # Yield each section
        for section in sections:
            yield {
                "text": "\n".join([section["title"]] + section["content"]),
                "metadata": {
                    "source": str(path),
                    "title": section["title"],
                    "type": "markdown"
                }
            }

class HTMLLoader(BaseLoader):
    """Load text from HTML files or URLs."""
    
    def __init__(self, min_text_length: int = 50):
        """Initialize HTML loader.
        
        Args:
            min_text_length: Minimum text length to consider a block
        """
        self.min_text_length = min_text_length
    
    def load(self, path: Union[str, Path]) -> Iterable[Dict[str, str]]:
        """Extract text from HTML maintaining structure."""
        logger.info(f"Loading HTML: {path}")
        
        # Handle URLs vs local files
        if isinstance(path, str) and urlparse(path).scheme in ['http', 'https']:
            response = requests.get(path)
            response.raise_for_status()
            html = response.text
            source = path
        else:
            with open(path) as f:
                html = f.read()
            source = str(path)
            
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Extract text by sections
        for section in self._extract_sections(soup):
            if len(section["text"]) >= self.min_text_length:
                yield {
                    "text": section["text"],
                    "metadata": {
                        "source": source,
                        "title": section.get("title", ""),
                        "type": "html",
                        "url": path if isinstance(path, str) else None
                    }
                }
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract meaningful text sections from HTML."""
        sections = []
        
        # Main content areas
        for tag in ["article", "main", "div", "section"]:
            for element in soup.find_all(tag):
                # Check if element has substantial unique text
                text = element.get_text(separator=" ", strip=True)
                if len(text) >= self.min_text_length:
                    title = ""
                    # Look for a heading
                    heading = element.find(["h1", "h2", "h3"])
                    if heading:
                        title = heading.get_text(strip=True)
                    sections.append({"title": title, "text": text})
                    
        return sections

class SQLLoader(BaseLoader):
    """Load text from SQL databases."""
    
    def __init__(
        self,
        query: str,
        text_columns: List[str],
        metadata_columns: Optional[List[str]] = None
    ):
        """Initialize SQL loader.
        
        Args:
            query: SQL query to execute
            text_columns: Columns to combine as text
            metadata_columns: Optional columns to include as metadata
        """
        self.query = query
        self.text_columns = text_columns
        self.metadata_columns = metadata_columns or []
    
    def load(self, path: str) -> Iterable[Dict[str, str]]:
        """Extract text from database.
        
        Args:
            path: Database connection string or path
        """
        logger.info(f"Loading from database: {path}")
        
        conn = sqlite3.connect(path)
        try:
            cursor = conn.cursor()
            cursor.execute(self.query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor:
                row_dict = dict(zip(columns, row))
                
                # Combine text columns
                text_parts = []
                for col in self.text_columns:
                    if col in row_dict and row_dict[col]:
                        text_parts.append(str(row_dict[col]))
                
                if text_parts:
                    # Extract metadata
                    metadata = {
                        "source": path,
                        "type": "sql",
                    }
                    for col in self.metadata_columns:
                        if col in row_dict:
                            metadata[col] = row_dict[col]
                            
                    yield {
                        "text": "\n".join(text_parts),
                        "metadata": metadata
                    }
                    
        finally:
            conn.close()

def load_documents(
    paths: Union[str, List[str], Path, List[Path]],
    sql_config: Optional[Dict] = None,
) -> Iterable[Dict[str, str]]:
    """Load documents from multiple paths.
    
    Args:
        paths: Single path or list of paths to load
        sql_config: Optional SQL loader configuration
        
    Returns:
        Iterator of documents with metadata
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) if not isinstance(p, str) or urlparse(p).scheme == '' else p 
            for p in paths]
    
    # Map extensions to loaders
    loaders = {
        ".pdf": PDFLoader(),
        ".md": MarkdownLoader(),
        ".markdown": MarkdownLoader(),
        ".html": HTMLLoader(),
        ".htm": HTMLLoader(),
        ".db": SQLLoader(**sql_config) if sql_config else None,
        ".sqlite": SQLLoader(**sql_config) if sql_config else None,
    }
    
    for path in paths:
        if isinstance(path, str) and urlparse(path).scheme in ['http', 'https']:
            # Handle URLs
            if path.lower().endswith(('.html', '.htm')):
                yield from HTMLLoader().load(path)
            else:
                logger.warning(f"Unsupported URL type: {path}")
        elif isinstance(path, Path) and path.is_file():
            loader = loaders.get(path.suffix.lower())
            if loader:
                yield from loader.load(path)
            else:
                logger.warning(f"No loader found for {path}")
        elif isinstance(path, Path) and path.is_dir():
            # Recursively process directories
            yield from load_documents(list(path.glob("*")), sql_config=sql_config) 