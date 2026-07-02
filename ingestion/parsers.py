"""
Format-specific document parsers for the Hybrid RAG ingestion pipeline.

Supported formats:
- Markdown (.md)   — strips YAML frontmatter, converts to plaintext via markdownify
- Plain text (.txt) — reads UTF-8 with chardet encoding-detection fallback
- HTML (.html/.htm) — extracts body text via BeautifulSoup, preserving heading hierarchy
- PDF (.pdf)       — extracts text with pdfplumber, falls back to pypdf on failure

The top-level `DocumentParser` dispatcher routes by file extension and raises
`UnsupportedFormatError` for any extension not in the supported set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ingestion.models import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class UnsupportedFormatError(Exception):
    """Raised when a file's extension is not supported by any registered parser."""

    def __init__(self, filename: str, extension: str) -> None:
        self.filename = filename
        self.extension = extension
        super().__init__(
            f"Unsupported file format '{extension}' for file '{filename}'. "
            f"Supported formats: .md, .txt, .html, .htm, .pdf"
        )


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------


class MarkdownParser:
    """
    Parses Markdown files into normalized plaintext.

    Strategy:
    1. Strip YAML frontmatter (--- ... --- delimiters at file start).
    2. Attempt to render Markdown → HTML via the ``markdown`` package, then
       strip HTML tags with BeautifulSoup to produce clean plaintext.
    3. If ``markdown`` is unavailable, fall back to a regex-based stripper
       that removes the most common Markdown syntax.
    4. Collapse excessive blank lines.

    Note: ``markdownify`` converts HTML → Markdown (the inverse direction) and
    is therefore *not* used here; we use it only as a dependency signal.
    """

    # Matches YAML frontmatter at the very start of the file: --- ... ---
    _YAML_FRONTMATTER_RE = re.compile(
        r"^---\s*\n.*?\n---\s*\n",
        re.DOTALL,
    )
    # Also handle +++ TOML frontmatter
    _TOML_FRONTMATTER_RE = re.compile(
        r"^\+\+\+\s*\n.*?\n\+\+\+\s*\n",
        re.DOTALL,
    )

    def parse(self, file_path: Path, source_url: str | None = None) -> ParsedDocument:
        raw = file_path.read_text(encoding="utf-8")

        # 1. Strip frontmatter
        content = self._strip_frontmatter(raw)

        # 2. Convert Markdown → plaintext
        plaintext = self._to_plaintext(content)

        # 3. Normalise whitespace: collapse 3+ blank lines to 2
        plaintext = re.sub(r"\n{3,}", "\n\n", plaintext).strip()

        return ParsedDocument(
            filename=file_path.name,
            format="md",
            content=plaintext,
            source_url=source_url,
        )

    def _strip_frontmatter(self, text: str) -> str:
        for pattern in (self._YAML_FRONTMATTER_RE, self._TOML_FRONTMATTER_RE):
            match = pattern.match(text)
            if match:
                return text[match.end():]
        return text

    @staticmethod
    def _to_plaintext(markdown_text: str) -> str:
        """Convert Markdown source to plaintext.

        Primary path: markdown → HTML → BeautifulSoup text extraction.
        Fallback:     regex-based Markdown syntax stripping.
        """
        try:
            import markdown as md_lib  # type: ignore[import-untyped]
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            html = md_lib.markdown(
                markdown_text,
                extensions=["fenced_code", "tables"],
            )
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n")
        except ImportError:
            pass

        # Fallback: regex stripper
        return MarkdownParser._regex_strip_markdown(markdown_text)

    @staticmethod
    def _regex_strip_markdown(text: str) -> str:
        """Minimal regex-based Markdown stripper used when libraries are absent."""
        # Remove ATX heading markers (# ## ### …)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic markers
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
        # Remove inline code
        text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.DOTALL)
        # Remove links — keep display text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove image syntax
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Remove blockquote markers
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        return text


class TextParser:
    """
    Parses plain-text files.

    Tries UTF-8 first; on failure attempts chardet-based encoding detection,
    then falls back to latin-1 (which never raises a decode error).
    """

    def parse(self, file_path: Path, source_url: str | None = None) -> ParsedDocument:
        content = self._read_with_fallback(file_path)
        # Normalise line endings and collapse excessive blank lines
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        return ParsedDocument(
            filename=file_path.name,
            format="txt",
            content=content,
            source_url=source_url,
        )

    @staticmethod
    def _read_with_fallback(file_path: Path) -> str:
        # 1. Try UTF-8
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass

        # 2. Try chardet detection
        raw_bytes = file_path.read_bytes()
        try:
            import chardet  # type: ignore[import-untyped]

            detected = chardet.detect(raw_bytes)
            encoding = detected.get("encoding") or "latin-1"
            logger.debug(
                "chardet detected encoding '%s' for '%s'", encoding, file_path.name
            )
            return raw_bytes.decode(encoding, errors="replace")
        except ImportError:
            pass

        # 3. Final fallback: latin-1 never raises
        logger.warning(
            "chardet not available; falling back to latin-1 for '%s'", file_path.name
        )
        return raw_bytes.decode("latin-1", errors="replace")


class HtmlParser:
    """
    Parses HTML files into plaintext using BeautifulSoup.

    Preserves heading hierarchy by prepending heading-level markers
    (e.g., "# Heading 1", "## Heading 2") before converting to plain text.
    Strips script, style, and nav elements entirely.
    """

    _SKIP_TAGS = {"script", "style", "nav", "footer", "head"}
    _HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}

    def parse(self, file_path: Path, source_url: str | None = None) -> ParsedDocument:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        raw = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")

        # Remove unwanted elements in-place
        for tag in soup.find_all(self._SKIP_TAGS):
            tag.decompose()

        lines: list[str] = []
        self._extract_text(soup.body or soup, lines)

        content = "\n".join(lines)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        return ParsedDocument(
            filename=file_path.name,
            format="html",
            content=content,
            source_url=source_url,
        )

    def _extract_text(self, element: object, lines: list[str]) -> None:
        """Recursively walk the BeautifulSoup tree and collect text lines."""
        from bs4 import NavigableString, Tag  # type: ignore[import-untyped]

        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                lines.append(text)
            return

        if not isinstance(element, Tag):
            return

        tag_name = element.name.lower() if element.name else ""

        if tag_name in self._HEADING_TAGS:
            prefix = self._HEADING_TAGS[tag_name]
            heading_text = element.get_text(separator=" ", strip=True)
            if heading_text:
                lines.append(f"\n{prefix} {heading_text}\n")
            return  # headings are leaf nodes for our purposes

        if tag_name == "p":
            para_text = element.get_text(separator=" ", strip=True)
            if para_text:
                lines.append(para_text)
                lines.append("")  # blank line after paragraph
            return

        if tag_name in ("ul", "ol"):
            for li in element.find_all("li", recursive=False):
                item_text = li.get_text(separator=" ", strip=True)
                if item_text:
                    lines.append(f"- {item_text}")
            lines.append("")
            return

        if tag_name == "pre":
            code_text = element.get_text()
            if code_text.strip():
                lines.append(f"```\n{code_text.rstrip()}\n```")
                lines.append("")
            return

        # For all other tags, recurse into children
        for child in element.children:
            self._extract_text(child, lines)


class PdfParser:
    """
    Parses PDF files into plaintext.

    Uses pdfplumber as the primary extractor (handles complex layouts well).
    Falls back to pypdf if pdfplumber raises an exception or returns empty text.
    """

    def parse(self, file_path: Path, source_url: str | None = None) -> ParsedDocument:
        content = self._extract_with_pdfplumber(file_path)

        if not content.strip():
            logger.warning(
                "pdfplumber returned empty content for '%s'; trying pypdf fallback",
                file_path.name,
            )
            content = self._extract_with_pypdf(file_path)

        # Normalise whitespace
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        return ParsedDocument(
            filename=file_path.name,
            format="pdf",
            content=content,
            source_url=source_url,
        )

    @staticmethod
    def _extract_with_pdfplumber(file_path: Path) -> str:
        try:
            import pdfplumber  # type: ignore[import-untyped]

            with pdfplumber.open(file_path) as pdf:
                pages: list[str] = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)
        except Exception as exc:
            logger.warning(
                "pdfplumber failed on '%s': %s — trying pypdf fallback",
                file_path.name,
                exc,
            )
            return ""

    @staticmethod
    def _extract_with_pypdf(file_path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]

            reader = PdfReader(str(file_path))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except Exception as exc:
            logger.error(
                "pypdf also failed on '%s': %s — returning empty content",
                file_path.name,
                exc,
            )
            return ""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class DocumentParser:
    """
    Routes a file to the appropriate format-specific parser based on extension.

    Supported extensions:
        .md              → MarkdownParser
        .txt             → TextParser
        .html / .htm     → HtmlParser
        .pdf             → PdfParser

    Raises `UnsupportedFormatError` for any other extension.
    """

    _EXTENSION_MAP: dict[str, object] = {}  # populated after class definitions

    def __init__(self) -> None:
        self._markdown_parser = MarkdownParser()
        self._text_parser = TextParser()
        self._html_parser = HtmlParser()
        self._pdf_parser = PdfParser()

        self._registry: dict[str, object] = {
            ".md": self._markdown_parser,
            ".txt": self._text_parser,
            ".html": self._html_parser,
            ".htm": self._html_parser,
            ".pdf": self._pdf_parser,
        }

    def parse(self, file_path: Path, source_url: str | None = None) -> ParsedDocument:
        """
        Parse *file_path* and return a `ParsedDocument`.

        Args:
            file_path:  Path to the file to parse.  Must exist.
            source_url: Optional URL representing the document's origin.

        Returns:
            A fully-populated `ParsedDocument` instance.

        Raises:
            UnsupportedFormatError: If the file's extension is not supported.
            FileNotFoundError:      If *file_path* does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        parser = self._registry.get(ext)

        if parser is None:
            raise UnsupportedFormatError(
                filename=file_path.name,
                extension=ext if ext else "(no extension)",
            )

        logger.debug("Parsing '%s' with %s", file_path.name, type(parser).__name__)
        return parser.parse(file_path, source_url=source_url)  # type: ignore[union-attr]
