from __future__ import annotations

import asyncio
import io
import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import UUID

import fitz  # type: ignore[import]

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import pytesseract
    from pytesseract import TesseractError, TesseractNotFoundError
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]
    TesseractError = Exception  # type: ignore[assignment]
    TesseractNotFoundError = Exception  # type: ignore[assignment]

from app.models.section import SectionCreate
from app.services.concept_extraction import extract_and_store_concepts
from app.services.embeddings import embed_paper_sections
from app.services.papers import get_paper, update_paper_status
from app.services.sections import replace_sections
from app.services.storage import download_pdf_from_storage


HEADING_NUMERIC_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+")
UPPERCASE_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9\s\-:,]{2,}$")
COMMON_HEADING_KEYWORDS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgements",
    "acknowledgments",
    "references",
}
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

SNIPPET_MAX_CHARS = 400
MIN_TEXT_LENGTH_FOR_OCR = 500
PARAGRAPH_MIN_CHARS = 350
FALLBACK_CHUNK_SIZE = 1400
FALLBACK_CHUNK_OVERLAP = 200


@dataclass
class PageBlock:
    page_index: int
    text: str
    font_size: float
    max_font_size: float
    is_bold: bool
    x0: float
    y0: float
    char_start: int = 0
    char_end: int = 0


@dataclass
class DocumentExtraction:
    full_text: str
    blocks: List[PageBlock]
    page_starts: List[int]
    page_ends: List[int]
    median_font_size: float


@dataclass
class ParsedSection:
    title: Optional[str]
    content: str
    char_start: int
    char_end: int
    page_number: Optional[int]


async def parse_pdf_task(paper_id: UUID) -> None:
    print(f"[parse_pdf_task] Starting parsing for paper {paper_id}")
    paper = await get_paper(paper_id)
    if not paper:
        print(f"[parse_pdf_task] Paper {paper_id} no longer exists")
        return
    if not paper.file_path:
        print(f"[parse_pdf_task] Paper {paper_id} has no associated file")
        await update_paper_status(paper_id, "failed")
        return

    await update_paper_status(paper_id, "parsing")

    try:
        pdf_bytes = await download_pdf_from_storage(paper.file_path)
        parsed_sections = await asyncio.to_thread(_parse_pdf_bytes, pdf_bytes)
        if not parsed_sections:
            raise RuntimeError("Parsing pipeline returned no sections")

        section_models = [
            SectionCreate(
                paper_id=paper_id,
                title=section.title,
                content=section.content,
                char_start=section.char_start,
                char_end=section.char_end,
                page_number=section.page_number,
                snippet=_build_snippet(section.content),
            )
            for section in parsed_sections
        ]

        await replace_sections(paper_id, section_models)
        try:
            await extract_and_store_concepts(paper_id, section_models)
        except Exception as exc:  # pragma: no cover - background task logging
            print(
                f"[parse_pdf_task] Failed to extract concepts for paper {paper_id}: {exc}"
            )
        try:
            await embed_paper_sections(paper_id)
        except Exception as exc:  # pragma: no cover - background task logging
            print(
                f"[parse_pdf_task] Failed to generate embeddings for paper {paper_id}: {exc}"
            )
        await update_paper_status(paper_id, "parsed")
        print(
            f"[parse_pdf_task] Completed parsing for paper {paper_id} with "
            f"{len(section_models)} sections"
        )
    except Exception as exc:  # pragma: no cover - background task logging
        await update_paper_status(paper_id, "failed")
        print(f"[parse_pdf_task] Failed to parse paper {paper_id}: {exc}")


def _parse_pdf_bytes(pdf_bytes: bytes) -> List[ParsedSection]:
    extraction = _extract_document(pdf_bytes)
    if not extraction.full_text or not extraction.full_text.strip():
        raise RuntimeError("Unable to extract text from PDF")

    sections = _build_sections_from_headings(extraction)
    if not _is_viable_section_set(sections, extraction.full_text):
        sections = _build_paragraph_sections(extraction)

    if not sections:
        sections = _build_chunk_sections(extraction)

    if not sections:
        raise RuntimeError("Failed to segment PDF content into sections")

    return sections


def _extract_document(pdf_bytes: bytes) -> DocumentExtraction:
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except (fitz.FileDataError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "Unable to open PDF document. The file may be corrupted or unsupported."
        ) from exc

    try:
        extraction = _extract_using_pymupdf(document)
        if (
            len(extraction.full_text.strip()) < MIN_TEXT_LENGTH_FOR_OCR
            and pytesseract is not None
            and Image is not None
        ):
            ocr_extraction = _extract_with_ocr(document)
            if (
                ocr_extraction
                and len(ocr_extraction.full_text.strip())
                > len(extraction.full_text.strip())
            ):
                extraction = ocr_extraction
        return extraction
    finally:
        document.close()


def _extract_using_pymupdf(document: fitz.Document) -> DocumentExtraction:
    blocks: List[PageBlock] = []
    page_starts: List[int] = []
    page_ends: List[int] = []
    font_sizes: List[float] = []
    parts: List[str] = []
    cursor = 0

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_blocks, page_fonts = _extract_page_blocks(page, page_index)
        page_text, page_blocks = _build_page_text(page_blocks)

        page_start = cursor
        page_starts.append(page_start)

        for block in page_blocks:
            block.char_start += page_start
            block.char_end += page_start
            blocks.append(block)

        cursor += len(page_text)
        page_ends.append(cursor)
        parts.append(page_text)
        if page_index < document.page_count - 1:
            parts.append("\n\n")
            cursor += 2

        font_sizes.extend(page_fonts)

    full_text = "".join(parts)
    median_font = statistics.median(font_sizes) if font_sizes else 0.0
    return DocumentExtraction(
        full_text=full_text,
        blocks=blocks,
        page_starts=page_starts,
        page_ends=page_ends,
        median_font_size=median_font,
    )


def _extract_page_blocks(
    page: fitz.Page, page_index: int
) -> Tuple[List[PageBlock], List[float]]:
    raw = page.get_text("dict")
    page_blocks: List[PageBlock] = []
    font_sizes: List[float] = []

    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue

        lines = block.get("lines", [])
        text_lines: List[str] = []
        span_sizes: List[float] = []
        is_bold = False

        for line in lines:
            spans = line.get("spans", [])
            line_parts: List[str] = []
            for span in spans:
                span_text = span.get("text", "")
                if not span_text:
                    continue
                line_parts.append(span_text)
                size = float(span.get("size", 0.0) or 0.0)
                if size > 0:
                    span_sizes.append(size)
                flags = span.get("flags", 0)
                if isinstance(flags, int) and flags & 2:
                    is_bold = True
            if line_parts:
                text_lines.append("".join(line_parts))

        text = "\n".join(text_lines).strip()
        if not text:
            continue

        if span_sizes:
            font_sizes.extend(span_sizes)
            avg_size = statistics.mean(span_sizes)
            max_size = max(span_sizes)
        else:
            avg_size = 0.0
            max_size = 0.0

        x0, y0, *_ = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        page_blocks.append(
            PageBlock(
                page_index=page_index,
                text=text,
                font_size=avg_size,
                max_font_size=max_size or avg_size,
                is_bold=is_bold,
                x0=float(x0),
                y0=float(y0),
            )
        )

    page_blocks.sort(key=lambda blk: (round(blk.y0, 2), round(blk.x0, 2)))
    return page_blocks, font_sizes


def _build_page_text(blocks: List[PageBlock]) -> tuple[str, List[PageBlock]]:
    if not blocks:
        return "", []

    parts: List[str] = []
    cursor = 0
    for idx, block in enumerate(blocks):
        if idx > 0:
            parts.append("\n")
            cursor += 1
        block.char_start = cursor
        parts.append(block.text)
        cursor += len(block.text)
        block.char_end = cursor

    return "".join(parts), blocks


def _extract_with_ocr(document: fitz.Document) -> Optional[DocumentExtraction]:
    if pytesseract is None or Image is None:
        return None

    page_texts: List[str] = []
    page_starts: List[int] = []
    page_ends: List[int] = []
    parts: List[str] = []
    cursor = 0

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        try:
            pixmap = page.get_pixmap(dpi=200)
        except Exception:  # pragma: no cover - PyMuPDF rendering specifics
            page_texts.append("")
            continue

        try:
            with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                text = pytesseract.image_to_string(image)
        except (TesseractError, TesseractNotFoundError, OSError):  # pragma: no cover - external dependency behaviour
            return None

        cleaned = text.strip()
        page_texts.append(cleaned)

    if not any(page_texts):
        return None

    for index, text in enumerate(page_texts):
        page_starts.append(cursor)
        parts.append(text)
        cursor += len(text)
        page_ends.append(cursor)
        if index < len(page_texts) - 1:
            parts.append("\n\n")
            cursor += 2

    full_text = "".join(parts)
    return DocumentExtraction(
        full_text=full_text,
        blocks=[],
        page_starts=page_starts,
        page_ends=page_ends,
        median_font_size=0.0,
    )


def _build_sections_from_headings(extraction: DocumentExtraction) -> List[ParsedSection]:
    if not extraction.blocks:
        return []

    headings = [
        block
        for block in extraction.blocks
        if _is_heading_candidate(block, extraction.median_font_size)
    ]
    if not headings:
        return []

    headings.sort(key=lambda blk: blk.char_start)
    sections: List[ParsedSection] = []
    full_text = extraction.full_text

    first_start = headings[0].char_start
    leading_content, leading_start, leading_end = _slice_clean(full_text, 0, first_start)
    if leading_content:
        sections.append(
            ParsedSection(
                title="_front_matter_",
                content=leading_content,
                char_start=leading_start,
                char_end=leading_end,
                page_number=_infer_page_number(
                    leading_start, extraction.page_starts, extraction.page_ends
                ),
            )
        )

    for index, heading in enumerate(headings):
        start = heading.char_end
        end = headings[index + 1].char_start if index + 1 < len(headings) else len(full_text)
        content, char_start, char_end = _slice_clean(full_text, start, end)
        if not content:
            continue

        sections.append(
            ParsedSection(
                title=_normalize_title(heading.text),
                content=content,
                char_start=char_start,
                char_end=char_end,
                page_number=_infer_page_number(
                    char_start, extraction.page_starts, extraction.page_ends
                ),
            )
        )

    return sections


def _build_paragraph_sections(extraction: DocumentExtraction) -> List[ParsedSection]:
    full_text = extraction.full_text
    sections: List[ParsedSection] = []
    last_index = 0

    for match in PARAGRAPH_SPLIT_RE.finditer(full_text):
        section = _section_from_range(extraction, last_index, match.start())
        if section:
            sections.append(section)
        last_index = match.end()

    final_section = _section_from_range(extraction, last_index, len(full_text))
    if final_section:
        sections.append(final_section)

    if not sections:
        return []

    merged_sections: List[ParsedSection] = []
    idx = 0
    while idx < len(sections):
        current = sections[idx]
        while (
            len(current.content) < PARAGRAPH_MIN_CHARS
            and idx + 1 < len(sections)
        ):
            nxt = sections[idx + 1]
            combined_end = nxt.char_end
            combined_content = extraction.full_text[current.char_start:combined_end]
            current = ParsedSection(
                title=current.title,
                content=combined_content,
                char_start=current.char_start,
                char_end=combined_end,
                page_number=current.page_number or nxt.page_number,
            )
            idx += 1
        merged_sections.append(current)
        idx += 1

    return merged_sections


def _build_chunk_sections(extraction: DocumentExtraction) -> List[ParsedSection]:
    full_text = extraction.full_text
    total_length = len(full_text)
    if total_length == 0:
        return []

    chunk_size = max(FALLBACK_CHUNK_SIZE, 600)
    overlap = min(FALLBACK_CHUNK_OVERLAP, chunk_size // 2)
    sections: List[ParsedSection] = []
    start = 0
    index = 0

    while start < total_length:
        end = min(total_length, start + chunk_size)
        content, char_start, char_end = _slice_clean(full_text, start, end)
        if content:
            sections.append(
                ParsedSection(
                    title=f"_auto_chunk_{index:03d}",
                    content=content,
                    char_start=char_start,
                    char_end=char_end,
                    page_number=_infer_page_number(
                        char_start, extraction.page_starts, extraction.page_ends
                    ),
                )
            )
            index += 1

        if end == total_length:
            break
        start = max(end - overlap, char_end)

    return sections


def _section_from_range(
    extraction: DocumentExtraction, start: int, end: int
) -> Optional[ParsedSection]:
    if end <= start:
        return None
    content, char_start, char_end = _slice_clean(extraction.full_text, start, end)
    if not content:
        return None
    return ParsedSection(
        title=None,
        content=content,
        char_start=char_start,
        char_end=char_end,
        page_number=_infer_page_number(char_start, extraction.page_starts, extraction.page_ends),
    )


def _is_heading_candidate(block: PageBlock, median_font: float) -> bool:
    text = block.text.strip()
    if not text:
        return False
    if len(text) > 160:
        return False

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if alpha_chars < 3:
        return False

    uppercase_chars = sum(1 for ch in text if ch.isupper())
    uppercase_ratio = uppercase_chars / alpha_chars if alpha_chars else 0.0

    keyword_match = text.lower() in COMMON_HEADING_KEYWORDS
    numeric_match = bool(HEADING_NUMERIC_RE.match(text))
    uppercase_match = bool(UPPERCASE_HEADING_RE.match(text))
    large_font = median_font > 0 and (
        block.font_size >= median_font * 1.2
        or block.max_font_size >= median_font * 1.25
    )

    if keyword_match or numeric_match or uppercase_match:
        return True
    if large_font:
        return True
    if block.is_bold and (
        median_font == 0
        or block.font_size >= median_font * 1.05
        or uppercase_ratio >= 0.6
    ):
        return True
    if uppercase_ratio >= 0.75 and len(text.split()) <= 10:
        return True
    return False


def _is_viable_section_set(sections: List[ParsedSection], full_text: str) -> bool:
    if len(sections) < 2:
        return False
    total_length = len(full_text.strip())
    if total_length == 0:
        return False
    covered = sum(len(section.content) for section in sections)
    coverage_ratio = covered / total_length if total_length else 0
    return coverage_ratio >= 0.4 or (
        coverage_ratio >= 0.25 and len(sections) >= 3
    )


def _slice_clean(full_text: str, start: int, end: int) -> tuple[str, int, int]:
    start = max(start, 0)
    end = min(end, len(full_text))
    if end <= start:
        return "", start, start

    segment = full_text[start:end]
    leading = len(segment) - len(segment.lstrip())
    trailing = len(segment) - len(segment.rstrip())
    char_start = start + leading
    char_end = end - trailing
    if char_end <= char_start:
        return "", char_start, char_start
    return full_text[char_start:char_end], char_start, char_end


def _infer_page_number(
    char_index: int, page_starts: List[int], page_ends: List[int]
) -> Optional[int]:
    for idx, (start, end) in enumerate(zip(page_starts, page_ends)):
        if start <= char_index < end:
            return idx + 1
    if page_starts:
        if char_index >= page_starts[-1]:
            return len(page_starts)
        return 1
    return None


def _normalize_title(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:255]


def _build_snippet(content: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", content).strip()
    if not cleaned:
        return None
    if len(cleaned) <= SNIPPET_MAX_CHARS:
        return cleaned
    return cleaned[:SNIPPET_MAX_CHARS].rstrip() + "…"
