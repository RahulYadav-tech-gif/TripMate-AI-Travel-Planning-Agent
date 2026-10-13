"""Parses destination guide text files into section-level chunks for ChromaDB."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DocumentChunk:
    """A section chunk with city, section name, and text content."""

    id: str
    city: str
    country: str
    section: str
    content: str
    source: str

    def to_metadata(self) -> dict:
        """Metadata dictionary stored alongside embedding vectors."""
        return {
            "city": self.city,
            "country": self.country,
            "section": self.section,
            "source": self.source,
        }


KNOWN_SECTIONS = [
    "VISA & ENTRY",
    "BEST TIME TO VISIT",
    "LOCAL CUSTOMS",
    "PACKING TIPS",
    "SAFETY & HEALTH",
]


def parse_destination_file(file_path: Path) -> List[DocumentChunk]:
    """Splits a single destination text file into section-based DocumentChunks."""
    text = file_path.read_text(encoding="utf-8")
    filename = file_path.name

    # Extract City and Country from header: e.g. "DESTINATION GUIDE: TOKYO, JAPAN"
    title_match = re.search(r"DESTINATION GUIDE:\s*([^,\n]+)(?:,\s*([^\n]+))?", text, re.IGNORECASE)
    if title_match:
        city = title_match.group(1).strip().lower()
        country = (title_match.group(2) or "").strip().lower()
    else:
        city = file_path.stem.lower()
        country = ""

    chunks: List[DocumentChunk] = []

    # Split text on section headings
    pattern = r"\n(?=" + r"|".join(re.escape(s) for s in KNOWN_SECTIONS) + r")"
    parts = re.split(pattern, text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        matched_section = None
        for sec in KNOWN_SECTIONS:
            if part.startswith(sec):
                matched_section = sec
                break

        if matched_section:
            content_body = part[len(matched_section) :].strip()
            # Prefix with city and section for clearer semantic match
            formatted_content = f"[{city.upper()} - {matched_section}]\n{content_body}"
            chunk_id = f"{city}_{matched_section.lower().replace(' ', '_').replace('&', 'and')}"

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    city=city,
                    country=country,
                    section=matched_section,
                    content=formatted_content,
                    source=filename,
                )
            )

    return chunks


def load_all_destination_chunks(data_dir: Path) -> List[DocumentChunk]:
    """Loads and parses all destination guide text files from data directory."""
    chunks: List[DocumentChunk] = []
    txt_files = sorted(data_dir.glob("*.txt"))

    for file_path in txt_files:
        if file_path.name.lower() == "readme.txt":
            continue
        file_chunks = parse_destination_file(file_path)
        chunks.extend(file_chunks)

    return chunks
