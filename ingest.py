from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
def read_text_file(file_path: Path) -> str:
    """
    Read a plain text file or markdown file and return it's content.
    For now, we start with only .txt and .md files.
    PDFs will be added later.
    """
    return file_path.read_text(encoding="utf-8", errors="ignore")

def clean_text(text: str) -> str:
    """
    Basic cleanup: 
    - collpase repeated whitespace
    - remove extra blank lines
    """
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t ]+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character based chunking with overlap
    Later we can improve this to token based or paragraph aware chunking
    """
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start+chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - chunk_overlap
    return chunks


def process_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read one file, clean it, chunk it, and attach metadata
    """
    raw_text = read_text_file(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    records: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": f"{file_path.stem}_chunk_{i}",
            "text": chunk,
            "source": file_path.name,
            "chunk_index": i
        })
    return records

def save_chunks(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.md"))
    if not input_files:
        print(f"No .txt or .md files found in {DATA_DIR.resolve()}")
        return
    
    all_records: List[Dict[str, Any]] = []
    for file_path in input_files:
        records = process_file(file_path)
        all_records.extend(records)
        print(f"Processed {file_path.name}: {len(records)} chunks")
    output_file = OUTPUT_DIR / "chunks.json"
    save_chunks(all_records, output_file)
    print(f"\n Saved {len(all_records)} chunks to {output_file.resolve()}")

if __name__ == "__main__":
    main()


































































