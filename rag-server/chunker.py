"""
Markdown chunker with metadata extraction.
Parses markdown files by headings and extracts rich metadata.
"""
import re
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document


def parse_markdown_by_headings(content: str) -> List[Dict[str, Any]]:
    """
    Parse markdown content by headings (h1-h6).
    Returns list of chunks with content and metadata.
    """
    lines = content.split('\n')
    chunks = []

    # Current heading hierarchy
    heading_levels = {}

    # Current content buffer
    current_content = []
    current_headings = []

    for line in lines:
        # Match markdown headings (# to ######)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if heading_match:
            # Save previous chunk if exists
            if current_content:
                chunk_text = '\n'.join(current_content).strip()
                if chunk_text:
                    chunks.append({
                        'content': chunk_text,
                        'headings': current_headings.copy()
                    })

            # Update heading hierarchy
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            heading_levels[level] = title

            # Keep only headings of higher levels (lower number)
            heading_levels = {k: v for k, v in heading_levels.items() if k < level}

            # Build full heading path
            path_parts = []
            for l in sorted(heading_levels.keys()):
                path_parts.append(heading_levels[l])
            path_parts.append(title)

            current_headings = path_parts
            current_content = [line]
        else:
            current_content.append(line)

    # Save last chunk
    if current_content:
        chunk_text = '\n'.join(current_content).strip()
        if chunk_text:
            chunks.append({
                'content': chunk_text,
                'headings': current_headings
            })

    return chunks


def chunk_file(file_path: Path, content: str) -> List[Document]:
    """
    Chunk a single markdown file by headings.
    Returns list of LlamaIndex Documents with metadata.
    """
    # Get relative path from vault
    vault_path = Path("/vault/obsidian")
    try:
        rel_path = file_path.relative_to(vault_path)
    except ValueError:
        rel_path = file_path

    # Extract title from first heading or filename
    title = rel_path.stem

    # Parse by headings
    parsed_chunks = parse_markdown_by_headings(content)

    documents = []
    for i, chunk in enumerate(parsed_chunks):
        # Build metadata
        metadata = {
            'source': str(rel_path),
            'file_name': rel_path.name,
            'title': title,
            'chunk_index': i,
            'total_chunks': len(parsed_chunks),
            'heading_path': ' > '.join(chunk['headings']) if chunk['headings'] else title,
        }

        # Add heading levels as separate fields for filtering
        if chunk['headings']:
            metadata['heading_level'] = len(chunk['headings'])
            metadata['top_heading'] = chunk['headings'][0] if chunk['headings'] else ''

        doc = Document(
            text=chunk['content'],
            metadata=metadata,
            id_=f"{rel_path}_{i}"
        )
        documents.append(doc)

    return documents


def chunk_directory(vault_path: str) -> List[Document]:
    """
    Chunk all markdown files in a directory.
    Returns list of all Documents.
    """
    vault = Path(vault_path)
    all_documents = []

    # Find all markdown files
    for md_file in vault.rglob("*.md"):
        # Skip obsidian config
        if '.obsidian' in md_file.parts:
            continue

        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
            continue

        docs = chunk_file(md_file, content)
        all_documents.extend(docs)
        print(f"Chunked {md_file.name}: {len(docs)} chunks")

    # Also handle .qmd files
    for qmd_file in vault.rglob("*.qmd"):
        if '.obsidian' in qmd_file.parts:
            continue

        try:
            content = qmd_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {qmd_file}: {e}")
            continue

        docs = chunk_file(qmd_file, content)
        all_documents.extend(docs)
        print(f"Chunked {qmd_file.name}: {len(docs)} chunks")

    return all_documents
