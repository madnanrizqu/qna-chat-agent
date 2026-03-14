#!/usr/bin/env python
"""Load documents from data directory into vector database.

This script reads all .txt files from the data directory, generates embeddings,
and stores them in the vector database for similarity search.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings import vector_store, text_splitter


def load_data_directory(data_path: str = "data") -> list[str]:
    """Load all .txt files from data directory into vector store.

    Args:
        data_path: Path to data directory (relative to project root)

    Returns:
        List of document IDs that were created
    """
    # Get absolute path to data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / data_path

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all .txt files
    txt_files = list(data_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️  No .txt files found in {data_dir}")
        return []

    print(f"📂 Found {len(txt_files)} documents in {data_dir}")

    # Read document contents, chunk them, and derive categories from filenames
    documents = []
    categories = []
    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding="utf-8").strip()
            if not content:
                print(f"⚠️  Skipping empty file: {txt_file.name}")
                continue

            category = txt_file.stem  # e.g., "billing_policy" from "billing_policy.txt"

            # Extract title from first line (e.g., "Document 1 — Billing Policy")
            first_line = content.split("\n", 1)[0]
            title = first_line.strip()

            # Chunk the document
            chunks = text_splitter.split_text(content)
            print(
                f"  • {txt_file.name} ({len(content)} chars) → {len(chunks)} chunks → category: {category}"
            )

            # Each chunk inherits the parent file's category
            for chunk in chunks:
                # Post process: If chunk doesn't start with the title, prepend it
                if not chunk.startswith(title):
                    chunk = f"{title}\n{chunk}"
                documents.append(chunk)
                categories.append(category)

        except Exception as e:
            print(f"❌ Error reading {txt_file.name}: {e}")
            continue

    if not documents:
        print("⚠️  No valid documents to load")
        return []

    # Batch insert into vector store with categories
    print(f"\n🔄 Generating embeddings and storing {len(documents)} chunks...")
    try:
        doc_ids = vector_store.store_documents(documents, categories)
        print(f"✅ Successfully loaded {len(doc_ids)} chunks")
        print(f"\nChunk IDs:")
        for i, doc_id in enumerate(doc_ids, 1):
            preview = (
                documents[i - 1][:60] + "..."
                if len(documents[i - 1]) > 60
                else documents[i - 1]
            )
            print(f"  {i}. {doc_id} (category: {categories[i-1]}) - {preview}")
        return doc_ids

    except Exception as e:
        print(f"❌ Error storing documents: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load documents from data directory into vector database"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory (default: data)",
    )

    args = parser.parse_args()

    try:
        doc_ids = load_data_directory(args.data_dir)
        if doc_ids:
            print(f"\n✨ Done! {len(doc_ids)} chunks are now searchable.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to load documents: {e}")
        sys.exit(1)
