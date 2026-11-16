"""Ask questions about a local PDF with Ollama and LangChain.

The retrieval index lives in memory for this run. By default, the PDF text,
embeddings, and chat requests stay on this machine through local Ollama.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


@dataclass
class Chunk:
    """A piece of PDF text and the page it came from."""

    page: int
    text: str
    vector: list[float] | None = None


def read_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text page-by-page using Poppler's pdftotext command."""
    if not pdf_path.is_file():
        raise ValueError(f"PDF not found: {pdf_path}")

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pdftotext is missing. Install Poppler, then try again."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or "Poppler could not read this file."
        raise RuntimeError(details) from exc

    pages = []
    for page_number, page_text in enumerate(result.stdout.split("\f"), start=1):
        cleaned = page_text.strip()
        if cleaned:
            pages.append((page_number, cleaned))

    if not pages:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; OCR is not "
            "included in this first version."
        )
    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, preferring word breaks."""
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.6), end)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def make_chunks(
    pages: list[tuple[int, str]], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Keep each chunk's page number so answers can point back to the PDF."""
    return [
        Chunk(page=page_number, text=piece)
        for page_number, page_text in pages
        for piece in split_text(page_text, chunk_size, overlap)
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compare two embeddings. Higher scores mean the texts are more alike."""
    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def embed_chunks(
    chunks: list[Chunk], embeddings: OllamaEmbeddings, batch_size: int
) -> None:
    """Embed chunks in batches and update one live terminal progress line."""
    total = len(chunks)
    total_batches = math.ceil(total / batch_size)
    completed = 0

    for batch_number, start in enumerate(range(0, total, batch_size), start=1):
        batch = chunks[start : start + batch_size]
        batch_end = start + len(batch)
        print(
            f"\r\033[KEmbedding batch {batch_number}/{total_batches} "
            f"(chunks {start + 1}-{batch_end} of {total})...",
            end="",
            flush=True,
        )
        vectors = embeddings.embed_documents([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("The embedding model returned an unexpected number of vectors.")

        for chunk, vector in zip(batch, vectors):
            chunk.vector = vector
            completed += 1
            percent = completed / total * 100
            print(
                f"\r\033[KEmbedded {completed}/{total} chunks ({percent:.1f}%)",
                end="",
                flush=True,
            )

    print()


def create_prompt_chain(chat_model: str, base_url: str):
    """Join a reusable prompt, local chat model, and string output parser."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the PDF excerpts below. "
                "Cite the page for each claim with the format [Page 3]. If the "
                "excerpts do not contain the answer, say you could not find it "
                "in this PDF.\n\nPDF excerpts:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    model = ChatOllama(
        model=chat_model,
        base_url=base_url,
        temperature=0,
    )
    return prompt | model | StrOutputParser()


def ask_loop(
    chunks: list[Chunk],
    embeddings: OllamaEmbeddings,
    answer_chain,
    top_k: int,
) -> None:
    print("\nReady. Ask a question; type 'exit' to finish.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not question:
            continue

        try:
            question_vector = embeddings.embed_query(question)
            ranked = sorted(
                chunks,
                key=lambda chunk: cosine_similarity(
                    chunk.vector or [], question_vector
                ),
                reverse=True,
            )[:top_k]
            context = "\n\n".join(
                f"[Page {chunk.page}]\n{chunk.text}" for chunk in ranked
            )
            answer = answer_chain.invoke({"context": context, "question": question})
        except Exception as exc:  # Ollama errors vary across client versions.
            print(f"Ollama request failed: {exc}")
            print("Check that Ollama is running and both model names are available.")
            continue

        print(f"\nAssistant: {answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF using local Ollama models."
    )
    parser.add_argument("pdf", type=Path, help="path to a text-based PDF")
    parser.add_argument(
        "--chat-model",
        default=os.getenv("OLLAMA_CHAT_MODEL", "llama3"),
        help="Ollama chat model (default: llama3 or OLLAMA_CHAT_MODEL)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        help=(
            "Ollama embedding model (default: nomic-embed-text or "
            "OLLAMA_EMBEDDING_MODEL)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument("--top-k", type=int, default=4, help="chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=1000, help="chunk size")
    parser.add_argument(
        "--chunk-overlap", type=int, default=150, help="overlap between chunks"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="chunks to embed per Ollama request (default: 64)",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be zero or more and smaller than chunk size")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    print(f"Reading {args.pdf}...")
    try:
        pages = read_pdf(args.pdf)
    except (ValueError, RuntimeError) as exc:
        print(f"Could not read PDF: {exc}", file=sys.stderr)
        return 1

    chunks = make_chunks(pages, args.chunk_size, args.chunk_overlap)
    print(f"Found text on {len(pages)} pages; split it into {len(chunks)} chunks.")
    print(f"Creating local embeddings with {args.embedding_model}...")
    embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=args.base_url)
    try:
        embed_chunks(chunks, embeddings, args.embedding_batch_size)
    except Exception as exc:  # Ollama errors vary across client versions.
        print(f"Could not create embeddings: {exc}", file=sys.stderr)
        print(
            f"Make sure Ollama is running and the model exists. "
            f"You can fetch it with: ollama pull {args.embedding_model}",
            file=sys.stderr,
        )
        return 1

    answer_chain = create_prompt_chain(args.chat_model, args.base_url)
    print(f"Ready to answer from {args.pdf.name} with {args.chat_model}.")
    ask_loop(chunks, embeddings, answer_chain, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
