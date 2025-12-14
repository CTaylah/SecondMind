from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings



import hashlib
from pathlib import Path
from typing import Optional
import yaml
import re

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TEST_FOLDER_PATH = Path("/home/cardell/Documents/BackupAcademic")

HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

def load_markdown_files(root: Path) -> list[Path]:
    return root.rglob("*.md")


#Performs basic normalization of whitespace and returns text
#Returns (file content, metadata)
def read_markdown(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")
    text = text.strip()

    metadata = {
        "file_name": path.name,
        "file_path": str(path),
    }
    return (text, metadata)


def normalize_markdown(text: str) -> str:
    # Remove fenced code block language identifiers
    text = re.sub(r"```[\w+-]*", "```", text)

    # Convert links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    return text


def split_frontmatter(text: str):
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    
    meta = yaml.safe_load(match.group(1)) or {}

    body = text[match.end():]
    return meta, body


def split_by_headers(text: str, metadata: dict, headers: Optional[list[tuple]] = None):
    if not headers:
        headers = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
    splitter = MarkdownHeaderTextSplitter(headers)
    chunks = splitter.split_text(text)
    for document in chunks:
        document.metadata.update(metadata)
    return chunks

#Storing a hash allows to quickly check if changes have been made to this chunk
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def chunk_documents(docs, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def enrich_chunks(chunks, file_path:Path=None, frontmatter: dict=None):
    enriched = []
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "chunk_hash": hash_text(chunk.page_content),
        })
        enriched.append(chunk)
    return enriched


def ingest(folder_path: str):
    files = load_markdown_files(folder_path)

    file_contents = [read_markdown(f) for f in files]
    header_content = [ split_by_headers(normalize_markdown(text), metadata) for text, metadata in file_contents]

    document_chunks = []
    for docs in header_content:
        if len(docs) == 0:
            continue
        else:
            document_chunks.append(
                    enrich_chunks(chunk_documents(docs))
                )
        
    return document_chunks




# Setup Chroma
embedding_model = OpenAIEmbeddings()

vector_store = Chroma(
    collection_name="notes",
    embedding_function=embedding_model,
    persist_directory="./data/chroma"
)
