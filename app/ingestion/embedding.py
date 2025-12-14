from ingestion import markdown_processing

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_openai import OpenAIEmbeddings

from pathlib import Path
from uuid import uuid4



class ChromaDBManager:
    def __init__(self, collection_name, embedding_model):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory="./data/chroma",
        )

    def update(self, chunks):
        for documents in chunks:
            docs_to_add = []
            ids_to_add = []

            for chunk in documents:
                doc_id = (
                    f"{chunk.metadata['file_path']}::"
                    f"{chunk.metadata['chunk_index']}"
                )

                existing = self.vector_store.get(ids=[doc_id])

                if existing["ids"]:
                    existing_hash = existing["metadatas"][0]["chunk_hash"]
                    if existing_hash == chunk.metadata["chunk_hash"]:
                        continue  # unchanged → skip

                # new or changed chunk
                docs_to_add.append(chunk)
                ids_to_add.append(doc_id)

            if docs_to_add:
                self.vector_store.add_documents(
                    documents=docs_to_add,
                    ids=ids_to_add
                )


class EmbeddingManager:

    def __init__(self, collection_name, data_source_path):
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.chroma_manager = ChromaDBManager(collection_name=collection_name, embedding_model=embedding_model)

        chunks = markdown_processing.ingest(data_source_path)

        self.chroma_manager.update(chunks)


    def similarity_search(self, query: str, k: int=5) -> list:
        results = self.vector_store.similarity_search(query, k)

        return results
    
    def format_chunks(self, chunks):
        context_string = ""
        for i, chunk in enumerate(chunks):
            context_string += f"[DOC {i}]\n"
            context_string += f"[Source: {chunk.metadata['file_name']}]\n"
            context_string += f"[Content: {chunk.page_content}]\n"
        return context_string

    def make_query(self, query:str, k: int=3):
        return self.format_chunks(
            self.similarity_search(query=query, k=k)
        )