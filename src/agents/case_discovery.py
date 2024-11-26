import os, faiss
import numpy as np
from query_decompose.preprocess import PreprocessAttachment
from sentence_transformers import SentenceTransformer

class CaseDiscoveryAgent:
    def __init__(self, pipe, case_docs_dir="../../data/casedocs/", index_path="case_index.faiss", embedding_dim=768) -> None:
        self.case_docs_dir = case_docs_dir
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.preprocessor = PreprocessAttachment()
        self.doc_metadata = []
        self.index = self._initialise_index()

        self.pipe = pipe

    def _initialise_index(self):
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            print(f"Loaded FAISS index from {self.index_path}")
        else:
            index = faiss.IndexFlatL2(self.embedding_dim)
            print("Initialized a new FAISS index")
        return index
    
    def build_index(self):
        for file_name in os.listdir(self.case_docs_dir):
            file_path = os.path.join(self.case_docs_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            try:
                document_text = self.preprocessor(file_path, uploads=False)

                embedding = self.embedding_model.encode(document_text)
                self.index.add(np.array([embedding]))

                self.doc_metadata.append({"file_name": file_name, "text": document_text})
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")

    def fusion_retrieval(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)

        retrieved_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.doc_metadata):
                doc_meta = self.doc_metadata[idx]
                retrieved_docs.append({
                    "file_name": doc_meta["file_name"],
                    "text": doc_meta["text"],
                    "score": -distance
                })

        return retrieved_docs
    
    def generate_summary(self, documents, query: str) -> str:
        concatenated_docs = "\n\n".join([doc["text"] for doc in documents])

        summary = self.pipe(concatenated_docs+'\n\n'+query, max_length=4096, temperature=0.7, top_p=0.9)
        return summary[0]["generated_text"]
    
    def retrieve_and_generate(self, query, top_k=5):
        retrieval_results = self.fusion_retrieval(query, top_k)

        summary = self.generate_summary(retrieval_results, query)
        return {"retrieval_results": retrieval_results, "summary": summary}
    
    def __call__(self, query: str):
        self.build_index()
        results = self.retrieve_and_generate(query)
        return results["summary"]