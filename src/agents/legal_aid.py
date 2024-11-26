from sentence_transformers import SentenceTransformer
import os, faiss
import numpy as np

class LegalAidAgent:
    def __init__(self, pipe, knowledge_base_path="../../data/casedocs/", embedding_dim=768, index_path="legal_aid_index.faiss"):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.qa_pipeline = pipe
        self.index = self._initialise_index()
        self.index_exist = False

        self.doc_metadata = []

    def _initialise_index(self):
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            print(f"Loaded FAISS index from {self.index_path}")
            self.index_exist = True
        else:
            index = faiss.IndexFlatL2(self.embedding_dim)
            print("Initialized a new FAISS index")
        return index

    def build_index(self):
        for file_name in os.listdir(self.knowledge_base_path):
            file_path = os.path.join(self.knowledge_base_path, file_name)
            if not os.path.isfile(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                embedding = self.embedding_model.encode(content)
                self.index.add(np.array([embedding]))
                self.doc_metadata.append({"file_name": file_name, "content": content})
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")
        self.index_exist = True

    def find_relevant_documents(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)

        retrieved_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.doc_metadata):
                doc_meta = self.doc_metadata[idx]
                retrieved_docs.append({"file_name": doc_meta["file_name"], "content": doc_meta["content"], "score": -distance})

        return retrieved_docs

    def provide_aid(self, query: str):
        relevant_docs = self.find_relevant_documents(query)
        context = "\n\n".join([doc["content"] for doc in relevant_docs])

        answer = self.qa_pipeline(f"Answer the following questions:\n\nQuestion: {query}\n\nContext: {context}", max_length=4096, temperature=0.7, top_p=0.9)
        return {"relevant_docs": relevant_docs, "answer": answer["answer"]}

    def __call__(self, query: str):
        self.build_index()
        results = self.provide_aid(query)
        return results
