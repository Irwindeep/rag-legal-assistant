from sentence_transformers import SentenceTransformer, util
import json

class LegalRetriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", index_file="../data/preprocessed.json"):
        self.model = SentenceTransformer(embedding_model)
        self.documents = json.load(open(index_file))
        self.embeddings = [self.model.encode(doc["text"], convert_to_tensor=True) for doc in self.documents]

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = [util.cos_sim(query_embedding, doc_emb)[0].item() for doc_emb in self.embeddings]
        ranked_docs = sorted(zip(scores, self.documents), key=lambda x: x[0], reverse=True)[:top_k]
        return [{"score": score, "text": doc["text"], "file_name": doc["file_name"]} for score, doc in ranked_docs]
