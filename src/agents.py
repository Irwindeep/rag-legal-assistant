from retriever import LegalRetriever
from transformers import pipeline

class CaseDiscoveryAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def handle(self, query):
        results = self.retriever.retrieve(query)
        return results

class DocumentSummarizationAgent:
    def __init__(self, model="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model)

    def handle(self, text, max_length=200):
        summary = self.summarizer(text, max_length=max_length, truncation=True)
        return summary[0]["summary_text"]

class LegalDraftingAgent:
    def __init__(self):
        pass

    def handle(self, query):
        # Placeholder for drafting logic
        return f"Drafting document with query: {query}"

class QueryResolutionAgent:
    def __init__(self, retriever, model="gpt-4"):
        self.retriever = retriever
        self.generator = pipeline("text-generation", model=model)

    def handle(self, query, context_docs=3, max_tokens=300):
        retrieved_docs = self.retriever.retrieve(query, top_k=context_docs)
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])

        prompt = (
            f"The following is a legal query: '{query}'\n"
            f"Use the provided context to generate an accurate, grounded response.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:"
        )
        response = self.generator(prompt, max_new_tokens=max_tokens, truncation=True)

        return {
            "query": query,
            "context": [doc["text"][:500] for doc in retrieved_docs],  # Trim for readability
            "retrieved_docs": [{"file_name": doc["file_name"], "score": doc["score"]} for doc in retrieved_docs],
            "answer": response[0]["generated_text"],
        }