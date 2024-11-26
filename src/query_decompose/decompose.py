from .preprocess import PreprocessAttachment
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy.cli import download

LEGAL_TASK_LABELS = {
    "case discovery": ["case law", "precedent", "legal cases"],
    "document summarization": ["summarize", "abstract", "key points"],
    "legal drafting": ["draft", "compose", "notice", "contract"],
    "query resolution": ["answer", "resolve", "legal query"],
}

class Decomposer:
    def __init__(self) -> None:
        self.attachment_preprocessor = PreprocessAttachment()
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            download("en_core_web_trf")
            self.nlp = spacy.load("en_core_web_trf")

        self.embedding_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

        self.task_embeddings = {
            task: self.embedding_model.encode(" ".join(keywords))
            for task, keywords in LEGAL_TASK_LABELS.items()
        }

    def _most_relevant_task(self, sentence: str) -> str:
        sentence_embedding = self.embedding_model.encode(sentence)

        similarities = {
            task: util.cos_sim(sentence_embedding, task_emb).item()
            for task, task_emb in self.task_embeddings.items()
        }

        return max(similarities, key=similarities.get)

    def _input_analysis(self, input_text: str) -> dict:
        doc = self.nlp(input_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        subqueries = [
            {"text": sentence, "task": self._most_relevant_task(sentence)}
            for sentence in sentences
        ]

        overall_intent = max(
            [subquery["task"] for subquery in subqueries],
            key=lambda task: len([sq for sq in subqueries if sq["task"] == task]),
        )

        return {"intent": overall_intent, "sub_queries": subqueries}

    def __call__(self, input_text: str = None, file_name: str = None):
        try:
            preprocessed_text = ""
            if file_name:
                preprocessed_text = self.attachment_preprocessor(file_name)

            full_text = f"{input_text}\n{preprocessed_text}".strip() if input_text else preprocessed_text
            if not full_text:
                raise ValueError("No valid input provided")
            
            result = self._input_analysis(full_text)
            return result
        
        except Exception as e:
            return {"error": str(e)}
