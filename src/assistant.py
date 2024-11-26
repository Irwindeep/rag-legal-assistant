from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .agents.case_discovery import CaseDiscoveryAgent
from .agents.legal_aid import LegalAidAgent
from .agents.legal_draft import LegalDraftingAgent
from .query_decompose.decompose import Decomposer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

class Assistant:
    def __init__(self, pipe=pipe):
        self.pipe = pipe
        self.case_discovery_agent = CaseDiscoveryAgent(pipe)
        self.legal_aid_agent = LegalAidAgent(pipe)
        self.legal_drafting_agent = LegalDraftingAgent(pipe)
        self.decomposer = Decomposer()

        self.task_alloc = {
            "case discovery": self.case_discovery_agent,
            "document summarization": self.case_discovery_agent,
            "legal drafting": self.legal_drafting_agent,
            "query resolution": self.legal_aid_agent
        }

    def _find_intent(self, text_query: str = None, attachment: str = None):
        intent = self.decomposer(text_query, attachment)
        
        return intent
    
    def __call__(self, text_query: str = None, attachment: str = None):
        intent = self._find_intent(text_query, attachment)
        subqueries = intent['sub_queries']

        results = []

        for subquery in subqueries:
            results.append(self.task_alloc[subquery['task']](subquery['text']))

        return "".join(results)
    