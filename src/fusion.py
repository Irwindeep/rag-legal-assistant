class FusionRAG:
    def __init__(self, agents):
        self.agents = agents

    def retrieve_and_generate(self, query):
        case_results = self.agents["case_discovery"].discover_cases(query)
        summaries = [self.agents["summarization"].summarize(doc["text"]) for doc in case_results]
        return {
            "query": query,
            "related_cases": case_results,
            "summaries": summaries,
        }
