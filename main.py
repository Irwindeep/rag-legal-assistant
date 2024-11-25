from src.retriever import LegalRetriever
from src.agents import CaseDiscoveryAgent
from src.agents import DocumentSummarizationAgent
from src.fusion import FusionRAG

def main():
    # Initialize components
    retriever = LegalRetriever()
    case_discovery_agent = CaseDiscoveryAgent(retriever)
    summarization_agent = DocumentSummarizationAgent()
    
    # Multi-Agent Setup
    agents = {
        "case_discovery": case_discovery_agent,
        "summarization": summarization_agent,
    }
    fusion_rag = FusionRAG(agents)

    # Query the system
    query = "Important cases about environmental law in India"
    results = fusion_rag.retrieve_and_generate(query)

    # Output the results
    print("Query:", results["query"])
    print("\nRelated Cases:")
    for case in results["related_cases"]:
        print(f"- {case['file_name']} (Score: {case['score']})")
    print("\nSummaries:")
    for summary in results["summaries"]:
        print(f"- {summary}")

if __name__ == "__main__":
    main()
