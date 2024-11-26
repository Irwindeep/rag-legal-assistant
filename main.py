from src.assistant import Assistant

def main(query=None, attachment=None):
    legal_assistant = Assistant()
    return legal_assistant(text_query=query, attachment=attachment)

if __name__ == "__main__":
    query = "What was the main argument made by the petitioner, Masud Khan, in this case?"
    main(query=query)
