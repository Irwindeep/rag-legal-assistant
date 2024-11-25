import os
from PyPDF2 import PdfReader
from tqdm import tqdm
import json

def parse_pdfs(pdf_directory, output_file="preprocessed.json"):
    data = []
    for file in tqdm(os.listdir(pdf_directory)):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_directory, file))
            text = " ".join([page.extract_text() for page in reader.pages])
            data.append({"file_name": file, "text": text})
    with open(output_file, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    parse_pdfs("../data/", "../data/preprocessed.json")
