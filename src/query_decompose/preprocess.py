import os
import pdfplumber
import pytesseract
from PIL import Image

class PreprocessAttachment:
    def __init__(self) -> None:
        self.attachment_dir = "uploads/"
        self.allowed_extensions = [".pdf", ".txt", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"]

    def _pdf_to_text(self, pdf_path: str) -> str:
        text = ''
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''

        return text
    
    def _txt_to_text(self, txt_path: str) -> str:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
        
    def _img_to_text(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise RuntimeError(f"Error processing image file: {e}")
        
    def __call__(self, file_name: str) -> str:
        file_path = os.path.join("./uploads", file_name)
        file_extension = os.path.splitext(file_path)[-1].lower()

        if file_extension == ".pdf": return self._pdf_to_text(file_path)
        if file_extension == ".txt": return self._txt_to_text(file_path)
        if file_extension == self.allowed_extensions: return self._img_to_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
