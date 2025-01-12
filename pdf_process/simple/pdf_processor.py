import logging
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Generator
from tqdm import tqdm


class PDFProcessor:
    """Class for extracting and preprocessing text from PDF files"""

    def __init__(self, min_line_length: int = 10):
        self.min_line_length = min_line_length
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def clean_text(self, text: str) -> str:
        """Cleans and formats the text"""
        # Remove unnecessary spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s\.,!?;]', '', text)
        # Convert multiple punctuation marks to single ones
        text = re.sub(r'([.,!?;])\1+', r'\1', text)
        return text.strip()

    def process_pdf(self, pdf_path: str) -> Generator[str, None, None]:
        """Processes the PDF file in chunks"""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Split the text into lines and clean
                lines = text.split('\n')
                cleaned_lines = [
                    self.clean_text(line)
                    for line in lines
                    if len(line.strip()) >= self.min_line_length
                ]

                # Yield each page as a separate chunk
                if cleaned_lines:
                    yield ' '.join(cleaned_lines)

            doc.close()

        except Exception as e:
            logging.error(f"PDF processing error ({pdf_path}): {str(e)}")
            yield ""

    def process_pdf_directory(self, pdf_dir: str, output_file: str):
        """Processes all PDFs in a directory and merges them"""
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found: {pdf_dir}")

        logging.info(f"A total of {len(pdf_files)} PDF files will be processed")

        # Chunk size (10MB)
        chunk_size = 10 * 1024 * 1024
        current_chunk = []
        current_size = 0
        chunk_counter = 0

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
                for text_chunk in self.process_pdf(str(pdf_file)):
                    if text_chunk:
                        chunk_size_bytes = len(text_chunk.encode('utf-8'))

                        if current_size + chunk_size_bytes > chunk_size:
                            # Write the current chunk
                            f.write('\n'.join(current_chunk) + '\n')
                            current_chunk = []
                            current_size = 0
                            chunk_counter += 1

                            if chunk_counter % 10 == 0:
                                logging.info(f"{chunk_counter} chunks processed")

                        current_chunk.append(text_chunk)
                        current_size += chunk_size_bytes

            # Write the remaining chunk
            if current_chunk:
                f.write('\n'.join(current_chunk))

        logging.info(f"Process completed. Output file: {output_file}")
