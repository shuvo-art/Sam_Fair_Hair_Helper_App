import random
import string
import os
import fitz  # PyMuPDF
from typing import List, Dict
import openai
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re
import argparse

load_dotenv()

class WordVectorizerOpenAIPinecone:
    def __init__(
        self,
        folder_path: str,
        pinecone_index_name: str,
    ):
        """
        Initialize the PDF vectorizer with OpenAI embeddings and Pinecone storage.
        Processes PDF files from the specified folder path, using the PDF name as part of the vector ID.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        self.folder_path = folder_path
        self.index_name = pinecone_index_name

        # OpenAI setup
        self.embedding_model = "text-embedding-3-small"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Pinecone setup
        self.pc = Pinecone(os.environ.get('PINECONE_API_KEY'))

        if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
            print(f"Creating new index: {pinecone_index_name}")
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=1536,  # Dimension for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(pinecone_index_name)

    def generate_unique_id(self) -> str:
        """Generate a 5-character unique alphanumeric ID."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

    def sanitize_vector_id(self, vector_id: str) -> str:
        """Sanitize the vector ID to ensure it contains only ASCII characters and is valid for Pinecone."""
        sanitized = re.sub(r'[^\w\d]', '_', vector_id)
        sanitized = re.sub(r'[^\x00-\x7F]+', '_', sanitized)
        if len(sanitized) > 64:
            prefix = sanitized[:54]
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            sanitized = f"{prefix}_{suffix}"
        return sanitized

    def extract_text_from_pdf_page(self, doc: fitz.Document, page_num: int) -> str:
        """Extract text from a single PDF page, including OCR from images."""
        try:
            page = doc.load_page(page_num)
            text_content = page.get_text().strip()

            ocr_texts = []
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    ocr_texts.append(ocr_text)
            except Exception as e:
                print(f"OCR error on page {page_num + 1}: {e}")

            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                try:
                    ocr_text = pytesseract.image_to_string(image).strip()
                    if ocr_text:
                        ocr_texts.append(ocr_text)
                except Exception as e:
                    print(f"OCR error on image {img_idx} of page {page_num + 1}: {e}")

            full_text = text_content + "\n\n" + "\n\n".join(ocr_texts)
            return full_text.strip()

        except Exception as e:
            print(f"Error extracting text from page {page_num + 1}: {e}")
            return ""

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using OpenAI's embedding model."""
        try:
            truncated = text[:8000]
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=truncated
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []

    def delete_vectors_by_pdf_name(self, pdf_name: str):
        """Delete all existing vectors in the index related to a specific PDF name."""
        try:
            results = self.index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
                filter={"pdf_name": {"$eq": pdf_name}}
            )
            
            if results and "matches" in results:
                vector_ids = [match["id"] for match in results["matches"] if "id" in match]
                
                if vector_ids:
                    batch_size = 100
                    for i in range(0, len(vector_ids), batch_size):
                        batch = vector_ids[i:i+batch_size]
                        self.index.delete(ids=batch)
                    
                    print(f"Deleted {len(vector_ids)} vectors for PDF: {pdf_name}")
                else:
                    print(f"No vectors found for PDF: {pdf_name}")
            else:
                print(f"No matches found for PDF: {pdf_name}")
                
        except Exception as e:
            print(f"Error deleting vectors for PDF {pdf_name}: {e}")

    def embed_and_store_pdf(self, file_path: str, replace_existing: bool = True) -> str:
        """Process a single PDF document page by page and create embeddings for each page."""
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            if replace_existing:
                self.delete_vectors_by_pdf_name(pdf_name)
                
            doc = fitz.open(file_path)
            print(f"Processing {file_path} with {len(doc)} pages")
            
            vectors_batch = []
            batch_size = 50
            
            for page_num in range(len(doc)):
                page_text = self.extract_text_from_pdf_page(doc, page_num)
                
                if not page_text:
                    print(f"Warning: No text extracted from page {page_num + 1} of {file_path}")
                    continue

                embedding = self.create_embedding(page_text)
                if not embedding:
                    print(f"Warning: Failed to create embedding for page {page_num + 1} of {file_path}")
                    continue

                sanitized_pdf_name = self.sanitize_vector_id(pdf_name)
                unique_id = self.generate_unique_id()
                vector_id = f"{sanitized_pdf_name}_{page_num}_{unique_id}"
                vector_id = self.sanitize_vector_id(vector_id)

                metadata = {
                    "file_name": os.path.basename(file_path),
                    "pdf_name": pdf_name,
                    "page_number": page_num + 1,
                    "text": page_text[:8000],
                    "char_count": len(page_text)
                }

                vectors_batch.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                if len(vectors_batch) >= batch_size or page_num == len(doc) - 1:
                    if vectors_batch:
                        self.index.upsert(vectors=vectors_batch)
                        print(f"Uploaded batch of {len(vectors_batch)} vectors")
                        vectors_batch = []

            doc.close()
            print(f"Successfully processed PDF: {pdf_name}")
            return f"PDF {pdf_name} processed and uploaded successfully"
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return f"Error processing PDF {file_path}: {str(e)}"

    def query_similar(self, query_text: str, top_k: int = 5):
        """Query Pinecone for similar documents based on the input text."""
        query_embedding = self.create_embedding(query_text)
        if not query_embedding:
            print("Failed to create query embedding")
            return {"matches": []}
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        return results

    def process_all_pdfs_in_folder(self, replace_existing: bool = True):
        """
        Process all PDFs in the given folder.
        If replace_existing is True, it will replace any existing vectors for PDFs with the same name.
        """
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)
                print(f"Processing {file_path}")
                self.embed_and_store_pdf(file_path, replace_existing=replace_existing)

    def process_specific_pdf(self, filename: str, replace_existing: bool = True):
        """
        Process a specific PDF file from the folder.
        If replace_existing is True, it will replace any existing vectors for the PDF with the same name.
        """
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.exists(file_path):
                print(f"Processing specific file: {file_path}")
                self.embed_and_store_pdf(file_path, replace_existing=replace_existing)
            else:
                print(f"File not found: {file_path}")
        else:
            print(f"Not a PDF file: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Process a PDF file and store embeddings in Pinecone")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    folder_path = os.path.dirname(args.pdf_path)
    vectorizer = WordVectorizerOpenAIPinecone(
        folder_path=folder_path,
        pinecone_index_name="test-value"
    )
    response = vectorizer.embed_and_store_pdf(args.pdf_path)
    print(f"AI Response: {response}")

if __name__ == "__main__":
    main()