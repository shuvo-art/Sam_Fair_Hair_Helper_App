import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import openai
from pinecone import Pinecone
from typing import List, Dict, Any
from collections import deque
from dotenv import load_dotenv
import time
import logging
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
import argparse
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ndis_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NDISAssistant")

# Load environment variables
load_dotenv()

class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
        
    def add_interaction(self, user_message, bot_response):
        self.history.append({"user": user_message, "bot": bot_response})
        
    def load_history(self, history_data: str):
        """Load conversation history from a string formatted as User: ... Assistant: ..."""
        try:
            self.history.clear()
            if not history_data:
                return
            
            lines = history_data.split('\n')
            user_message = None
            for line in lines:
                if line.startswith('User: '):
                    user_message = line[6:].strip()
                elif line.startswith('Assistant: ') and user_message:
                    bot_response = line[11:].strip()
                    self.add_interaction(user_message, bot_response)
                    user_message = None
        except Exception as e:
            logger.error(f"Error loading conversation history: {str(e)}")

    def get_conversation_context(self):
        context = ""
        if self.history:
            context = "Previous conversation:\n"
            for interaction in self.history:
                context += f"User: {interaction['user']}\n"
                context += f"Bot: {interaction['bot']}\n\n"
        return context
    
    def clear(self):
        self.history.clear()

class NDISAssistant:
    def __init__(
        self,
        budget_info: List[Dict] = None, 
        top_k: int = 5,
        similarity_threshold: float = 0.6, 
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4-turbo", 
        max_history: int = 5,
        pinecone_index_name: str = "test-value"
    ):
        self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.memory = ConversationMemory(max_history=max_history)
        self.last_sources = []
        
        self.query_count = 0
        self.start_time = datetime.now()

        try:
            self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
            try:
                self.knowledge_index = self.pc.Index(pinecone_index_name)
                logger.info(f"Connected to Pinecone index using new SDK method: {pinecone_index_name}")
            except Exception as e1:
                try:
                    self.knowledge_index = self.pc.index(pinecone_index_name)
                    logger.info(f"Connected to Pinecone index using classic method: {pinecone_index_name}")
                except Exception as e2:
                    self.knowledge_index = self.pc[pinecone_index_name]
                    logger.info(f"Connected to Pinecone index using dictionary access: {pinecone_index_name}")
            
            logger.info(f"Testing Pinecone connection...")
            test_vector = [0.0] * 1536
            test_response = self.knowledge_index.query(
                vector=test_vector,
                top_k=1,
                include_metadata=True
            )
            logger.info(f"Pinecone connection test successful")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
        
        self._load_budget(budget_info)
        self._load_api_key()
        
        logger.info("NDIS Assistant initialized with Pinecone knowledge base")

    def _load_budget(self, budget_info: List[Dict]) -> None:
        """Load and parse user's NDIS budget information."""
        try:
            if not budget_info:
                self.budget_data = []
                self.budget_info = "No budget information provided."
                return
                
            self.budget_data = budget_info
            budget_text = "User's NDIS Budget Information:\n"
            for entry in budget_info:
                budget_text += f"- Category: {entry.get('category', 'Unknown')}\n"
                budget_text += f"  Subcategory: {entry.get('subcategory', 'Unknown')}\n"
                budget_text += f"  Amount: ${entry.get('amount', 0):.2f}\n"
                budget_text += f"  Plan Period: {entry.get('startDate', 'Unknown')} to {entry.get('endDate', 'Unknown')}\n"
            
            self.budget_info = budget_text
            logger.info("Budget information parsed successfully")
            
        except Exception as e:
            logger.error(f"Error loading budget: {str(e)}")
            self.budget_info = "Error loading budget information."
            self.budget_data = []

    def _load_api_key(self) -> None:
        """Load OpenAI API key from environment variables."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Missing OPENAI_API_KEY environment variable")
            raise EnvironmentError("Missing OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key

    def _get_embedding(self, text: str, retry_attempts: int = 3) -> np.ndarray:
        """Generate embedding for query using OpenAI."""
        for attempt in range(retry_attempts):
            try:
                result = openai.Embedding.create(
                    model=self.embedding_model,
                    input=text
                )
                return np.array(result['data'][0]['embedding'])
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt+1} failed: {str(e)}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to generate embedding after {retry_attempts} attempts")
                    raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file, including OCR from images."""
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for page_num in range(len(doc)):
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
                    logger.error(f"OCR error on page {page_num + 1}: {str(e)}")

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
                        logger.error(f"OCR error on image {img_idx} of page {page_num + 1}: {str(e)}")

                page_text = text_content + "\n\n" + "\n\n".join(ocr_texts)
                if page_text.strip():
                    full_text.append(f"Page {page_num + 1}:\n{page_text.strip()}")

            doc.close()
            return "\n\n".join(full_text) if full_text else "No text extracted from PDF."
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return f"Error extracting text from PDF: {str(e)}"

    def find_relevant_content(self, query: str, pdf_name: str = None) -> str:
        """Find most relevant context chunks from the Pinecone knowledge base."""
        try:
            query_embedding = self._get_embedding(query)
            enhanced_query = f"NDIS {query}"
            enhanced_embedding = self._get_embedding(enhanced_query)
            
            try:
                filter = None
                if pdf_name:
                    filter = {"pdf_name": {"$eq": pdf_name}}
                
                query_response = self.knowledge_index.query(
                    vector=enhanced_embedding.tolist(),
                    top_k=self.top_k,
                    include_metadata=True,
                    filter=filter
                )
                
                relevant_pages = []
                self.last_sources = []
                
                matches = query_response.get('matches', []) if isinstance(query_response, dict) else query_response.matches
                
                for match in matches:
                    score = match.get('score', match.score) if isinstance(match, dict) else match.score
                    metadata = match.get('metadata', match.metadata) if isinstance(match, dict) else match.metadata
                    
                    if score >= self.similarity_threshold:
                        source_doc = metadata.get('source_document', 
                                     metadata.get('pdf_name', 
                                     metadata.get('filename', 'Unknown Document')))
                        page_num = metadata.get('page_number', 
                                  metadata.get('page', 
                                  metadata.get('page_num', 'N/A')))
                        content_text = metadata.get('text', 
                                      metadata.get('content', 
                                      metadata.get('chunk_text', '')))
                        
                        logger.info(f"Found metadata - Source: {source_doc}, Page: {page_num}, Score: {score}")
                        
                        relevant_pages.append({
                            'text': content_text,
                            'score': score,
                            'source_document': source_doc,
                            'page_number': page_num
                        })
                        
                        self.last_sources.append({
                            'pdf_name': source_doc,
                            'page': page_num,
                            'text': content_text[:500],
                            'score': round(float(score), 3)
                        })
                
                if relevant_pages:
                    context = "RELEVANT KNOWLEDGE BASE CONTEXT:\n"
                    for i, page in enumerate(relevant_pages, 1):
                        context += f"Section {i} [Source: {page['source_document']}, Page: {page['page_number']}]:\n{page['text']}\n\n"
                    logger.info(f"Found {len(relevant_pages)} relevant sections for query")
                    return context
                
                fallback_response = self.knowledge_index.query(
                    vector=query_embedding.tolist(),
                    top_k=self.top_k * 2,
                    include_metadata=True,
                    filter=filter
                )
                
                fallback_threshold = self.similarity_threshold * 0.8
                fallback_matches = fallback_response.get('matches', []) if isinstance(fallback_response, dict) else fallback_response.matches
                
                for match in fallback_matches:
                    score = match.get('score', match.score) if isinstance(match, dict) else match.score
                    metadata = match.get('metadata', match.metadata) if isinstance(match, dict) else match.metadata
                    
                    if score >= fallback_threshold:
                        source_doc = metadata.get('source_document', 
                                         metadata.get('pdf_name', 
                                         metadata.get('filename', 'Unknown Document')))
                        page_num = metadata.get('page_number', 
                                      metadata.get('page', 
                                      metadata.get('page_num', 'N/A')))
                        content_text = metadata.get('text', 
                                          metadata.get('content', 
                                          metadata.get('chunk_text', '')))
                        
                        relevant_pages.append({
                            'text': content_text,
                            'score': score,
                            'source_document': source_doc,
                            'page_number': page_num
                        })
                        
                        self.last_sources.append({
                            'pdf_name': source_doc,
                            'page': page_num,
                            'text': content_text[:500],
                            'score': round(float(score), 3)
                        })
                
                if relevant_pages:
                    context = "RELEVANT KNOWLEDGE BASE CONTEXT:\n"
                    for i, page in enumerate(relevant_pages, 1):
                        context += f"Section {i} [Source: {page['source_document']}, Page: {page['page_number']}]:\n{page['text']}\n\n"
                    logger.info(f"Found {len(relevant_pages)} relevant sections in fallback query")
                    return context
                
                logger.warning("No relevant context found in knowledge base")
                return f"No exact matches found in the knowledge base for {'PDF: ' + pdf_name if pdf_name else 'query'}. Using general NDIS knowledge."
                
            except Exception as e:
                logger.error(f"Pinecone query error: {str(e)}")
                return "Error retrieving context from knowledge base."
                
        except Exception as e:
            logger.error(f"Error finding relevant content: {str(e)}")
            return "Error retrieving context from knowledge base."

    def generate_response(self, context: str, user_message: str) -> str:
        """Generate response using OpenAI."""
        try:
            conversation_context = self.memory.get_conversation_context()
            source_info = ""
            if self.last_sources:
                source_info = "MANDATORY SOURCE CITATIONS - You MUST include these specific sources in your Sources section:\n"
                for i, source in enumerate(self.last_sources, 1):
                    pdf_name = source.get('pdf_name', 'Unknown document')
                    page_num = source.get('page', 'Unknown page')
                    source_info += f"{i}. Document: {pdf_name}, Page: {page_num}\n"
            
            system_prompt = """
            You are an NDIS Assistant, a friendly and knowledgeable support agent helping users with questions about the National Disability Insurance Scheme (NDIS) in Australia.
            # Core Responsibilities
            - Seamlessly blend information from the knowledge base with your own understanding of NDIS
            - Fill any gaps in knowledge base information with your comprehensive understanding of NDIS
            - Consider the user's budget information when answering queries; personalize the answer using this budget information when relevant 
            - Reference previous conversation points when relevant
            - Keep your response concise but informative
            - Provide concise, well-structured responses optimized for Rich text markdown rendering
            - Even when no exact matches are found in the knowledge base, provide accurate information about NDIS based on your training
            - Be sensitive and respectful when discussing disability or health topics
            # Response Structure
            - Start with a clear, direct answer to the query in 1-2 sentences
            - Use Rich-compatible markdown formatting with headers:
            * Level 2 headings for main sections
            * Level 3 headings for subsections
            * Extensive use of bullet points for key information: `* Point with brief description`
            * Use at least 3-5 bullet points under each heading when appropriate
            * Numbered lists for sequential steps: `1. Item`
            * **Bold** for emphasis: `**bold**`
            * *Italics* for terminology: `*italics*`
            * `code` formatting for references: `` `code` ``
            - Keep responses compact but comprehensive with bullet points
            - Include these two sections at the end of every response:
            1. ## Sources
                * ALWAYS cite specific documents and page numbers used from the knowledge base
                * Format as: [Document Name] (Page X)
                * List multiple sources if used
                * If no specific sources were found, state "Information based on general NDIS guidelines and policies"
            2. ## Relevant Links
                * Provide 3-5 relevant links from this approved list (focus on top 3):
                * NDIS Main Site: https://www.ndis.gov.au
                * NDIS Guidelines: https://ourguidelines.ndis.gov.au
                * Hai Helper: https://haihelper.com.au
                * Admin Review Tribunal (Appeals): https://www.art.gov.au/applying-review/national-disability-insurance-scheme
                * Australian Legislation: https://www.legislation.gov.au
                * eCase Search (Tribunal): https://www.art.gov.au/help-and-resources/ecase-search
                * Published Tribunal Decisions: https://www.art.gov.au/about-us/our-role/published-decisions
            # Formatting Notes
            - Use empty lines between sections for better readability
            - Format budget amounts as "$X,XXX.XX" when mentioning specific figures
            # Interaction Style
            - Maintain a warm, conversational tone
            - Engage in small talk when appropriate, but keep focus on NDIS-related topics
            - End responses with a brief encouraging note that invites further questions
            # CRITICAL SOURCE CITATION REQUIREMENTS
            - You MUST cite the specific documents and page numbers provided in the SOURCE DOCUMENTS section
            - Under no circumstances should you use "Information based on general NDIS guidelines and policies" if specific sources were provided
            - Format citations exactly as: "Document Name, from Page number `X`"
            - List ALL provided source documents in your Sources section
            """
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": f"""
                KNOWLEDGE BASE CONTENT:
                {context}
                SOURCE DOCUMENTS:
                {source_info}
                USER'S BUDGET INFORMATION:
                {self.budget_info}
                CONVERSATION HISTORY:
                {conversation_context}
                CURRENT QUESTION:
                {user_message}
                """}
            ]
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1200,
                top_p=1.0
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm experiencing a technical issue. Could you please try again in a moment?"

    def answer_question(self, user_message: str, pdf_path: str = None) -> str:
        """Process user message and generate a response."""
        try:
            self.query_count += 1
            logger.info(f"Processing query #{self.query_count}: {user_message[:50]}...")
            
            if user_message.lower() == "clear history":
                self.memory.clear()
                return "I've cleared our conversation history."
                
            pdf_name = None
            if pdf_path:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                user_message = f"{user_message}\n\n[User-uploaded PDF: {pdf_name}]"
            
            relevant_content = self.find_relevant_content(user_message, pdf_name)
            response = self.generate_response(relevant_content, user_message)
            self.memory.add_interaction(user_message, response)
            return response
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try again or rephrase your question."

    def get_sources(self) -> List[Dict]:
        """Return sources used in the last response."""
        return self.last_sources

    def print_sources(self) -> None:
        """Display sources used in the last response."""
        if not self.last_sources:
            print("\nSources: None used.")
            return
            
        print("\nSources used in the last response:")
        for i, source in enumerate(self.last_sources, 1):
            pdf_name = source.get('pdf_name', 'Unknown document')
            page_num = source.get('page', 'Unknown page')
            score = source.get('score', 0.0)
            print(f"{i}. Document: {pdf_name}")
            print(f"   Page: {page_num}")
            print(f"   Relevance Score: {score}")
            print(f"   Preview: {source.get('text', '')[:150]}...\n")

def interactive_chat():
    """Interactive chat interface for NDIS Assistant."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        has_rich = True
        console = Console()
    except ImportError:
        has_rich = False
        print("\nTip: Install 'rich' package for better markdown rendering: pip install rich\n")
    
    try:
        assistant = NDISAssistant(
            budget_info=[],
            top_k=5,
            similarity_threshold=0.4,
            pinecone_index_name="test-value"
        )
        
        print("\n🌟 Welcome to the NDIS Assistant! 🌟")
        print("Ask me anything about the National Disability Insurance Scheme (NDIS).")
        print("(Type 'exit' to quit, 'clear history' to reset our conversation, or 'sources' to see last sources)\n")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == "exit":
                print("\nThank you for using the NDIS Assistant. Goodbye!")
                break
                
            if user_input.lower() == "sources":
                assistant.print_sources()
                continue
                
            response = assistant.answer_question(user_input)
            print()
            if has_rich:
                console.print(Markdown(response))
            else:
                print(f"NDIS Assistant: {response}")
            print()
            
    except Exception as e:
        print(f"Error initializing NDIS Assistant: {str(e)}")
        print("Please ensure your OpenAI API key and Pinecone API key are set in the .env file")

def main():
    parser = argparse.ArgumentParser(description="NDIS Assistant for chat interactions")
    parser.add_argument("text_input", nargs="?", default="", help="User text input")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--budget", help="Budget information JSON")
    parser.add_argument("--history", help="Conversation history")
    args = parser.parse_args()

    budget_info = json.loads(args.budget) if args.budget else []
    history_data = args.history if args.history else ""

    assistant = NDISAssistant(
        budget_info=budget_info,
        top_k=5,
        similarity_threshold=0.4,
        pinecone_index_name="test-value"
    )
    assistant.memory.load_history(history_data)
    
    user_message = args.text_input
    if args.image:
        try:
            image = Image.open(args.image)
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                user_message = f"{user_message}\n\n[Image Text]: {ocr_text}" if user_message else f"[Image Text]: {ocr_text}"
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            user_message = f"{user_message}\n\n[Error processing image]" if user_message else "[Error processing image]"
    
    response = assistant.answer_question(user_message, args.pdf)
    print(f"AI Response: {response}")

if __name__ == "__main__":
    main()