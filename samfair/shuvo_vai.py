import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import time
import logging
from datetime import datetime
import argparse

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

class NDISAssistantBotOpenAI:
    def __init__(
        self, 
        embeddings_path: str, 
        budget_info: str,
        conversation_history: str = None, 
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4",
        max_history: int = 10
    ):
        """
        Initialize the NDIS Assistant Bot using OpenAI.
        
        Args:
            embeddings_path: Path to pre-generated embeddings file
            budget_info: User's NDIS budget information as a JSON string
            conversation_history: Prior conversation history
            top_k: Number of most relevant documents to retrieve
            similarity_threshold: Minimum similarity score to consider a document relevant
            embedding_model: OpenAI embedding model to use
            chat_model: OpenAI chat model to use
            max_history: Maximum number of conversation turns to retain
        """
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.conversation_history = []
        self.last_sources = []
        self.max_history = max_history
        
        # Statistics tracking
        self.query_count = 0
        self.start_time = datetime.now()

        # Initialize OpenAI client
        self._load_api_key()
        self.client = OpenAI(api_key=self.api_key)

        # Load data
        self._load_embeddings(embeddings_path)
        self._load_budget(budget_info)

        # Parse conversation history if provided
        if conversation_history:
            self._parse_conversation_history(conversation_history)
        
        logger.info(f"NDIS Assistant Bot initialized with {len(self.page_ids)} documents")

    def _load_embeddings(self, embeddings_path: str) -> None:
        """Load pre-generated OpenAI embeddings with improved error handling."""
        try:
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
                
            logger.info(f"Attempting to load embeddings from: {embeddings_path}")
            
            data = np.load(embeddings_path, allow_pickle=True)
            logger.info(f"NPZ file loaded with keys: {data.files}")
            
            required_keys = ['page_ids', 'embeddings', 'metadata']
            missing_keys = [key for key in required_keys if key not in data.files]
            if missing_keys:
                raise KeyError(f"Missing required keys in embeddings file: {missing_keys}")
            
            self.page_ids = data['page_ids']
            if isinstance(self.page_ids, np.ndarray):
                self.page_ids = self.page_ids.tolist()
            logger.info(f"Loaded {len(self.page_ids)} page IDs")
            
            self.embeddings = data['embeddings']
            if len(self.embeddings.shape) != 2:
                raise ValueError(f"Embeddings have unexpected shape: {self.embeddings.shape}, expected 2D array")
            logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
            
            metadata_raw = data['metadata']
            logger.info(f"Raw metadata type: {type(metadata_raw)}, size: {getattr(metadata_raw, 'size', 'N/A')}")
            
            try:
                if isinstance(metadata_raw, np.ndarray) and metadata_raw.size == 1:
                    metadata_str = metadata_raw.item()
                elif isinstance(metadata_raw, np.ndarray) and metadata_raw.size > 0:
                    metadata_str = metadata_raw[0]
                else:
                    metadata_str = str(metadata_raw)
                    
                logger.info(f"Metadata string type: {type(metadata_str)}")
                self.metadata = json.loads(metadata_str)
                
                if not all(page_id in self.metadata for page_id in self.page_ids):
                    missing_ids = [pid for pid in self.page_ids if pid not in self.metadata]
                    logger.warning(f"Some page IDs missing from metadata: {missing_ids[:5]}...")
                    
                logger.info(f"Successfully parsed metadata with {len(self.metadata)} entries")
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON decoding error: {str(je)}")
                logger.error(f"First 100 chars of metadata string: {str(metadata_str)[:100]}...")
                raise
                
            logger.info(f"Successfully loaded NDIS knowledge base with {len(self.page_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def _load_budget(self, budget_info: str) -> None:
        """Load user's NDIS budget information from a JSON string."""
        try:
            if budget_info:
                # Remove any extra quotes or whitespace that might have been added during escaping
                budget_info_cleaned = budget_info.strip().strip('"')
                budget_data = json.loads(budget_info_cleaned)
                formatted_budget = "Your NDIS plan includes:\n"
                for item in budget_data:
                    formatted_budget += f"- {item['category']} - {item['subcategory']}: ${item['amount']} (From {item['startDate']} to {item['endDate']})\n"
                self.budget_info = formatted_budget
                logger.info("Budget information loaded successfully")
            else:
                self.budget_info = "No budget information available."
                logger.warning("No budget information provided")
        except json.JSONDecodeError as e:
            logger.error(f"Error loading budget: {str(e)}")
            logger.error(f"Problematic budget string: {budget_info[:100]}...")
            self.budget_info = "Error loading budget information."
        except Exception as e:
            logger.error(f"Error loading budget: {str(e)}")
            self.budget_info = "Error loading budget information."

    def _load_api_key(self) -> None:
        """Load OpenAI API key from environment variables."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Missing OPENAI_API_KEY environment variable")
            raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

    def _get_embedding(self, text: str, retry_attempts: int = 3) -> Optional[np.ndarray]:
        """
        Generate embedding for query using OpenAI with retry logic.
        """
        for attempt in range(retry_attempts):
            try:
                result = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return np.array(result.data[0].embedding)
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt+1} failed: {str(e)}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to generate embedding after {retry_attempts} attempts")
                    raise

    def _parse_conversation_history(self, history: str) -> None:
        """Parse conversation history string into internal format."""
        try:
            lines = history.strip().split('\n')
            for line in lines:
                if line.startswith("User: "):
                    self.conversation_history.append({"role": "user", "content": line[6:]})
                elif line.startswith("Assistant: "):
                    self.conversation_history.append({"role": "assistant", "content": line[11:]})
            logger.info(f"Parsed {len(self.conversation_history)} history entries")
        except Exception as e:
            logger.error(f"Error parsing conversation history: {str(e)}")
            self.conversation_history = []
    
    def _truncate_history(self) -> None:
        """Truncate conversation history to maximum length."""
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
            logger.info(f"Truncated conversation history to {len(self.conversation_history)} messages")

    def find_relevant_content(self, query: str) -> str:
        """
        Find most relevant context chunks from the knowledge base.
        """
        try:
            query_embedding = self._get_embedding(query)
            similarities = [
                (page_id, 1 - cosine(query_embedding, self.embeddings[i]))
                for i, page_id in enumerate(self.page_ids)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            top_scores = [score for _, score in similarities[:10]]
            logger.info(f"Top 10 similarity scores: {[round(s, 3) for s in top_scores]}")
            
            effective_threshold = self.similarity_threshold
            if not any(score >= self.similarity_threshold for _, score in similarities[:self.top_k]):
                min_acceptable = 0.3
                if similarities and similarities[0][1] >= min_acceptable:
                    effective_threshold = max(min_acceptable, similarities[0][1] * 0.9)
                    logger.info(f"Adapting threshold from {self.similarity_threshold} to {effective_threshold}")
                else:
                    logger.warning(f"No good matches found. Best score: {similarities[0][1] if similarities else 'N/A'}")
            
            relevant_pages = []
            self.last_sources = []

            for page_id, score in similarities[:self.top_k]:
                if score >= effective_threshold:
                    try:
                        page_info = self.metadata[page_id].copy()
                        page_info['score'] = score
                        relevant_pages.append(page_info)
                        self.last_sources.append({
                            'document': page_info.get('file_name', 'Unknown'),
                            'page': page_info.get('page_number', 'Unknown'),
                            'text': page_info.get('text', '')[:500],
                            'score': round(score, 3)
                        })
                    except KeyError:
                        logger.error(f"Key error for page_id {page_id} - not found in metadata")
                    except Exception as e:
                        logger.error(f"Error processing page {page_id}: {str(e)}")

            if relevant_pages:
                context = "RELEVANT KNOWLEDGE BASE CONTEXT:\n"
                for i, page in enumerate(relevant_pages):
                    context += f"Chunk {i + 1} (Score: {page['score']:.3f}):\n{page['text']}\n\n"
                logger.info(f"Found {len(relevant_pages)} relevant pages for query")
                return context
            
            logger.warning("No relevant context found in knowledge base")
            return "No relevant context found in the knowledge base."
                
        except Exception as e:
            logger.error(f"Error finding relevant content: {str(e)}")
            return "Error retrieving context from knowledge base."

    def debug_search(self, query: str) -> dict:
        """
        Debug function to test similarity searching without generating a response.
        """
        try:
            start_time = time.time()
            query_embedding = self._get_embedding(query)
            embedding_time = time.time() - start_time
            
            start_time = time.time()
            similarities = [
                (page_id, 1 - cosine(query_embedding, self.embeddings[i]))
                for i, page_id in enumerate(self.page_ids)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarity_time = time.time() - start_time
            
            top_results = []
            for page_id, score in similarities[:10]:
                if page_id in self.metadata:
                    result = {
                        'page_id': page_id,
                        'score': round(score, 4),
                        'file': self.metadata[page_id].get('file_name', 'Unknown'),
                        'page': self.metadata[page_id].get('page_number', 'Unknown'),
                        'preview': self.metadata[page_id].get('text', '')[:100] + '...'
                    }
                    top_results.append(result)
            
            return {
                'query': query,
                'embedding_time_ms': round(embedding_time * 1000, 2),
                'similarity_calc_time_ms': round(similarity_time * 1000, 2), 
                'threshold': self.similarity_threshold,
                'top_results': top_results,
                'would_retrieve': any(result['score'] >= self.similarity_threshold for result in top_results)
            }
            
        except Exception as e:
            logger.error(f"Debug search error: {str(e)}")
            return {'error': str(e)}

    def answer_question(self, query: str, temperature: float = 0.7) -> str:
        """
        Generate a context-aware answer to the user's query.
        """
        try:
            self.query_count += 1
            logger.info(f"Processing query #{self.query_count}: {query[:50]}...")
            
            self.conversation_history.append({"role": "user", "content": query})
            
            start_time = time.time()
            relevant_content = self.find_relevant_content(query)
            retrieval_time = time.time() - start_time
            logger.info(f"Content retrieval completed in {retrieval_time:.2f}s")
            
            relevant_lines = relevant_content.split("Chunk")
            if len(relevant_lines) > 4:
                relevant_content = "Chunk".join(relevant_lines[:4])
                logger.info("Limited context to top 3 chunks due to length")

            history_context = ""
            for entry in self.conversation_history[:-1]:
                history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"

            system_prompt = """
            You are an NDIS Assistant, a friendly and knowledgeable support agent helping users with questions, issues, and requests about the National Disability Insurance Scheme (NDIS) in Australia. Act as a human support agent, providing clear, accurate, and warm responses without mentioning you are an AI, referencing internal systems like knowledge bases, or using citations to specific documents in the response body.

            ### 🎯 Your Role
            Your goal is to:

            - Deliver detailed, personalized, and engaging responses by blending NDIS rules and guidelines with your own understanding, elaborating with practical examples, scenarios, and context to address the user's needs comprehensively.
            - Offer friendly, actionable advice focused on NDIS topics (e.g., eligibility, planning, supports, plan management, appeals), making responses feel like a one-on-one conversation tailored to the user's situation.
            - Ask polite follow-up questions if the query is unclear to better understand the user's needs.
            - End responses with an encouraging, supportive note that invites further engagement.

            ### 📚 Information to Use
            1. **Primary Source**: NDIS knowledge base content (provided in the query).

            - Use this as the foundation for accurate NDIS-specific rules, guidelines, or details, but integrate the information naturally without referencing it directly or citing specific documents in the response body.
            - Expand on the content with detailed explanations, real-world applications, or user-relevant scenarios to provide a richer answer.

            2. **Secondary Source**: Your own knowledge.

            - Leverage your general understanding to add depth, clarify complex concepts, or provide context where the knowledge base is limited or technical.
            - Offer insights, examples, or scenarios that align with NDIS principles and Australian disability support frameworks, making the response intuitive and engaging.
            - Use your reasoning to anticipate the user's goals or challenges and address them proactively.

            3. **Tertiary Source**: User's NDIS budget (if provided).

            - Include budget details only when the query directly relates to funding, budgeting, or specific supports, and personalize the response with specific examples of how the user can use their funding.
            - Use phrases like "Based on your NDIS plan…" to tie the budget to the user's needs.

            4. **Additional Sources**: Official NDIS resources or trusted websites.

            - Reference these naturally in the response body, e.g., "The NDIS website explains…", without implying reliance on a knowledge base.
            - Use these sources to provide credibility and context, but avoid direct citations or references to specific documents unless explicitly requested by the user.

            ### 🛡️ Guidelines

            1. **Stay in Character**:

            - Focus on NDIS-related topics, addressing the user as if you're a dedicated support agent with deep expertise.
            - Politely redirect off-topic questions, e.g., "I'd love to help with NDIS-related questions. Could you share more details about what you need?".

            2. **Source Integration**:

            - Do not use citations (e.g., "[Source: Document Name, Page X]") or mention the knowledge base in the response body. Instead, weave NDIS rules and guidelines into the response naturally, as if drawing from your own expertise.
            - When a user asks for sources, information, or where they can learn more, ALWAYS include a "Source" section at the end of your response listing knowledge base documents and pages used, formatted as "Source: (Document Name, Page X)" for each unique source.
            - For external sources referenced in the body, use natural phrasing, e.g., "The NDIS website explains…", to provide credibility without breaking character.
            - When using your own knowledge, no attribution is needed unless referencing a specific external source.

            3. **Budget Information**:

            - Include budget details (categories, subcategories, amounts) only when the query involves funding, plan management, or specific supports.
            - Format as a concise list, e.g., "Your plan includes: - Core Supports: $200 for daily life…", and provide personalized examples of how the user can apply these funds.
            - Ensure budget details are prioritized in responses when relevant, addressing specific funding categories like therapy or equipment.

            4. **Tone & Style**:

            - Use a warm, conversational tone, like a friendly support agent speaking directly to the user.
            - Write clear, full sentences with minimal jargon, ensuring accessibility.
            - Use bullet points or numbered lists for steps, options, or explanations to enhance clarity and structure responses, avoiding single paragraphs for explanations.
            - Be sensitive and respectful, especially on disability or health topics.
            - Create detailed yet concise responses, prioritizing the user's needs and interests.
            - Responses should be empathetic and supportive, recognizing the unique challenges faced by individuals navigating the NDIS system.

            ### 🌐 Additional Resources:

            - NDIS Main Site: https://www.ndis.gov.au
            - NDIS Guidelines: https://ourguidelines.ndis.gov.au
            - Admin Review Tribunal (Appeals): https://www.art.gov.au/applying-review/national-disability-insurance-scheme
            - Hai Helper: https://haihelper.com.au
            - Australian Legislation: https://www.legislation.gov.au
            - eCase Search (Tribunal): https://www.art.gov.au/help-and-resources/ecase-search
            - Published Tribunal Decisions: https://www.art.gov.au/about-us/our-role/published-decisions

            ### 🧠 Response Structure:

            Craft responses that feel natural, engaging, and tailored to the user's query, like a personalized conversation with a friendly NDIS support agent. Avoid rigid structures or explicit references to sources in the response body, and instead weave the following elements into a cohesive, context-appropriate response:

            - **Answer the Query**: 
            - Provide a clear, detailed answer that integrates NDIS rules and guidelines naturally, without citing specific documents or repeating the user's question in any form, including in headings or introductory text.
            - Use bullet points to structure explanations, incorporating practical examples or scenarios to make the information relatable and comprehensive, drawing on your knowledge to add depth.
            - Use headings that describe the topic or purpose of the section in a general way (e.g., "Understanding NDIS Supports" instead of "What Are NDIS Supports?") to avoid query-like phrasing.

            - **Provide Context or Guidance**: 
            - Personalize the response by addressing the user's potential needs, goals, or challenges. 
            - Include budget details only if the query involves funding or plan specifics, formatted concisely (e.g., a bullet list) with examples of how to use the funds. 
            - Otherwise, offer insights, tips, or applications to enhance understanding, using bullet points for clarity.

            - **Offer Actionable Steps**: 
            - Suggest next steps or advice when relevant, using numbered lists or bullet points for clarity if the query calls for procedural guidance. 
            - Tailor steps to the user's situation, making them practical and encouraging.

            - **Source Information**: 
            - When the user asks for sources, information, or where they can learn more, ONLY THEN include a "Source" section listing all knowledge base documents and pages used, formatted as "Source: (Document Name, Page X)".
            - Include the source section at the end of your response after the main content.

            - **Additional Resources**: 
            - When the user asks for more information, resources, websites, or links, ONLY THEN include 2-3 relevant links from the "🌐 Additional Resources" list above based on relevance.
            - Prioritize the NDIS Main Site, NDIS Guidelines, and Hai Helper as the most valuable general resources. Use at least one of these as a primary reference, and incorporate other links based on their relevance.
            - Format as "For more information, you can visit:" followed by the links on separate lines or bullet points.
            - Include this section at the end of your response.

            - **Encouraging Tone**:
            - Close with a warm, positive note (e.g., "I'm here to help with any other questions!") that invites further engagement and feels supportive.

            When crafting responses:

            - Adapt the format to the query's nature (e.g., a brief paragraph for simple questions, lists for procedural queries, or detailed explanations with bullet points for complex topics).
            - ALWAYS Use markdown flexibly for readability (e.g., **bold** for emphasis, *italics* for tone, bullet points or numbered lists for steps or explanations).
            - Ensure a logical flow with smooth transitions, avoiding repetitive or formulaic phrasing.
            - Create responses that feel like a tailored, expert conversation, using NDIS rules as a foundation but elaborating with your own insights to make the answer detailed, engaging, and user-focused.
            - Explanations should be clear and concise, avoiding jargon or overly technical language unless necessary for the context.
            - ALWAYS Use headings and subheadings to break text into sections for readability, ensuring headings describe the topic generally without mimicking the query.
            - Responses should be empathetic and supportive, recognizing the unique challenges faced by individuals navigating the NDIS system.
            - Do not repeat the user's question in the response, including in headings, subheadings, or body text; directly address the topic or issue raised.
            - IMPORTANT: When a user asks where they can find more information, where they can learn more, or asks for resources, ALWAYS provide relevant links from the Additional Resources section based on relevance.
            """

            source_info = ""
            for i, source in enumerate(self.last_sources, 1):
                source_info += f"Source {i}: {source['document']} (Page {source['page']}), Score: {source['score']}\n"
                source_info += f"Text preview: {source['text'][:100]}...\n\n"

            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": f"""
                KNOWLEDGE BASE CONTENT (Primary Source):
                {relevant_content}

                USER'S BUDGET INFORMATION (Secondary Source):
                {self.budget_info}

                SOURCE INFORMATION (FOR YOUR REFERENCE, DO NOT CITE IN RESPONSE BODY):
                {source_info}

                CONVERSATION HISTORY (Use For context if relevant):
                {history_context}

                CURRENT QUESTION:
                {query}

                ## Instructions:

                - Use the knowledge base content as the foundation for NDIS-specific details, but integrate it naturally without referencing or citing it directly in the response body (e.g., avoid "[Source: Document Name, Page X]"). Blend it with your own knowledge to provide a detailed, intuitive, and personalized answer.

                - Expand on the knowledge base with practical explanations, real-world examples, and scenarios that make the information relatable and comprehensive, addressing the user's potential needs or goals. Use bullet points to structure explanations instead of single paragraphs.

                - Only include budget information if the question directly relates to funding, budgeting, or specific supports in the user's NDIS plan, formatted concisely (e.g., bullet list). When included, personalize the response by explaining how the user can apply their funding, prioritizing specific categories like therapy or equipment.

                - Write a friendly, natural response as a human NDIS support agent would, adapting the structure to the query's nature (e.g., brief paragraph for simple questions, lists for steps, or detailed explanation with bullet points for complex topics).

                - Do not mention the knowledge base or use citations in the response body. If referencing external web or X sources, use natural phrasing (e.g., "The NDIS website explains…") and cite at the paragraph's end. If using your own knowledge, no attribution is needed unless citing a specific external source.

                - Include the following elements in a cohesive, conversational flow:

                - **Answer the Query**:
                    - Provide a clear, detailed answer integrating NDIS rules naturally, without repeating the user's question in any form, including in headings, subheadings, or body text.
                    - ALWAYS Use bullet points for explanations, incorporating examples or scenarios to enhance clarity and engagement.
                    - ALWAYS Use headings that describe the topic or purpose generally (e.g., "Exploring NDIS Funding" instead of "What Is NDIS Funding?") to avoid query-like phrasing.

                - **Relevant Context or Guidance**:
                    - Address the user's potential goals or challenges, personalizing with budget details (if applicable) or insights.
                    - Format budget details as a concise bullet list with examples of use, if relevant.
                    - Use bullet points to structure guidance or tips.

                - **Practical Next Steps**:
                    - Offer tailored advice or steps using numbered lists or bullet points for procedural queries.
                    - Ensure steps are practical and encouraging.

                - **Source Information**:
                    - When the user asks for sources, information, or where they can learn more, ONLY THEN include at the end of your response a "Source" section listing knowledge base documents and pages used, formatted as "Source: (Document Name, Page X)".

                - **Additional Resources**:
                    - When the user asks for more information, resources, websites, or links, ONLY THEN include 2-3 relevant links from the "🌐 Additional Resources" list provided.
                    - Format as "For more information, you can visit:" followed by the links.
                    - Prioritize the NDIS Main Site, NDIS Guidelines, and Hai Helper as the most valuable general resources. Use at least one of these as a primary reference, and incorporate other links based on their relevance.

                - **Encouraging Close**:
                    - End with a positive, supportive note (e.g., "Let me know how I can assist further!").

                - Always provide answers in markdown format, using **bold** for emphasis, *italics* for tone, and bullet points or numbered lists for steps or explanations. Use headings and subheadings to break text into sections for readability, ensuring headings describe the topic generally without mimicking the query.

                - IMPORTANT: When a user asks where they can find more information, where they can learn more, or asks for resources, ONLY THEN provide relevant links from the Additional Resources section.

                - Responses should be empathetic and supportive, recognizing the unique challenges faced by individuals navigating the NDIS system.
                """}
            ]

            start_time = time.time()
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2024,
                        top_p=1.0
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    generation_time = time.time() - start_time
                    logger.info(f"Response generated in {generation_time:.2f}s after {attempt+1} attempts")
                    
                    self.conversation_history.append({"role": "assistant", "content": answer})
                    self._truncate_history()
                    
                    return answer
                    
                except Exception as e:
                    logger.warning(f"Response generation attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to generate response after {max_retries} attempts")
                        return "I apologize, but I'm having trouble connecting to my knowledge base right now. Could you please try again in a moment?"

        except Exception as e:
            logger.error(f"Error in answering question: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try asking again or rephrase your question."

    def get_sources(self) -> List[Dict]:
        """Return sources used in the last response."""
        return self.last_sources

    def print_sources(self) -> None:
        """Display sources used in the last response."""
        if not self.last_sources:
            print("\nSources: None used.")
            return
            
        print("\nSources:")
        for i, source in enumerate(self.last_sources[:3], 1):
            print(f"{i}. {source['document']} (Page {source['page']}) — Score: {source['score']}")

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
        return "Conversation history has been cleared."
        
    def get_stats(self) -> Dict:
        """Return statistics about the assistant's usage."""
        return {
            "queries_processed": self.query_count,
            "session_start": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration": str(datetime.now() - self.start_time).split('.')[0],
            "knowledge_base_size": len(self.page_ids)
        }

def main(
    conversation_history: str = None,
    user_input: str = None,
    embeddings_path: str = None,
    budget_info: str = None
) -> dict:
    """
    Process NDIS Assistant Bot queries and return a dictionary response for backend use.
    """
    response_dict = {
        "NDIS Assistant": "",
        "Sources": [],
        "Header": "\n" + "="*50 + "\nNDIS Assistant Bot (OpenAI)\n" + "="*50 + "\n",
        "Footer": "\n" + "-"*50,
        "Status": "success",
        "Extra": ""
    }

    if not user_input:
        response_dict["NDIS Assistant"] = "Please provide a question or command."
        return response_dict

    if not embeddings_path or not budget_info:
        response_dict["NDIS Assistant"] = "Error: Missing embeddings path or budget information."
        response_dict["Status"] = "error"
        return response_dict

    try:
        chatbot = NDISAssistantBotOpenAI(
            embeddings_path=embeddings_path,
            budget_info=budget_info,
            conversation_history=conversation_history,
            top_k=5,
            similarity_threshold=0.4
        )

        user_query = user_input.strip()

        if user_query.lower() == '/exit':
            response_dict["NDIS Assistant"] = "Thank you for using the NDIS Assistant. Goodbye!"
            
        elif user_query.lower() == '/clear':
            response_dict["NDIS Assistant"] = chatbot.clear_conversation()
            
        elif user_query.lower() == '/stats':
            stats = chatbot.get_stats()
            stats_text = "Session Statistics:\n"
            for key, value in stats.items():
                stats_text += f"- {key.replace('_', ' ').title()}: {value}\n"
            response_dict["NDIS Assistant"] = stats_text.strip()
            
        elif user_query.lower().startswith('/search '):
            search_query = user_query[8:].strip()
            if search_query:
                results = chatbot.debug_search(search_query)
                extra_text = f"Debug searching for: '{search_query}'\n\n"
                extra_text += "Search Results:\n"
                extra_text += f"Query processing time: {results['embedding_time_ms']}ms (embedding) + {results['similarity_calc_time_ms']}ms (matching)\n"
                extra_text += f"Similarity threshold: {results['threshold']}\n"
                extra_text += f"Would retrieve content: {'Yes' if results.get('would_retrieve') else 'No'}\n"
                extra_text += "\nTop 10 matches:\n"
                for i, result in enumerate(results.get('top_results', []), 1):
                    extra_text += f"{i}. Score: {result['score']} - {result['file']} (Page {result['page']})\n"
                    extra_text += f"   Preview: {result['preview']}\n"
                response_dict["Extra"] = extra_text.strip()
            else:
                response_dict["NDIS Assistant"] = "Please provide a search query after /search"
                
        elif not user_query:
            response_dict["NDIS Assistant"] = "Please type your question or type /exit to quit."
            
        else:
            response_dict["NDIS Assistant"] = chatbot.answer_question(user_query)
            sources = chatbot.get_sources()
            """ if sources:
                response_dict["Sources"] = [
                    f"{i}. {source['document']} (Page {source['page']}) — Score: {source['score']}"
                    for i, source in enumerate(sources[:3], 1)
                ] """

        return response_dict

    except Exception as e:
        logger.critical(f"Error in main: {str(e)}")
        response_dict["NDIS Assistant"] = f"Error: {str(e)}\nPlease check the configuration and try again."
        response_dict["Status"] = "error"
        response_dict["Extra"] = ""
        return response_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDIS Assistant Bot")
    parser.add_argument("text_input", nargs="?", default="", help="User's text input")
    parser.add_argument("--upload", help="Path to PDF file")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--budget", help="User's budget information as JSON string")
    parser.add_argument("--history", help="Conversation history as a string")
    args = parser.parse_args()

    embeddings_path = "C:\\Users\\bdCalling\\Desktop\\Shuvo\\Sam_Fair_Hai_Helper_App\\samfair\\knowledge_base_embeddings_openai.npz"

    response = main(
        conversation_history=args.history,
        user_input=args.text_input,
        embeddings_path=embeddings_path,
        budget_info=args.budget if args.budget else "[]"
        # budget_info=budget_info
    )

    print("AI Response:", response["NDIS Assistant"])
    if response["Sources"]:
        print("\nSources:")
        for source in response["Sources"]:
            print(source)
    if response["Extra"]:
        print(f"\n{response['Extra']}")