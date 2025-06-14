# main.py
import os
import json
import base64
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import chromadb
# Import the Google Generative AI client library
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

# Configure the Google Generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# Define the embedding model for Gemini (must match what was used in prepare_data_for_rag.py)
GEMINI_EMBEDDING_MODEL = "models/embedding-001" 
# For chat, using a multimodal model to handle potential image input
# 'gemini-1.5-flash' is a good balance of performance and cost with multimodal support.
# 'gemini-pro-vision' is an older multimodal model.
# 'gemini-1.5-pro' is a more capable, but potentially more expensive, multimodal model.
GEMINI_CHAT_MODEL = "gemini-1.5-flash" 

CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "tds_course_data"

# Define the embedding function for ChromaDB to use Gemini API
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name=GEMINI_EMBEDDING_MODEL
)

# Initialize ChromaDB client and load the collection globally
# This ensures it's loaded once when the app starts
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    tds_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=gemini_ef)
    print(f"Successfully loaded ChromaDB collection: '{CHROMA_COLLECTION_NAME}' with {tds_collection.count()} documents.")
except Exception as e:
    print(f"Error loading ChromaDB collection: {e}. Please ensure `prepare_data_for_rag.py` was run correctly and the 'chroma_db' directory exists.")
    tds_collection = None # Set to None if loading fails, handle gracefully in API endpoint

# Initialize Gemini GenerativeModel for chat completions
gemini_chat_model = genai.GenerativeModel(GEMINI_CHAT_MODEL)

app = FastAPI(title="TDS Virtual TA API")

# --- Pydantic Models for API Request/Response ---
class QuestionRequest(BaseModel):
    question: str
    # Image is base64 encoded string. Use Optional as it might not always be provided.
    image: Optional[str] = Field(None, description="Base64 encoded image (e.g., JPEG, PNG, WebP)")

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

# --- Helper to clean text for display/LLM input ---
def clean_text_for_llm(text):
    """Basic cleaning to remove excessive whitespace and common HTML remnants."""
    if not isinstance(text, str):
        return ""
    # Remove any remaining HTML tags (e.g., from cooked Discourse posts)
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple spaces/newlines with single space
    return re.sub(r'\s+', ' ', text).strip()


# --- API Endpoint ---
@app.post("/api/", response_model=AnswerResponse)
async def answer_student_question(request: QuestionRequest):
    if tds_collection is None:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded. Please ensure data preparation is complete.")

    try:
        # Prepare content for Gemini chat model
        chat_contents = []

        # Add the student's text question
        chat_contents.append({"text": request.question})

        # Handle image attachment if provided
        if request.image:
            try:
                # Decode base64 image data
                image_bytes = base64.b64decode(request.image)
                # For Gemini, the image needs to be a PIL Image object
                # It's important to specify the correct MIME type based on the actual image.
                # The prompt example had 'project-tds-virtual-ta-q1.webp' so using 'image/webp'
                # If your images are usually PNG or JPEG, change this accordingly.
                image_mime_type = "image/webp" # Assuming webp based on previous context, adjust if different
                img = Image.open(io.BytesIO(image_bytes))
                chat_contents.append(img)
                print("Image successfully processed and added to chat_contents.")
            except Exception as e:
                print(f"Warning: Could not process image: {e}")
                # If image processing fails, still allow the text-based question
                chat_contents.append({"text": "(Student attempted to send an image, but it could not be processed.)"})


        # 1. Embed the student's question for retrieval (ChromaDB handles this internally on query_texts)
        # 2. Retrieve relevant chunks from vector DB
        retrieval_results = tds_collection.query(
            query_texts=[request.question], # ChromaDB uses its configured embedding function here
            n_results=5, # Retrieve top 5 most relevant chunks
            include=['documents', 'metadatas'] # Request text content and metadata
        )

        context_chunks_formatted = []
        links_for_response = []
        
        if retrieval_results and retrieval_results['documents'] and retrieval_results['documents'][0]:
            # Each item in retrieval_results['documents'][0] is a text string.
            # Each item in retrieval_results['metadatas'][0] is a dictionary of metadata.
            for i, doc_content in enumerate(retrieval_results['documents'][0]):
                metadata = retrieval_results['metadatas'][0][i]
                
                # Clean the content retrieved from the DB before passing to LLM
                cleaned_doc_content = clean_text_for_llm(doc_content)

                context_chunks_formatted.append(
                    f"--- Source {i+1} ---\n"
                    f"Title: {metadata.get('title', 'N/A')}\n"
                    f"Type: {metadata.get('type', 'N/A')}\n"
                    f"URL: {metadata.get('source_url', 'N/A')}\n"
                    f"Content: {cleaned_doc_content}"
                )
                
                links_for_response.append(Link(
                    url=metadata.get('source_url', '#'),
                    text=metadata.get('title', 'Relevant Content')
                ))
        else:
            context_chunks_formatted.append("No highly relevant context found in the knowledge base.")


        # 3. Construct LLM Prompt for Gemini
        # Gemini takes a list of parts as `contents`.
        # The first part can be a system instruction or a general preamble.
        # Then, the user's question, image (if applicable), and context are added.

        # System/Preamble for the LLM
        system_instruction = (
            "You are a helpful and clever Teaching Assistant for IIT Madras's Online Degree in Data Science, "
            "specifically for the 'Tools in Data Science' course. "
            "Your goal is to answer student questions based *only* on the provided context, which includes "
            "course content and Discourse forum discussions. "
            "If the answer cannot be found in the provided context, politely state that you don't have enough information. "
            "Be concise, direct, and helpful. Prioritize information directly from the provided context. "
            "When referring to sources, you can mention them in your answer if it sounds natural, but prioritize answering the question directly. "
            "If the student's question involves an image, integrate information from the image into your understanding. "
            "Do not invent information. Your response should be solely derived from the provided context."
        )
        
        # Combine retrieved context into a single string for the LLM's text part
        context_for_llm = "\n\n".join(context_chunks_formatted)

        # Build the final prompt content for the LLM
        # The initial chat_contents already contains the text question and potential image.
        # Now, add the system instruction and the context from RAG.
        # Gemini's `generate_content` is more conversational than completion for multimodal.
        # It's recommended to structure turn-based interaction.
        # For a single query, we combine system instruction and context into the first user turn if it's the only one.
        
        # Gemini models often work best with clear turns.
        # We'll put the system instruction in a "text" part, followed by the user query parts.
        
        # The first 'part' can be the system instruction for the model to follow
        combined_contents = [
            {"text": system_instruction}, # Preamble/system instruction
            *chat_contents, # The original user text question and image
            {"text": f"\n\nHere is additional relevant context from the course materials and forum discussions:\n{context_for_llm}"}
        ]

        # 4. Call Gemini LLM for chat completions
        gemini_response = gemini_chat_model.generate_content(
            combined_contents,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Lower temperature for more factual, less creative responses
                max_output_tokens=800, # Limit response length
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        llm_answer = gemini_response.text.strip()
        
        # In case Gemini returns content in 'parts' for text (e.g., if it's a list)
        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            if hasattr(gemini_response.candidates[0].content, 'parts') and gemini_response.candidates[0].content.parts:
                # Ensure we get text from parts. This handles cases where text might be split.
                llm_answer = "".join([part.text for part in gemini_response.candidates[0].content.parts if hasattr(part, 'text')]).strip()
            else:
                llm_answer = "" # No text parts found

        if not llm_answer:
            llm_answer = "I apologize, but I could not generate a clear answer based on the provided information."

        return AnswerResponse(answer=llm_answer, links=links_for_response)

    except HTTPException as e:
        raise e # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        print(f"An unexpected error occurred in the API: {e}")
        # Log the full exception for debugging in production
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Run the application (for local testing) ---
# To run locally: uvicorn main:app --reload
# This block is for local execution only and won't be used in Cloud Run deployment.
if __name__ == "__main__":
    import uvicorn
    print("\nTo test locally, run `uvicorn main:app --reload`")
    print("Or for production-like behavior (no reload), run this script directly:")
    uvicorn.run(app, host="0.0.0.0", port=8000)

