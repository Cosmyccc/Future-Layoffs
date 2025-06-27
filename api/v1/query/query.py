from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
from app.controllers import ProcessController
import tempfile
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.controllers import QuestionController, QuestionContext
from app.services.utils import ServerUtils
from core.shared_state import GlobalState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class Question(BaseModel):
    question: str

query_router = APIRouter()

@query_router.get("/", tags=["Query"])
async def health():
    return {"message": "Query working properly !!!"}

@query_router.post("/ask-question", tags=["Query"])
async def ask_question(request: Question, state: GlobalState = Depends(GlobalState)):
    if state.get_github_url() is None:
        raise HTTPException(status_code=400, detail="GitHub URL is not available. Call /process_repository first.")
    
    with tempfile.TemporaryDirectory() as local_path:
        clone = await ProcessController.clone_repository(state.get_github_url(), local_path)
        if not clone:
            raise HTTPException(status_code=400, detail="Failed to clone repository.")
        
        index, documents, file_type_counts, filenames = await ProcessController.load_and_index_files(local_path)
        if index is None:
            raise HTTPException(status_code=400, detail="No documents found to index.")
     
    
    # Create a Gemini model instance
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-pro")

    template = """
    Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {user_input} | FileCount: {file_type_counts} | FileNames: {filenames}
    Instr:
    1. Answer based on context/docs.
    2. Focus on repo/code.
    3. Consider:
        a. Purpose/features - describe.
        b. Functions/code - provide details/samples.
        c. Setup/usage - give instructions.
    4. Unsure? Say "I am not sure".
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["g_repo_name", "g_github_url", "conversation_history", "user_input", "numbered_documents", "file_type_counts", "filenames"]
    )

    conversation_history = ""  # You need to define the conversation history here
    question_context = QuestionContext(index, documents, llm, state.get_repo_name(), state.get_github_url(), conversation_history, file_type_counts, filenames)

    user_input = ServerUtils.format_user_question(request.question)
    
    answer = await QuestionController.ask_question(user_input, question_context)
    
    return {"answer": answer}
