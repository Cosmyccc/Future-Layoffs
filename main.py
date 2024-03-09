from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.cors import CORSMiddleware
from langchain import PromptTemplate, LLMChain
from api.index import *
import os
import tempfile
from fastapi.responses import JSONResponse
from langchain.llms import Cohere
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*", "Access-Control-Allow-Origin"],
)

g_github_url = None
g_repo_name = None

@app.get('/')
async def root():
    return {'message': 'Hello World!'}

@app.post('/process_repository')
async def process_repository(data: dict):
    global g_github_url, g_repo_name
    
    github_url = data.get('github_url')
    g_github_url = github_url
    repo_name = github_url.split("/")[-1]
    g_repo_name = repo_name
    
    with tempfile.TemporaryDirectory() as local_path:
        if github_clone_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                raise HTTPException(status_code=400, detail="No documents found to index.")

    return {"message": "Repository processed successfully."}

@app.post('/ask_question')
async def ask_question_endpoint(request: Request):
    data = await request.json()
    global g_github_url, g_repo_name
    
    if g_github_url is None:
        raise HTTPException(status_code=400, detail="GitHub URL is not available. Call /process_repository first.")
    
    with tempfile.TemporaryDirectory() as local_path:
        if github_clone_repo(g_github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                raise HTTPException(status_code=400, detail="No documents found to index.")
    
    user_input = data.get('user_input')
    
    cohere_api_key = os.getenv("Cohere_api_key")
    llm = Cohere(cohere_api_key=cohere_api_key, model="command", temperature=0.6)

    template = """
    Repo: {repo_name} ({repo_name}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {user_input} | FileCount: {file_type_counts} | FileNames: {filenames}

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

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    conversation_history = ""
    question_context = QuestionContext(index, documents, llm_chain, model_name, g_repo_name, g_github_url, conversation_history, file_type_counts, filenames)

    user_input = format_user_question(user_input)
    
    answer = ask_question(user_input, question_context)
    
    return JSONResponse(content={"answer": answer}, status_code=200)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)