from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from app.controllers import ProcessController
import tempfile
from core.shared_state import GlobalState

class ProcessRepositoryRequest(BaseModel):
    github_url: str  

process_router = APIRouter()

@process_router.post("/process_repository", tags=["Process"])
async def process_repository(request: ProcessRepositoryRequest, state: GlobalState = Depends(GlobalState)):
    github_url = request.github_url
    
    if not github_url:
        raise HTTPException(status_code=400, detail="github_url is required.")
    
    state.set_github_url(github_url)
    repo_name = github_url.split("/")[-1]
    state.set_repo_name(repo_name)
    
    with tempfile.TemporaryDirectory() as local_path:
        print(local_path)
        clone = await ProcessController.clone_repository(github_url, local_path)
        if clone:  
            index, documents, file_type_counts, filenames = await ProcessController.load_and_index_files(local_path)
            print("FileName:",filenames)
            if index is None:
                raise HTTPException(status_code=400, detail="No documents found to index.")

    return {"message": "Repository processed successfully."}


