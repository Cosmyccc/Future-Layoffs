from fastapi import APIRouter
from .process import process_router

v1_process_router = APIRouter()  
v1_process_router.include_router(process_router, prefix="", tags=["Process"]) 

__all__ = ["v1_process_router"]
