from fastapi import APIRouter
from .query import query_router

v1_query_router = APIRouter()  
v1_query_router.include_router(query_router, prefix="", tags=["Query"]) 

__all__ = ["v1_query_router"]
