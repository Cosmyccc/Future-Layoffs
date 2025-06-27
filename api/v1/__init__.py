from fastapi import APIRouter

from .process import process_router
from .query import query_router


v1_router = APIRouter()
v1_router.include_router(process_router, prefix="/process")
v1_router.include_router(query_router,prefix="/query")