from fastapi import FastAPI
from api import router


def init_routers(app_: FastAPI) -> None:
    app_.include_router(router)
    
    
def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Future Layoffs",
        description="Future Layoff's API documentation",
        version="1.0.0",
        docs_url= "/docs",
        redoc_url="/redoc",
    )
    init_routers(app_=app_)
    return app_


app = create_app()