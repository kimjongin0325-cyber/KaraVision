from fastapi import FastAPI

from karawm.server.lifespan import lifespan
from karawm.server.router import router


def init_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
