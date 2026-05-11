from __future__ import annotations

from fastapi import FastAPI

from .routers.admin import router as admin_router
from .routers.auth import router as auth_router
from .routers.pages import router as pages_router
from .routers.public import router as public_router
from .routers.ux import router as ux_router
from .runtime import startup

app = FastAPI(title="Yelp Hybrid Recommender API", version="2.1.0")
app.add_event_handler("startup", startup)

for router in [
    public_router,
    auth_router,
    ux_router,
    admin_router,
    pages_router,
]:
    app.include_router(router)
