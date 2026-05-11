from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..runtime import APP_DIR

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/basic", response_class=HTMLResponse)
def demo_page_basic(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/experience", response_class=HTMLResponse)
def experience_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/experience/app", response_class=HTMLResponse)
def experience_app(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})


@router.get("/management", response_class=HTMLResponse)
def management_page(request: Request):
    return templates.TemplateResponse("management.html", {"request": request})


@router.get("/login", response_class=HTMLResponse)
def demo_page_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/app", response_class=HTMLResponse)
def demo_page_app_main(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})


@router.get("/", response_class=HTMLResponse)
def root_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
