"""Centralized exception handlers for TrustRAG FastAPI backend."""

import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.agents.base import AgentError, AgentExecutionError, AgentInputError

logger = logging.getLogger(__name__)


def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    """Standardized JSON error envelope."""
    content: Dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
        }
    }
    if details is not None:
        content["error"]["details"] = details

    return JSONResponse(status_code=status_code, content=content)


async def agent_input_error_handler(request: Request, exc: AgentInputError) -> JSONResponse:
    """Handle agent input validation errors."""
    logger.warning(f"Agent input validation error on {request.url.path}: {exc}")
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_type="AgentInputError",
        message=str(exc),
    )


async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    """Handle general agent runtime errors without leaking stack traces."""
    logger.error(f"Agent runtime failure on {request.url.path}: {exc}", exc_info=True)
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="AgentExecutionError",
        message=f"Agent execution encountered an internal failure: {str(exc)}",
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic/FastAPI request validation errors."""
    logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
    # Format readable error messages
    error_messages = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{loc}: {msg}")

    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_type="ValidationError",
        message="Request payload failed validation schema requirements.",
        details=error_messages,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle standard FastAPI and Starlette HTTPExceptions."""
    return create_error_response(
        status_code=exc.status_code,
        error_type="HTTPException",
        message=str(exc.detail),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions, preventing raw stack trace exposure."""
    logger.critical(f"Unhandled server exception on {request.url.path}: {exc}", exc_info=True)
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="InternalServerError",
        message="An unexpected server error occurred while processing the request.",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers on the FastAPI application instance."""
    app.add_exception_handler(AgentInputError, agent_input_error_handler)  # type: ignore
    app.add_exception_handler(AgentError, agent_error_handler)  # type: ignore
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
    app.add_exception_handler(Exception, unhandled_exception_handler)
