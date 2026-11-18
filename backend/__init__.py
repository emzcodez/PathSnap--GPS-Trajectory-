"""PathSnap Backend Package

Joint Ownership: Prateek & Priyansh
Provides FastAPI application, route prediction endpoint, and evaluator evaluation service.
"""

from .api import router as api_router
from .evaluator import ServerSideEvaluator

__all__ = ["api_router", "ServerSideEvaluator"]

