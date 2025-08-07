"""
Routes package for Titans Finance API

Contains all API route definitions organized by functionality.
"""

from .prediction_routes import router as prediction_router
from .model_routes import router as model_router

__all__ = ['prediction_router', 'model_router']