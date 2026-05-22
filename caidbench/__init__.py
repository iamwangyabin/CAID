"""CAIDBench: continual AI-generated/deepfake image detection benchmark framework."""

from .registry import build_method, list_methods, register_method

__all__ = ["build_method", "list_methods", "register_method"]
