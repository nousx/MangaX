"""Editor core module

This module holds the core data structures, type definitions and managers of the editor.
"""

from .async_job_manager import AsyncJobManager
from .resource_manager import ResourceManager
from .resources import ImageResource
from .types import MaskType

__all__ = [
    # Types
    "MaskType",
    # Resources
    "ImageResource",
    # Managers
    "AsyncJobManager",
    "ResourceManager",
]

