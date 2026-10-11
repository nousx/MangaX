"""Shared editor image LRU and prefetching."""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from manga_translator.utils import open_pil_image
from PIL import Image

from .resources import ImageResource


def _release_gpu_memory():
    """Free GPU memory"""
    try:
        import torch

        if torch.cuda.is_available():
            pass
            pass
    except ImportError:
        pass
    except Exception:
        pass


def _trim_working_set() -> bool:
    """Ask Windows to trim the working set of the current process."""
    try:
        import ctypes
        import os

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process_id = os.getpid()
        process_handle = kernel32.OpenProcess(0x0400 | 0x0100, False, process_id)
        if not process_handle:
            return False

        empty_working_set_ok = False
        try:
            if hasattr(psapi, "EmptyWorkingSet"):
                empty_working_set_ok = bool(psapi.EmptyWorkingSet(process_handle))
                if empty_working_set_ok:
                    return True

            set_ws_ok = bool(kernel32.SetProcessWorkingSetSize(process_handle, -1, -1))
            if set_ws_ok:
                return True
            return False
        finally:
            kernel32.CloseHandle(process_handle)
    except Exception:
        return False


def _current_process_memory_bytes() -> int:
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        rss = getattr(info, "rss", 0) or 0
        wset = getattr(info, "wset", 0) or 0
        return max(rss, wset)
    except Exception:
        return 0


def _estimate_image_bytes(image: Image.Image | None) -> int:
    if image is None:
        return 0
    try:
        channels = max(1, len(image.getbands()))
        return int(image.width) * int(image.height) * channels
    except Exception:
        return 0


class ResourceManager:
    """Shared editor image LRU and prefetch store."""

    def __init__(self):
        """Initialise the resource manager"""
        self.logger = logging.getLogger(__name__)

        # The image cache and current are accessed by the read-ahead thread, the load thread and the main thread at the same time
        # and are protected by a re-entrant lock; decoding (open_pil_image) always happens outside the lock, which only guards dict reads and writes.
        self._lock = threading.RLock()

        # The resource currently loaded
        self._current_image: Optional[ImageResource] = None

        # Resource cache (for fast switching)
        self._image_cache: Dict[str, ImageResource] = {}
        self._cache_limit = 5  # At most 5 images are cached

        self._export_cleanup_threshold_bytes = 2 * 1024 * 1024 * 1024

    # ==================== Image management ====================

    @staticmethod
    def _resolve_image_path(image_path: str) -> str:
        """Normalise an image path and check that it exists."""
        from pathlib import Path

        path_obj = Path(image_path)
        if not path_obj.exists():
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

        return str(path_obj.resolve())

    def load_image(self, image_path: str) -> ImageResource:
        """Load the base image resource being edited and update current_image."""
        image_path = self._resolve_image_path(image_path)

        with self._lock:
            cached = self._image_cache.get(image_path)
            if cached is not None:
                self.logger.debug(f"Image loaded from cache: {image_path}")
                cached.touch()
                self._current_image = cached
                return cached

        try:
            self.logger.debug(f"Loading image: {image_path}")
            # Decoding happens outside the lock: a slow operation should not block other threads reading the cache
            image = open_pil_image(image_path, eager=True)
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise

        with self._lock:
            # Double check: another thread (such as read-ahead) may have put it in the cache while decoding
            cached = self._image_cache.get(image_path)
            if cached is not None:
                cached.touch()
                self._current_image = cached
                return cached

            resource = ImageResource(
                path=image_path,
                image=image,
                width=image.width,
                height=image.height,
            )
            self._add_to_cache(image_path, resource)
            self._current_image = resource
            self.logger.debug(
                f"Image loaded successfully: {image_path} ({image.width}x{image.height})"
            )
            return resource

    def prefetch_image(self, image_path: str) -> ImageResource:
        """Read an image resource ahead into the LRU, without switching current_image."""
        image_path = self._resolve_image_path(image_path)

        with self._lock:
            cached = self._image_cache.get(image_path)
            if cached is not None:
                return cached

        image = open_pil_image(image_path, eager=True)

        with self._lock:
            cached = self._image_cache.get(image_path)
            if cached is not None:
                return cached

            resource = ImageResource(
                path=image_path,
                image=image,
                width=image.width,
                height=image.height,
            )
            self._add_to_cache(image_path, resource)
            return resource

    def activate_prefetched_image(
        self,
        image_path: str,
        image: Image.Image,
        *,
        qimage: Any = None,
    ) -> ImageResource:
        """Pin a prefetched document image without decoding it again."""
        image_path = self._resolve_image_path(image_path)
        with self._lock:
            resource = self._image_cache.get(image_path)
            if resource is None or resource.image is None:
                resource = ImageResource(
                    path=image_path,
                    image=image,
                    width=image.width,
                    height=image.height,
                    qimage=qimage,
                )
                self._add_to_cache(image_path, resource)
            elif resource.qimage is None and qimage is not None:
                resource.qimage = qimage
            resource.touch()
            self._current_image = resource
            return resource

    def load_detached_image(self, image_path: str) -> Image.Image:
        """Load an auxiliary image, without writing current_image and without polluting the cache."""
        image_path = self._resolve_image_path(image_path)
        self.logger.debug(f"Loading detached image: {image_path}")
        return open_pil_image(image_path, eager=True)

    def _add_to_cache(self, path: str, resource: ImageResource) -> None:
        """Add an image to the cache (the caller must hold self._lock)

        Eviction is a true LRU: the entry with the oldest last_access goes, and **the current page is never evicted** -
        evicting it would disconnect _current_image from the cache, and it would have to be decoded again on switching back.

        Args:
            path: the image path
            resource: the image resource
        """
        if len(self._image_cache) >= self._cache_limit:
            current = self._current_image
            candidates = [
                (cached_path, cached)
                for cached_path, cached in self._image_cache.items()
                if cached is not current and cached_path != path
            ]
            if candidates:
                oldest_path = min(candidates, key=lambda item: item[1].last_access)[0]
                old_resource = self._image_cache.pop(oldest_path)
                old_resource.release()
                self.logger.debug(
                    f"Removed least recently used image from cache: {oldest_path}"
                )
            else:
                # Edge case: when only the current page is left in the cache, exceed the limit rather than evict it
                self.logger.debug(
                    "Cache eviction skipped: only the current image is cached"
                )

        self._image_cache[path] = resource

    def release_image_from_cache(self, path: str) -> bool:
        """Release the given image from the cache

        Args:
            path: the image path

        Returns:
            bool: whether it was released
        """
        from pathlib import Path

        # Normalise the path so it matches the keys in the cache
        path = str(Path(path).resolve())
        with self._lock:
            resource = self._image_cache.pop(path, None)
            if resource is None:
                return False
            resource.release()
        self.logger.debug(f"Released image from cache: {path}")
        return True

    def clear_image_cache(self) -> None:
        """Clear the whole image cache"""
        with self._lock:
            for resource in self._image_cache.values():
                resource.release()
            self._image_cache.clear()
        _release_gpu_memory()

    def release_image_cache_except_current(self, force: bool = False) -> int:
        """Keep only the current image and release the other images in image_cache."""
        if (
            not force
            and _current_process_memory_bytes() < self._export_cleanup_threshold_bytes
        ):
            return 0

        removed = 0
        with self._lock:
            current = self._current_image
            current_path = current.path if current is not None else None

            for path in list(self._image_cache.keys()):
                if path == current_path:
                    continue
                resource = self._image_cache.pop(path, None)
                if resource is not None and resource is not current:
                    resource.release()
                    removed += 1
        return removed

    def unload_image(self, release_from_cache: bool = False) -> None:
        """Unload the current image and all resources linked to it

        Args:
            release_from_cache: whether the image is released from the cache as well
        """
        with self._lock:
            if self._current_image:
                current_path = self._current_image.path

                # When it has to be released from the cache
                if release_from_cache and current_path in self._image_cache:
                    resource = self._image_cache.pop(current_path)
                    resource.release()
                    self.logger.debug(f"Released image from cache: {current_path}")

                self._current_image = None

        if release_from_cache:
            self.clear_image_cache()

        _release_gpu_memory()

        self.logger.debug("Image unloaded and memory released")

    def get_current_image(self) -> Optional[ImageResource]:
        """Get the current image resource

        Returns:
            Optional[ImageResource]: the current image resource, or None when nothing is loaded
        """
        with self._lock:
            return self._current_image

    def get_managed_images(self) -> List[Image.Image]:
        """Return the image objects the resource manager still holds."""
        images: List[Image.Image] = []
        with self._lock:
            if (
                self._current_image is not None
                and getattr(self._current_image, "image", None) is not None
            ):
                images.append(self._current_image.image)
            cached_resources = list(self._image_cache.values())
        for resource in cached_resources:
            image = getattr(resource, "image", None)
            if image is not None and not any(image is existing for existing in images):
                images.append(image)
        return images

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Return image-LRU ownership metrics."""
        managed_images = self.get_managed_images()
        managed_image_bytes = sum(
            _estimate_image_bytes(image) for image in managed_images
        )
        return {
            "process_bytes": _current_process_memory_bytes(),
            "managed_image_count": len(managed_images),
            "managed_image_bytes": managed_image_bytes,
            "image_cache_entries": len(self._image_cache),
            "current_image_path": (
                self._current_image.path if self._current_image is not None else None
            ),
        }

    def log_memory_snapshot(self, stage: str, logger=None) -> Dict[str, Any]:
        target_logger = logger or self.logger
        if not target_logger.isEnabledFor(logging.DEBUG):
            return {}
        snapshot = self.get_memory_snapshot()
        target_logger.debug(
            "Memory snapshot [%s]: process=%.2fMB managed_images=%s managed=%.2fMB",
            stage,
            snapshot["process_bytes"] / (1024 * 1024),
            snapshot["managed_image_count"],
            snapshot["managed_image_bytes"] / (1024 * 1024),
        )
        return snapshot

    # ==================== Resource clean-up ====================

    def cleanup_all(self) -> None:
        """Release the image LRU."""
        with self._lock:
            self._current_image = None
            for resource in self._image_cache.values():
                resource.release()
            self._image_cache.clear()
        _release_gpu_memory()

    def release_memory_after_export(self) -> None:
        """Trim process memory under pressure while retaining the image LRU."""
        if _current_process_memory_bytes() < self._export_cleanup_threshold_bytes:
            return
        import gc

        gc.collect()
        _release_gpu_memory()
        _trim_working_set()

    def __del__(self):
        """Destructor"""
        self.cleanup_all()
