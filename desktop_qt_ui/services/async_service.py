"""Service for submitting background coroutines from Qt's synchronous context."""
import logging
from typing import Coroutine, Optional

# Absolute import, to avoid relative import problems
from desktop_qt_ui.editor.core import AsyncJobManager


class AsyncService:
    """Compatibility layer over AsyncJobManager."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._job_manager = AsyncJobManager()
        self._running = True

    def submit_task(self, coro: Coroutine):
        """Submit a coroutine to the background event loop."""
        if not self._running:
            self.logger.warning("AsyncService is not running, task ignored")
            coro.close()
            return None

        try:
            future = self._job_manager.submit_coroutine(coro)
            if future is None:
                self.logger.error("Event loop is not available")
                coro.close()
                return None

            self.logger.debug("Task submitted to event loop")
            return future
        except Exception as e:
            try:
                coro.close()
            except Exception:
                pass
            self.logger.error(f"Failed to submit task: {e}", exc_info=True)
            return None
    
    def cancel_all_tasks(self):
        """Cancel all active asynchronous tasks (non-blocking)"""
        if not self._running:
            return
        
        try:
            self._job_manager.cancel_all()
        except Exception as e:
            self.logger.error(f"Error cancelling tasks: {e}")

    def shutdown(self):
        """Shut down the service"""
        self._running = False
        self._job_manager.shutdown(wait=False)

# Global instance
_async_service: Optional[AsyncService] = None

def get_async_service() -> AsyncService:
    global _async_service
    if _async_service is None:
        _async_service = AsyncService()
    return _async_service

def shutdown_async_service():
    global _async_service
    if _async_service:
        _async_service.shutdown()
        _async_service = None
