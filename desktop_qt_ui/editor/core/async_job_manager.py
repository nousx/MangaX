"""Manager of the asynchronous event loop."""

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Coroutine, Optional, Set


class AsyncJobManager:
    """Manages the dedicated event loop thread used by the editor's background coroutines."""
    
    def __init__(self):
        """Initialise the asynchronous job manager"""
        self.logger = logging.getLogger(__name__)
        
        # Set of the Futures returned by run_coroutine_threadsafe
        self._futures: Set[Future] = set()
        self._lock = threading.RLock()
        
        # Event loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stopping = False
        
        # Start the event loop
        self._start_event_loop()
    
    def _start_event_loop(self) -> None:
        """Start the event loop thread"""
        if self._running:
            return
        
        def run_loop():
            """Run the event loop in the thread"""
            import sys
            # In a worker thread on Windows, Windows Sockets has to be initialised by hand
            if sys.platform == 'win32':
                # Call WSAStartup directly through ctypes
                import ctypes
                try:
                    WSADATA_SIZE = 400
                    wsa_data = ctypes.create_string_buffer(WSADATA_SIZE)
                    ws2_32 = ctypes.WinDLL('ws2_32')
                    ws2_32.WSAStartup(0x0202, wsa_data)
                except Exception:
                    pass

                self._loop = asyncio.WindowsProactorEventLoopPolicy().new_event_loop()
            else:
                self._loop = asyncio.new_event_loop()

            asyncio.set_event_loop(self._loop)

            def handle_loop_exception(loop, context):
                exc = context.get("exception")
                if (
                    self._stopping
                    and isinstance(exc, ConnectionAbortedError)
                    and getattr(exc, "winerror", None) == 1236
                ):
                    self.logger.debug("Ignored expected Proactor self-pipe abort during shutdown")
                    return
                loop.default_exception_handler(context)

            self._loop.set_exception_handler(handle_loop_exception)
            self._running = True
            try:
                self._loop.run_forever()
            finally:
                try:
                    if self._loop and not self._loop.is_closed():
                        self._loop.close()
                except Exception as e:
                    self.logger.warning(f"Error closing event loop: {e}")
                finally:
                    asyncio.set_event_loop(None)
                    self._running = False
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Wait for the event loop to start
        import time
        timeout = 5.0
        start_time = time.time()
        while not self._loop and time.time() - start_time < timeout:
            time.sleep(0.01)
        
        if not self._loop:
            raise RuntimeError("Failed to start event loop")
    
    def submit_coroutine(self, coro: Coroutine) -> Optional[Future]:
        """Submit a coroutine to the dedicated event loop."""
        if not self._running or self._loop is None:
            return None

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._discard_future)
        return future

    def _discard_future(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)
    
    def cancel_all(self) -> int:
        with self._lock:
            futures = [future for future in self._futures if not future.done()]

        cancelled = 0
        for future in futures:
            if future.cancel():
                cancelled += 1
        return cancelled

    async def _drain_loop(self) -> None:
        """Cancel and collect the pending tasks of the event loop."""
        current_task = asyncio.current_task()
        pending = [
            task for task in asyncio.all_tasks()
            if task is not current_task and not task.done()
        ]

        for task in pending:
            task.cancel()

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            await asyncio.get_running_loop().shutdown_asyncgens()
        except Exception as e:
            self.logger.debug(f"Error shutting down async generators: {e}")

        shutdown_executor = getattr(asyncio.get_running_loop(), "shutdown_default_executor", None)
        if shutdown_executor is not None:
            try:
                await shutdown_executor()
            except Exception as e:
                self.logger.debug(f"Error shutting down default executor: {e}")
    
    def shutdown(self, wait: bool = True) -> None:
        """Shut down the job manager

        Args:
            wait: whether to wait for all tasks to finish
        """
        if not self._running:
            return
        
        self._stopping = True
        
        if not wait:
            self.cancel_all()
        
        # Stop the event loop
        if self._loop and not self._loop.is_closed():
            try:
                drain_future = asyncio.run_coroutine_threadsafe(self._drain_loop(), self._loop)
                drain_future.result(timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Error draining event loop: {e}")

            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                # The event loop may have stopped already
                pass
        
        # Wait for the thread to end
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        with self._lock:
            self._futures.clear()

        self._loop = None
        self._thread = None
        self._running = False
        self._stopping = False

    
    def __del__(self):
        """Destructor.

        While the interpreter exits, the object may be half-initialised (__init__ failed midway) or the modules it
        depends on may already be gone, so every access is defensive and no exception ever leaks from __del__.
        """
        try:
            if getattr(self, "_running", False):
                self.shutdown(wait=False)
        except Exception:
            # Ignore errors during destruction
            pass

