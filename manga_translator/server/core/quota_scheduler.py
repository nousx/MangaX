"""
Quota scheduler (QuotaScheduler)

Manages the scheduled tasks related to quotas, including the daily quota reset.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class QuotaScheduler:
    """Quota scheduler - manages the scheduled tasks related to quotas"""
    
    def __init__(self, quota_service):
        """
        Initialise the quota scheduler

        Args:
            quota_service: the QuotaManagementService instance
        """
        self.quota_service = quota_service
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("QuotaScheduler initialized")
    
    def start(self):
        """Start the scheduler"""
        if self._running:
            logger.warning("QuotaScheduler is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        logger.info("QuotaScheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("QuotaScheduler stopped")
    
    def _run_scheduler(self):
        """Run the main loop of the scheduler"""
        logger.info("QuotaScheduler main loop started")
        
        # Record the date of the last reset
        last_reset_date = datetime.now(timezone.utc).date()
        
        while self._running:
            try:
                current_date = datetime.now(timezone.utc).date()
                
                # Check whether the quotas have to be reset (a new day)
                if current_date > last_reset_date:
                    logger.info(f"New day detected, resetting daily quotas (last reset: {last_reset_date})")
                    self._reset_all_daily_quotas()
                    last_reset_date = current_date
                
                # Check once an hour
                if self._stop_event.wait(timeout=3600):  # 1 hour
                    break
                    
            except Exception as e:
                logger.error(f"Error in quota scheduler loop: {e}", exc_info=True)
                # After an error, wait a while before continuing
                if self._stop_event.wait(timeout=60):  # 1 minute
                    break
        
        logger.info("QuotaScheduler main loop ended")
    
    def _reset_all_daily_quotas(self):
        """Reset the daily quota of all users"""
        try:
            success = self.quota_service.reset_daily_quota(user_id=None)
            if success:
                logger.info("Successfully reset all daily quotas")
            else:
                logger.error("Failed to reset daily quotas")
        except Exception as e:
            logger.error(f"Error resetting daily quotas: {e}", exc_info=True)
    
    def force_reset_now(self):
        """Force a reset of all quotas now (triggered by hand)"""
        logger.info("Forcing immediate quota reset")
        self._reset_all_daily_quotas()
