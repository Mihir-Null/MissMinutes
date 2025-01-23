from collections import deque
import time
import asyncio
from typing import Optional

class AdaptiveRateLimiter:
    """Adaptive rate limiter with sliding window and dynamic delay adjustment"""
    def __init__(self):
        # Core settings
        self.max_requests_per_minute = 40
        self.window_size = 60  # seconds
        
        # Sliding window tracking
        self.request_timestamps = deque()
        
        # Adaptive delay settings
        self.base_delay = 1.5    # Start conservative
        self.min_delay = 0.5     # Fastest we'll go
        self.max_delay = 5.0     # Slowest we'll go
        self.current_delay = self.base_delay
        
        # Success/failure tracking
        self.success_streak = 0
        self.failure_streak = 0
        
        # Burst protection
        self.last_request_time = 0
        self.min_request_gap = 0.1  # Minimum 100ms between requests

    def _clean_old_requests(self):
        """Remove requests outside the current window"""
        now = time.time()
        while self.request_timestamps and (now - self.request_timestamps[0]) > self.window_size:
            self.request_timestamps.popleft()

    def _count_recent_requests(self):
        """Count requests in current window"""
        self._clean_old_requests()
        return len(self.request_timestamps)

    def _adjust_delay(self, success: bool):
        """Dynamically adjust delay based on success/failure patterns"""
        if success:
            self.success_streak += 1
            self.failure_streak = 0
            if self.success_streak >= 5:
                # After 5 successes, cautiously reduce delay
                self.current_delay = max(
                    self.min_delay,
                    self.current_delay * 0.9
                )
                self.success_streak = 0
        else:
            self.failure_streak += 1
            self.success_streak = 0
            # Exponential backoff on failures
            self.current_delay = min(
                self.max_delay,
                self.current_delay * (1.5 ** self.failure_streak)
            )

    async def acquire(self) -> bool:
        """Get permission to make a request"""
        try:
            now = time.time()
            
            # Enforce minimum gap between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_request_gap:
                await asyncio.sleep(self.min_request_gap - time_since_last)
            
            # Check current window
            recent_requests = self._count_recent_requests()
            
            if recent_requests >= self.max_requests_per_minute:
                # We're at limit, wait for oldest request to expire
                wait_time = self.window_size - (now - self.request_timestamps[0])
                await asyncio.sleep(wait_time)
            
            # Apply adaptive delay
            await asyncio.sleep(self.current_delay)
            
            # Record this request
            self.request_timestamps.append(time.time())
            self.last_request_time = time.time()
            
            self._adjust_delay(True)
            return True
            
        except Exception as e:
            self._adjust_delay(False)
            return False

    async def on_error(self, error: Exception):
        """Handle rate limit errors"""
        if "exceed_query_limit" in str(error):
            self._adjust_delay(False)
            # Force a longer cooldown on actual rate limit errors
            await asyncio.sleep(min(self.current_delay * 2, 10)) 