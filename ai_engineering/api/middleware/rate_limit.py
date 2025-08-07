#!/usr/bin/env python3
"""
Rate Limiting Middleware for Titans Finance API

This module provides rate limiting functionality to prevent API abuse
and ensure fair usage across all clients. It supports both Redis-based
distributed rate limiting and in-memory fallback for development.
"""

import time
import json
import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import logging

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis

logger = logging.getLogger(__name__)

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with support for different strategies:
    - Fixed window
    - Sliding window
    - Token bucket
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        redis_client: Optional[redis.Redis] = None,
        strategy: str = "sliding_window",
        key_func: Optional[callable] = None,
        exempt_paths: Optional[list] = None,
        rate_limit_headers: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.redis_client = redis_client
        self.strategy = strategy
        self.key_func = key_func or self._default_key_func
        self.exempt_paths = exempt_paths or ["/health", "/metrics", "/docs", "/redoc"]
        self.rate_limit_headers = rate_limit_headers

        # In-memory storage fallback
        self.memory_store: Dict[str, Dict] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting"""

        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Generate rate limit key
        rate_limit_key = self.key_func(request)

        try:
            # Check rate limits
            await self._check_rate_limits(rate_limit_key, request)

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            if self.rate_limit_headers:
                await self._add_rate_limit_headers(response, rate_limit_key)

            return response

        except RateLimitExceeded as e:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for key: {rate_limit_key}")

            response = Response(
                content=json.dumps({
                    "error": True,
                    "message": "Rate limit exceeded",
                    "details": str(e),
                    "retry_after": e.retry_after if hasattr(e, 'retry_after') else 60
                }),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Content-Type": "application/json"}
            )

            if self.rate_limit_headers:
                await self._add_rate_limit_headers(response, rate_limit_key, exceeded=True)

            return response

    def _default_key_func(self, request: Request) -> str:
        """Default key function based on client IP and API key"""
        client_ip = self._get_client_ip(request)

        # Try to get API key from Authorization header
        auth_header = request.headers.get("Authorization", "")
        api_key = ""
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:][:10]  # First 10 chars for identification

        # Create composite key
        key_data = f"{client_ip}:{api_key}:{request.method}:{request.url.path}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    async def _check_rate_limits(self, key: str, request: Request):
        """Check all rate limit windows"""
        current_time = time.time()

        # Define time windows in seconds
        windows = {
            "minute": (60, self.requests_per_minute),
            "hour": (3600, self.requests_per_hour),
            "day": (86400, self.requests_per_day)
        }

        for window_name, (window_size, limit) in windows.items():
            if self.strategy == "sliding_window":
                await self._check_sliding_window(key, window_name, window_size, limit, current_time)
            elif self.strategy == "fixed_window":
                await self._check_fixed_window(key, window_name, window_size, limit, current_time)
            elif self.strategy == "token_bucket":
                await self._check_token_bucket(key, window_name, window_size, limit, current_time)

    async def _check_sliding_window(self, key: str, window: str, window_size: int, limit: int, current_time: float):
        """Sliding window rate limiting"""
        window_key = f"rate_limit:sliding:{key}:{window}"

        if self.redis_client:
            # Redis-based sliding window
            pipe = self.redis_client.pipeline()

            # Remove expired entries
            pipe.zremrangebyscore(window_key, 0, current_time - window_size)

            # Count current requests
            pipe.zcard(window_key)

            # Add current request
            pipe.zadd(window_key, {str(current_time): current_time})

            # Set expiration
            pipe.expire(window_key, window_size + 10)

            results = pipe.execute()
            request_count = results[1] + 1  # +1 for current request

        else:
            # In-memory sliding window
            await self._cleanup_memory_store()

            if window_key not in self.memory_store:
                self.memory_store[window_key] = {"requests": [], "limit": limit}

            # Remove expired requests
            store = self.memory_store[window_key]
            store["requests"] = [
                req_time for req_time in store["requests"]
                if req_time > current_time - window_size
            ]

            # Add current request
            store["requests"].append(current_time)
            request_count = len(store["requests"])

        if request_count > limit:
            retry_after = window_size
            raise RateLimitExceeded(f"Rate limit exceeded for {window} window", retry_after=retry_after)

    async def _check_fixed_window(self, key: str, window: str, window_size: int, limit: int, current_time: float):
        """Fixed window rate limiting"""
        # Calculate window start
        window_start = int(current_time // window_size) * window_size
        window_key = f"rate_limit:fixed:{key}:{window}:{window_start}"

        if self.redis_client:
            # Redis-based fixed window
            current_count = self.redis_client.incr(window_key)

            if current_count == 1:
                # First request in window, set expiration
                self.redis_client.expire(window_key, window_size)

        else:
            # In-memory fixed window
            await self._cleanup_memory_store()

            if window_key not in self.memory_store:
                self.memory_store[window_key] = {"count": 0, "expires": window_start + window_size}

            # Check if window has expired
            if current_time > self.memory_store[window_key]["expires"]:
                self.memory_store[window_key] = {"count": 0, "expires": window_start + window_size}

            self.memory_store[window_key]["count"] += 1
            current_count = self.memory_store[window_key]["count"]

        if current_count > limit:
            retry_after = window_start + window_size - current_time
            raise RateLimitExceeded(f"Rate limit exceeded for {window} window", retry_after=retry_after)

    async def _check_token_bucket(self, key: str, window: str, window_size: int, limit: int, current_time: float):
        """Token bucket rate limiting"""
        bucket_key = f"rate_limit:bucket:{key}:{window}"
        refill_rate = limit / window_size  # tokens per second

        if self.redis_client:
            # Redis-based token bucket
            bucket_data = self.redis_client.hgetall(bucket_key)

            if bucket_data:
                tokens = float(bucket_data.get(b"tokens", limit))
                last_refill = float(bucket_data.get(b"last_refill", current_time))
            else:
                tokens = limit
                last_refill = current_time

            # Calculate tokens to add
            time_passed = current_time - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(limit, tokens + tokens_to_add)

            if tokens >= 1:
                # Consume token
                tokens -= 1

                # Update bucket
                self.redis_client.hset(bucket_key, mapping={
                    "tokens": tokens,
                    "last_refill": current_time
                })
                self.redis_client.expire(bucket_key, window_size * 2)
            else:
                retry_after = (1 - tokens) / refill_rate
                raise RateLimitExceeded(f"Token bucket empty for {window}", retry_after=retry_after)

        else:
            # In-memory token bucket
            await self._cleanup_memory_store()

            if bucket_key not in self.memory_store:
                self.memory_store[bucket_key] = {
                    "tokens": limit,
                    "last_refill": current_time,
                    "expires": current_time + window_size * 2
                }

            bucket = self.memory_store[bucket_key]

            # Check expiration
            if current_time > bucket["expires"]:
                bucket = self.memory_store[bucket_key] = {
                    "tokens": limit,
                    "last_refill": current_time,
                    "expires": current_time + window_size * 2
                }

            # Refill tokens
            time_passed = current_time - bucket["last_refill"]
            tokens_to_add = time_passed * refill_rate
            bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = current_time

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
            else:
                retry_after = (1 - bucket["tokens"]) / refill_rate
                raise RateLimitExceeded(f"Token bucket empty for {window}", retry_after=retry_after)

    async def _add_rate_limit_headers(self, response: Response, key: str, exceeded: bool = False):
        """Add rate limit headers to response"""
        try:
            # Get current usage for headers
            minute_usage = await self._get_current_usage(key, "minute", 60)
            hour_usage = await self._get_current_usage(key, "hour", 3600)

            # Calculate remaining requests
            minute_remaining = max(0, self.requests_per_minute - minute_usage)
            hour_remaining = max(0, self.requests_per_hour - hour_usage)

            # Add headers
            response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
            response.headers["X-RateLimit-Remaining-Minute"] = str(minute_remaining)
            response.headers["X-RateLimit-Remaining-Hour"] = str(hour_remaining)

            # Calculate reset time
            current_time = time.time()
            minute_reset = int((current_time // 60 + 1) * 60)
            hour_reset = int((current_time // 3600 + 1) * 3600)

            response.headers["X-RateLimit-Reset-Minute"] = str(minute_reset)
            response.headers["X-RateLimit-Reset-Hour"] = str(hour_reset)

            if exceeded:
                response.headers["Retry-After"] = "60"

        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}")

    async def _get_current_usage(self, key: str, window: str, window_size: int) -> int:
        """Get current usage for a window"""
        current_time = time.time()

        if self.strategy == "sliding_window":
            window_key = f"rate_limit:sliding:{key}:{window}"

            if self.redis_client:
                # Remove expired and count
                self.redis_client.zremrangebyscore(window_key, 0, current_time - window_size)
                return self.redis_client.zcard(window_key)
            else:
                if window_key in self.memory_store:
                    store = self.memory_store[window_key]
                    valid_requests = [
                        req_time for req_time in store["requests"]
                        if req_time > current_time - window_size
                    ]
                    return len(valid_requests)

        elif self.strategy == "fixed_window":
            window_start = int(current_time // window_size) * window_size
            window_key = f"rate_limit:fixed:{key}:{window}:{window_start}"

            if self.redis_client:
                count = self.redis_client.get(window_key)
                return int(count) if count else 0
            else:
                if window_key in self.memory_store:
                    if current_time <= self.memory_store[window_key]["expires"]:
                        return self.memory_store[window_key]["count"]

        return 0

    async def _cleanup_memory_store(self):
        """Clean up expired entries from memory store"""
        current_time = time.time()

        # Only cleanup periodically
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        self.last_cleanup = current_time
        expired_keys = []

        for key, data in self.memory_store.items():
            if "expires" in data and current_time > data["expires"]:
                expired_keys.append(key)
            elif "requests" in data:
                # For sliding window, clean old requests
                data["requests"] = [
                    req_time for req_time in data["requests"]
                    if req_time > current_time - 86400  # Keep last day
                ]
                if not data["requests"]:
                    expired_keys.append(key)

        for key in expired_keys:
            del self.memory_store[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit entries")

class CustomRateLimiter:
    """Custom rate limiter for specific endpoints"""

    def __init__(self, requests: int, window: int, key_func: Optional[callable] = None):
        self.requests = requests
        self.window = window
        self.key_func = key_func or (lambda request: f"custom:{request.client.host}")
        self.store = {}

    async def __call__(self, request: Request):
        """Check rate limit for request"""
        key = self.key_func(request)
        current_time = time.time()

        # Clean up old entries
        if key in self.store:
            self.store[key] = [
                req_time for req_time in self.store[key]
                if req_time > current_time - self.window
            ]
        else:
            self.store[key] = []

        # Check limit
        if len(self.store[key]) >= self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests} requests per {self.window} seconds"
            )

        # Add current request
        self.store[key].append(current_time)

# Decorator for endpoint-specific rate limiting
def rate_limit(requests: int, window: int, key_func: Optional[callable] = None):
    """Decorator for endpoint-specific rate limiting"""
    limiter = CustomRateLimiter(requests, window, key_func)

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            await limiter(request)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Utility functions
def create_rate_limit_key(prefix: str, *parts) -> str:
    """Create a standardized rate limit key"""
    key_parts = [prefix] + [str(part) for part in parts]
    return ":".join(key_parts)

def parse_rate_limit_config(config_str: str) -> Tuple[int, int]:
    """Parse rate limit configuration string like '100/hour' or '10/minute'"""
    parts = config_str.lower().split('/')
    if len(parts) != 2:
        raise ValueError("Rate limit config must be in format 'requests/window'")

    requests = int(parts[0])
    window_str = parts[1]

    window_multipliers = {
        'second': 1,
        'minute': 60,
        'hour': 3600,
        'day': 86400
    }

    window = window_multipliers.get(window_str)
    if window is None:
        raise ValueError(f"Unknown time window: {window_str}")

    return requests, window

# Example usage
if __name__ == "__main__":
    # Test configuration parsing
    print(parse_rate_limit_config("100/hour"))  # (100, 3600)
    print(parse_rate_limit_config("10/minute"))  # (10, 60)

    # Test key creation
    print(create_rate_limit_key("api", "user123", "endpoint"))  # api:user123:endpoint
