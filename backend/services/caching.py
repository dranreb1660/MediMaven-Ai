import time
from collections import OrderedDict
from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class InHouseCache:
    """Thread-safe in-memory cache with automatic expiration and size management"""
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self._lock = asyncio.Lock()  # Add thread safety
        
    async def get(self, key: str) -> Optional[Dict]:
        """Thread-safe get with automatic cleanup"""
        async with self._lock:
            if key not in self.cache:
                return None
                
            try:
                value, expiry = self.cache[key]
                if time.time() > expiry:
                    del self.cache[key]
                    return None
                    
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return value
                
            except (ValueError, TypeError, KeyError) as e:
                # Handle corrupted cache entries
                logger.warning(f"Corrupted cache entry for key {key}: {e}")
                self.cache.pop(key, None)
                return None
        
    async def set(self, key: str, value: Any, expire: int = 300):
        """Thread-safe set with size management"""
        if not key or expire <= 0:  # Validate inputs
            return
            
        async with self._lock:
            try:
                # Clean expired entries first
                await self._clean_expired_unsafe()
                
                # Remove oldest if at capacity
                while len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                
                self.cache[key] = (value, time.time() + expire)
                
            except Exception as e:
                logger.error(f"Cache set failed for key {key}: {e}")
                
    async def _clean_expired_unsafe(self):
        """Internal cleanup without locking (assumes already locked)"""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self.cache.items() if now > exp]
        for key in expired_keys:
            self.cache.pop(key, None)
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)