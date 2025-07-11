
import time
from collections import OrderedDict
from typing import Dict, Any, Optional


class InHouseCache:
    """In-memory cache with automatic expiration and size management"""
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[Dict]:
        if key not in self.cache:
            return None
        value, expiry = self.cache[key]
        if time.time() > expiry:
            del self.cache[key]
            return None
        # Move to end to mark recently used
        self.cache.move_to_end(key)
        return value
        
    def set(self, key: str, value: Any, expire: int = 300):
        if len(self.cache) >= self.max_size:
            self.clean_expired()
            if len(self.cache) >= self.max_size:
                # Remove oldest items if still full
                self.cache.popitem(last=False)
        self.cache[key] = (value, time.time() + expire)
        
    def clean_expired(self):
        now = time.time()
        expired_keys = [k for k, (_, expiry) in self.cache.items() if now > expiry]
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
                
