"""Caching Layer for Market Data"""
import time
from typing import Optional, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import create_logger

mcp_logger = create_logger("cache", "cache.log")

class MarketDataCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        mcp_logger.info(f"🗄️ Initialized cache with TTL={ttl_seconds}s")
    
    def _generate_key(self, tool_name: str, **kwargs) -> str:
        args_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{tool_name}:{args_str}"
    
    def get(self, tool_name: str, **kwargs) -> Optional[str]:
        key = self._generate_key(tool_name, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            age = time.time() - entry['timestamp']
            
            if age < self.ttl_seconds:
                mcp_logger.info(f"✅ CACHE HIT for {key}")
                return entry['data']
            else:
                del self.cache[key]
        
        mcp_logger.info(f"❌ CACHE MISS for {key}")
        return None
    
    def set(self, tool_name: str, data: str, **kwargs):
        key = self._generate_key(tool_name, **kwargs)
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        mcp_logger.info(f"💾 CACHED {key}")
    
    def clear(self):
        count = len(self.cache)
        self.cache.clear()
        mcp_logger.info(f"🗑️ CLEARED cache ({count} entries)")
    
    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        valid_entries = sum(
            1 for entry in self.cache.values()
            if (now - entry['timestamp']) < self.ttl_seconds
        )
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'ttl_seconds': self.ttl_seconds
        }

market_cache = MarketDataCache(ttl_seconds=300)