"""Semantic Cache Module - Two-level caching for RAG and LLM responses"""
import os
import json
import hashlib
from datetime import datetime
import numpy as np
from typing import Tuple, Dict, Optional
import logging

class SemanticCache:
    """Base semantic cache using cosine similarity"""
    
    def __init__(
        self, 
        embeddings, 
        cache_dir: str, 
        cache_name: str, 
        similarity_threshold: float = 0.95,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize semantic cache
        
        Args:
            embeddings: Embedding model instance
            cache_dir: Directory to store cache files
            cache_name: Name for this cache (e.g., "rag_cache", "llm_cache")
            similarity_threshold: Minimum similarity score (0-1) to consider a cache hit
            logger: Optional logger instance
        """
        self.embeddings = embeddings
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.similarity_threshold = similarity_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.cache_file = os.path.join(cache_dir, f"{cache_name}.json")
        self.cache_vectors_file = os.path.join(cache_dir, f"{cache_name}_vectors.npy")
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.total_queries = 0
        
        # Cache storage
        self.cache_queries = []  # List of queries
        self.cache_embeddings = []  # List of embeddings
        self.cache_data = {}  # Dict of cached data
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load or create cache
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from disk"""
        try:
            # Load cache metadata
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache_data = json.load(f)
                self.logger.info("✅ Loaded %s with %d entries", 
                               self.cache_name, len(self.cache_data))
            else:
                self.cache_data = {}
                self.logger.info("📝 Created new %s", self.cache_name)
            
            # Load cache embeddings
            if os.path.exists(self.cache_vectors_file):
                cache_array = np.load(self.cache_vectors_file, allow_pickle=True)
                self.cache_queries = cache_array.item().get('queries', [])
                self.cache_embeddings = cache_array.item().get('embeddings', [])
                self.logger.info("✅ Loaded %d cached embeddings for %s", 
                               len(self.cache_embeddings), self.cache_name)
            else:
                self.cache_queries = []
                self.cache_embeddings = []
                
        except Exception as e:
            self.logger.error("❌ Error loading %s: %s", self.cache_name, e)
            self.cache_data = {}
            self.cache_queries = []
            self.cache_embeddings = []
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            # Save metadata
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
            
            # Save embeddings
            cache_array = {
                'queries': self.cache_queries,
                'embeddings': self.cache_embeddings
            }
            np.save(self.cache_vectors_file, cache_array)
            
            self.logger.info("💾 Saved %s (%d entries)", self.cache_name, len(self.cache_data))
        except Exception as e:
            self.logger.error("❌ Error saving %s: %s", self.cache_name, e)
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a unique key for the query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, query: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if query exists in cache
        
        Args:
            query: User query string
            
        Returns:
            tuple: (cache_hit: bool, cached_data: dict or None)
        """
        self.total_queries += 1
        
        if len(self.cache_data) == 0:
            self.misses += 1
            self.logger.info("🔍 %s MISS (empty cache)", self.cache_name)
            return False, None
        
        try:
            # Generate embedding for current query
            self.logger.info("🔍 Searching %s...", self.cache_name)
            query_embedding = self.embeddings.embed_query(query)
            
            # Find most similar cached query
            max_similarity = 0.0
            most_similar_idx = -1
            
            for idx, cached_embedding in enumerate(self.cache_embeddings):
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_idx = idx
            
            if most_similar_idx == -1:
                self.misses += 1
                self.logger.info("🔍 %s MISS (no cached queries)", self.cache_name)
                return False, None
            
            most_similar_query = self.cache_queries[most_similar_idx]
            
            self.logger.info("📊 %s - Most similar query: similarity=%.4f", 
                           self.cache_name, max_similarity)
            self.logger.info("   Current query:  '%s'", query[:80])
            self.logger.info("   Cached query:   '%s'", most_similar_query[:80])
            
            if max_similarity >= self.similarity_threshold:
                # Cache hit!
                cache_key = self._generate_cache_key(most_similar_query)
                cached_data = self.cache_data.get(cache_key)
                
                if cached_data:
                    self.hits += 1
                    hit_rate = (self.hits / self.total_queries) * 100
                    self.logger.info("✅ %s HIT! (similarity=%.4f, hit_rate=%.1f%%)", 
                                   self.cache_name, max_similarity, hit_rate)
                    return True, cached_data
            
            self.misses += 1
            self.logger.info("🔍 %s MISS (similarity %.4f < threshold %.4f)", 
                           self.cache_name, max_similarity, self.similarity_threshold)
            return False, None
            
        except Exception as e:
            self.misses += 1
            self.logger.error("❌ %s lookup error: %s", self.cache_name, e)
            return False, None
    
    def set(self, query: str, data: Dict):
        """
        Store query result in cache
        
        Args:
            query: User query string
            data: Data dictionary to cache
        """
        try:
            cache_key = self._generate_cache_key(query)
            
            # Check if already cached (avoid duplicates)
            if cache_key in self.cache_data:
                self.logger.info("ℹ️ Query already in %s, skipping", self.cache_name)
                return
            
            # Store metadata
            data["timestamp"] = datetime.now().isoformat()
            data["access_count"] = 1
            self.cache_data[cache_key] = data
            
            # Generate and store embedding
            query_embedding = self.embeddings.embed_query(query)
            self.cache_queries.append(query)
            self.cache_embeddings.append(query_embedding)
            
            # Save cache
            self._save_cache()
            
            self.logger.info("💾 Cached to %s (total entries: %d)", 
                           self.cache_name, len(self.cache_data))
            
        except Exception as e:
            self.logger.error("❌ Error caching to %s: %s", self.cache_name, e)
    
    def update_access(self, cached_data: Dict):
        """Update access count for cached entry"""
        try:
            query = cached_data.get("query")
            if not query:
                return
            
            cache_key = self._generate_cache_key(query)
            if cache_key in self.cache_data:
                self.cache_data[cache_key]["access_count"] += 1
                self.cache_data[cache_key]["last_accessed"] = datetime.now().isoformat()
                # Save updated metadata
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache_data, f, indent=2)
        except Exception as e:
            self.logger.error("❌ Error updating access count in %s: %s", 
                            self.cache_name, e)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = (self.hits / self.total_queries * 100) if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache_data)
        }
    
    def clear(self):
        """Clear all cache"""
        self.cache_data = {}
        self.cache_queries = []
        self.cache_embeddings = []
        self.hits = 0
        self.misses = 0
        self.total_queries = 0
        
        # Remove cache files
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        if os.path.exists(self.cache_vectors_file):
            os.remove(self.cache_vectors_file)
        
        self.logger.info("🗑️ %s cleared", self.cache_name)


class TwoLevelCache:
    """Two-level semantic cache for RAG and LLM responses"""
    
    def __init__(
        self, 
        embeddings, 
        cache_dir: str,
        rag_similarity_threshold: float = 0.95,
        llm_similarity_threshold: float = 0.92,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize two-level cache system
        
        Args:
            embeddings: Embedding model instance
            cache_dir: Directory to store cache files
            rag_similarity_threshold: Similarity threshold for RAG cache
            llm_similarity_threshold: Similarity threshold for LLM cache
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Level 1: RAG Context Cache (saves embedding + retrieval)
        self.rag_cache = SemanticCache(
            embeddings=embeddings,
            cache_dir=cache_dir,
            cache_name="rag_cache",
            similarity_threshold=rag_similarity_threshold,
            logger=self.logger
        )
        
        # Level 2: LLM Response Cache (saves LLM generation)
        self.llm_cache = SemanticCache(
            embeddings=embeddings,
            cache_dir=cache_dir,
            cache_name="llm_cache",
            similarity_threshold=llm_similarity_threshold,
            logger=self.logger
        )
        
        self.logger.info("✅ Two-level cache initialized")
        self.logger.info("   Level 1: RAG Context Cache (threshold=%.2f)", 
                        rag_similarity_threshold)
        self.logger.info("   Level 2: LLM Response Cache (threshold=%.2f)", 
                        llm_similarity_threshold)
    
    def get_rag_context(self, query: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check Level 1 cache for RAG context
        
        Args:
            query: User query string
            
        Returns:
            tuple: (cache_hit: bool, cached_data: dict or None)
        """
        return self.rag_cache.get(query)
    
    def set_rag_context(self, query: str, context: str, token_stats: Dict):
        """
        Store RAG context in Level 1 cache
        
        Args:
            query: User query string
            context: Retrieved context
            token_stats: Token usage statistics
        """
        self.rag_cache.set(query, {
            "query": query,
            "context": context,
            "token_stats": token_stats
        })
    
    def get_llm_response(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check Level 2 cache for LLM response
        
        Args:
            query: User query string
            
        Returns:
            tuple: (cache_hit: bool, response: str or None)
        """
        cache_hit, cached_data = self.llm_cache.get(query)
        
        if cache_hit:
            self.llm_cache.update_access(cached_data)
            return True, cached_data.get("response")
        
        return False, None
    
    def set_llm_response(self, query: str, response: str, token_stats: Dict):
        """
        Store LLM response in Level 2 cache
        
        Args:
            query: User query string
            response: LLM generated response
            token_stats: Token usage statistics
        """
        self.llm_cache.set(query, {
            "query": query,
            "response": response,
            "token_stats": token_stats
        })
    
    def get_stats(self) -> Dict:
        """Get combined cache statistics"""
        return {
            "rag_cache": self.rag_cache.get_stats(),
            "llm_cache": self.llm_cache.get_stats()
        }
    
    def clear(self, cache_type: str = "all"):
        """
        Clear cache(s)
        
        Args:
            cache_type: "all", "rag", or "llm"
        """
        if cache_type in ["all", "rag"]:
            self.rag_cache.clear()
            self.logger.info("🗑️ RAG cache cleared")
        
        if cache_type in ["all", "llm"]:
            self.llm_cache.clear()
            self.logger.info("🗑️ LLM cache cleared")