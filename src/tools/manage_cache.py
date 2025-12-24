"""Cache management utility for two-level semantic cache"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from agents.finance_agent import finance_rag, rag_enabled

def show_cache_stats():
    """Display cache statistics"""
    if not rag_enabled:
        print("❌ RAG not enabled")
        return
    
    stats = finance_rag.get_cache_stats()
    
    print("=" * 70)
    print("⚡ TWO-LEVEL SEMANTIC CACHE STATISTICS")
    print("=" * 70)
    print()
    print("📊 LEVEL 1 - RAG Context Cache:")
    print(f"   Cache size: {stats['rag_cache']['cache_size']} entries")
    print(f"   Total queries: {stats['rag_cache']['total_queries']}")
    print(f"   Cache hits: {stats['rag_cache']['cache_hits']}")
    print(f"   Cache misses: {stats['rag_cache']['cache_misses']}")
    print(f"   Hit rate: {stats['rag_cache']['hit_rate']:.1f}%")
    print()
    print("📊 LEVEL 2 - LLM Response Cache:")
    print(f"   Cache size: {stats['llm_cache']['cache_size']} entries")
    print(f"   Total queries: {stats['llm_cache']['total_queries']}")
    print(f"   Cache hits: {stats['llm_cache']['cache_hits']}")
    print(f"   Cache misses: {stats['llm_cache']['cache_misses']}")
    print(f"   Hit rate: {stats['llm_cache']['hit_rate']:.1f}%")
    print("=" * 70)

def clear_cache(cache_type):
    """Clear the semantic cache"""
    if not rag_enabled:
        print("❌ RAG not enabled")
        return
    
    cache_names = {
        "all": "both caches",
        "rag": "RAG context cache",
        "llm": "LLM response cache"
    }
    
    response = input(f"⚠️  Clear {cache_names[cache_type]}? (y/N): ")
    if response.lower() == 'y':
        finance_rag.clear_cache(cache_type)
        print(f"✅ {cache_names[cache_type].title()} cleared")
    else:
        print("❌ Cancelled")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage two-level semantic cache')
    parser.add_argument('action', choices=['stats', 'clear'], 
                       help='Action to perform')
    parser.add_argument('--cache', choices=['all', 'rag', 'llm'], 
                       default='all',
                       help='Which cache to clear (default: all)')
    
    args = parser.parse_args()
    
    if args.action == 'stats':
        show_cache_stats()
    elif args.action == 'clear':
        clear_cache(args.cache)