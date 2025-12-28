"""
Quick Feedback Dashboard

View recent quality scores and performance metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feedback.async_feedback import async_feedback
import json
from datetime import datetime, timedelta


def show_recent_quality_scores(agent_type="finance", limit=10):
    """Show recent quality scores."""
    quality_dir = Path("feedback/quality_scores")
    
    if not quality_dir.exists():
        print(f"⚠️ Quality scores directory not found: {quality_dir}")
        return
    
    files = sorted(quality_dir.glob("quality_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("=" * 80)
    print(f"RECENT QUALITY SCORES - {agent_type.upper()} Agent")
    print("=" * 80)
    
    count = 0
    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            if data.get("agent_type") == agent_type:
                query = data.get('query', 'N/A')
                print(f"\nQuery: {query[:60]}...")
                print(f"Score: {data.get('quality_score', 0)}/5")
                print(f"Retries: {data.get('retry_count', 0)}")
                print(f"Time: {data.get('timestamp', 'N/A')}")
                
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    if count == 0:
        print(f"\nNo quality scores found for {agent_type} agent.")
        print("Ask some questions in the UI to generate quality scores!")


def show_performance_metrics():
    """Show current performance metrics."""
    metrics = async_feedback.get_performance_metrics()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    for agent, stats in sorted(metrics.items()):
        success_rate = stats.get("success_rate", 0)
        avg_latency = stats.get("avg_latency", 0)
        avg_score = stats.get("avg_score", 0)
        
        status = "✅" if success_rate >= 0.85 else "⚠️"
        
        print(f"\n{agent.upper()} Agent: {status}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Latency: {avg_latency:.2f}s")
        print(f"  Avg Quality Score: {avg_score:.2f}/5")


if __name__ == "__main__":
    show_recent_quality_scores("finance", limit=10)
    show_performance_metrics()
