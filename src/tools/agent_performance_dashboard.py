"""
Agent Performance Dashboard - Compare all agents' evaluation metrics.

Usage:
    python src/tools/agent_performance_dashboard.py
    python src/tools/agent_performance_dashboard.py --detailed
    python src/tools/agent_performance_dashboard.py --agent finance
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from judge.evaluation_runner import EvaluationRunner


def get_agent_stats(agent_name: str, eval_base_dir: str = "evaluations") -> Dict[str, Any]:
    """Get statistics for a specific agent."""
    eval_dir = os.path.join(src_dir, eval_base_dir, agent_name)
    
    if not os.path.exists(eval_dir):
        return None
    
    runner = EvaluationRunner(eval_dir=eval_dir)
    stats = runner.get_statistics()
    
    # Get detailed score breakdown
    recent_evals = runner.get_recent_evaluations(limit=100)
    
    if recent_evals:
        # Calculate average scores across all dimensions
        scores = {
            'overall': [],
            'accuracy': [],
            'completeness': [],
            'clarity': [],
            'safety': [],
            'composite': []
        }
        
        for eval in recent_evals:
            scores['overall'].append(eval.get('overall_score', 0))
            scores['accuracy'].append(eval.get('accuracy_score', 0))
            scores['completeness'].append(eval.get('completeness_score', 0))
            scores['clarity'].append(eval.get('clarity_score', 0))
            scores['safety'].append(eval.get('safety_score', 0))
            scores['composite'].append(eval.get('composite_score', 0))
        
        # Calculate averages
        avg_scores = {
            key: round(sum(values) / len(values), 2) if values else 0
            for key, values in scores.items()
        }
        
        # Calculate score distribution
        score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in scores['composite']:
            if 1 <= score <= 5:
                score_distribution[score] += 1
        
        stats['avg_scores'] = avg_scores
        stats['score_distribution'] = score_distribution
        stats['recent_evaluations'] = recent_evals[:10]  # Last 10
    
    return stats


def print_summary_dashboard(agents: List[str]):
    """Print summary dashboard for all agents."""
    print("=" * 80)
    print("AGENT PERFORMANCE DASHBOARD - SUMMARY")
    print("=" * 80)
    
    all_stats = {}
    for agent in agents:
        stats = get_agent_stats(agent)
        if stats and stats['total_evaluations'] > 0:
            all_stats[agent] = stats
    
    if not all_stats:
        print("\n❌ No evaluation data found for any agent.")
        print("Make sure evaluations are in: src/evaluations/")
        return
    
    # Summary table
    print(f"\n{'Agent':<12} {'Evals':<8} {'Avg Score':<12} {'Quality':<10} {'Status':<10}")
    print("-" * 80)
    
    for agent, stats in sorted(all_stats.items()):
        avg_score = stats['average_score']
        
        # Determine quality rating
        if avg_score >= 4.5:
            quality = "Excellent ⭐"
        elif avg_score >= 4.0:
            quality = "Good ✓"
        elif avg_score >= 3.5:
            quality = "Fair ~"
        else:
            quality = "Needs Work ⚠️"
        
        # Worker status
        status = "Running ✓" if stats.get('is_running') else "Stopped ✗"
        
        print(f"{agent.capitalize():<12} {stats['total_evaluations']:<8} "
              f"{avg_score:.2f}/5.0    {quality:<10} {status:<10}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    total_evals = sum(s['total_evaluations'] for s in all_stats.values())
    avg_all_agents = sum(s['average_score'] for s in all_stats.values()) / len(all_stats)
    
    print(f"Total Evaluations: {total_evals}")
    print(f"Average Score (All Agents): {avg_all_agents:.2f}/5.0")
    print(f"Agents Active: {len(all_stats)}/{len(agents)}")
    
    # Find best and worst performing
    best_agent = max(all_stats.items(), key=lambda x: x[1]['average_score'])
    worst_agent = min(all_stats.items(), key=lambda x: x[1]['average_score'])
    
    print(f"\nBest Performing: {best_agent[0].capitalize()} ({best_agent[1]['average_score']:.2f}/5.0)")
    print(f"Needs Improvement: {worst_agent[0].capitalize()} ({worst_agent[1]['average_score']:.2f}/5.0)")


def print_detailed_dashboard(agents: List[str]):
    """Print detailed dashboard with score breakdowns."""
    print("=" * 80)
    print("AGENT PERFORMANCE DASHBOARD - DETAILED")
    print("=" * 80)
    
    for agent in agents:
        stats = get_agent_stats(agent)
        
        if not stats or stats['total_evaluations'] == 0:
            continue
        
        print(f"\n{'=' * 80}")
        print(f"{agent.upper()} AGENT")
        print(f"{'=' * 80}")
        
        # Basic stats
        print(f"\nTotal Evaluations: {stats['total_evaluations']}")
        print(f"Queue Size: {stats['queue_size']}")
        print(f"Worker Status: {'Running ✓' if stats['is_running'] else 'Stopped ✗'}")
        
        # Score breakdown
        if 'avg_scores' in stats:
            avg_scores = stats['avg_scores']
            
            print(f"\n{'Score Breakdown:':<20}")
            print(f"  Composite:     {avg_scores['composite']:.2f}/5.0")
            print(f"  Overall:       {avg_scores['overall']:.2f}/5.0")
            print(f"  Accuracy:      {avg_scores['accuracy']:.2f}/5.0")
            print(f"  Completeness:  {avg_scores['completeness']:.2f}/5.0")
            print(f"  Clarity:       {avg_scores['clarity']:.2f}/5.0")
            print(f"  Safety:        {avg_scores['safety']:.2f}/5.0")
        
        # Score distribution
        if 'score_distribution' in stats:
            dist = stats['score_distribution']
            total = sum(dist.values())
            
            print(f"\n{'Score Distribution:'}")
            for score in [5, 4, 3, 2, 1]:
                count = dist.get(score, 0)
                percentage = (count / total * 100) if total > 0 else 0
                bar = "█" * int(percentage / 2)
                print(f"  {score} stars: {bar:<50} {count:>3} ({percentage:>5.1f}%)")
        
        # Recent evaluations
        if 'recent_evaluations' in stats and stats['recent_evaluations']:
            print(f"\n{'Recent Evaluations (Last 5):'}")
            for i, eval in enumerate(stats['recent_evaluations'][:5], 1):
                query = eval.get('user_query', 'N/A')[:60]
                score = eval.get('composite_score', eval.get('overall_score', 'N/A'))
                print(f"  {i}. [{score}/5] {query}...")


def print_single_agent_report(agent_name: str):
    """Print detailed report for a single agent."""
    stats = get_agent_stats(agent_name)
    
    if not stats:
        print(f"❌ No evaluation data found for agent: {agent_name}")
        print(f"Check: src/evaluations/{agent_name}/")
        return
    
    print("=" * 80)
    print(f"{agent_name.upper()} AGENT - DETAILED REPORT")
    print("=" * 80)
    
    if stats['total_evaluations'] == 0:
        print("\n❌ No evaluations yet for this agent.")
        return
    
    # Header info
    print(f"\nTotal Evaluations: {stats['total_evaluations']}")
    print(f"Average Score: {stats['average_score']:.2f}/5.0")
    print(f"Queue Size: {stats['queue_size']}")
    print(f"Worker Status: {'Running ✓' if stats['is_running'] else 'Stopped ✗'}")
    
    # Detailed scores
    if 'avg_scores' in stats:
        avg_scores = stats['avg_scores']
        
        print(f"\n{'SCORE BREAKDOWN':-^80}")
        scores_list = [
            ("Composite Score", avg_scores['composite']),
            ("Overall Quality", avg_scores['overall']),
            ("Accuracy", avg_scores['accuracy']),
            ("Completeness", avg_scores['completeness']),
            ("Clarity", avg_scores['clarity']),
            ("Safety", avg_scores['safety'])
        ]
        
        for label, score in scores_list:
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"{label:<20} {bar} {score:.2f}/5.0")
    
    # Recent evaluations with details
    if 'recent_evaluations' in stats:
        print(f"\n{'RECENT EVALUATIONS':-^80}")
        
        for i, eval in enumerate(stats['recent_evaluations'][:10], 1):
            print(f"\n{i}. Evaluation:")
            print(f"   Query: {eval.get('user_query', 'N/A')[:70]}...")
            print(f"   Score: {eval.get('composite_score', eval.get('overall_score', 'N/A'))}/5")
            
            strengths = eval.get('strengths', [])
            if strengths:
                print(f"   Strengths: {strengths[0][:60]}...")
            
            weaknesses = eval.get('weaknesses', [])
            if weaknesses:
                print(f"   Weaknesses: {weaknesses[0][:60]}...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent Performance Dashboard")
    parser.add_argument(
        '--mode',
        choices=['summary', 'detailed', 'agent'],
        default='summary',
        help='Dashboard mode'
    )
    parser.add_argument(
        '--agent',
        choices=['finance', 'market', 'goal', 'news', 'portfolio', 'tax'],
        help='Specific agent to view (use with --mode agent)'
    )
    
    args = parser.parse_args()
    
    agents = ['finance', 'market', 'goal', 'news', 'portfolio', 'tax']
    
    if args.mode == 'summary':
        print_summary_dashboard(agents)
    elif args.mode == 'detailed':
        print_detailed_dashboard(agents)
    elif args.mode == 'agent':
        if not args.agent:
            print("❌ Error: --agent required when using --mode agent")
            print("Example: python src/tools/agent_performance_dashboard.py --mode agent --agent finance")
            return
        print_single_agent_report(args.agent)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()