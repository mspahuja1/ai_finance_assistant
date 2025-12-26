"""View and analyze evaluation results."""
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from judge.evaluation_runner import EvaluationRunner


def main():
    """View recent evaluations."""
    # Use src/evaluations/finance since that's where they're actually saved
    eval_dir = os.path.join(os.path.dirname(__file__), "..", "evaluations", "finance")
    runner = EvaluationRunner(eval_dir=eval_dir)
    
    # Get recent evaluations
    recent = runner.get_recent_evaluations(limit=10)
    
    print("=" * 70)
    print("RECENT EVALUATIONS")
    print("=" * 70)
    
    if not recent:
        print("\nNo evaluations found.")
        print(f"Looking in: {runner.eval_dir.absolute()}")
        return
    
    print(f"\nFound {len(recent)} evaluations\n")
    
    for i, eval_result in enumerate(recent, 1):
        print(f"\n{'='*70}")
        print(f"Evaluation #{i}")
        print(f"{'='*70}")
        print(f"Query: {eval_result.get('user_query', 'N/A')[:100]}...")
        print(f"\nScores:")
        print(f"  Overall:      {eval_result.get('overall_score', 'N/A')}/5")
        print(f"  Composite:    {eval_result.get('composite_score', 'N/A')}/5")
        print(f"  Accuracy:     {eval_result.get('accuracy_score', 'N/A')}/5")
        print(f"  Completeness: {eval_result.get('completeness_score', 'N/A')}/5")
        print(f"  Clarity:      {eval_result.get('clarity_score', 'N/A')}/5")
        print(f"  Safety:       {eval_result.get('safety_score', 'N/A')}/5")
        
        # Show strengths
        strengths = eval_result.get('strengths', [])
        if strengths:
            print(f"\nStrengths:")
            for s in strengths[:3]:
                print(f"  ✓ {s}")
        
        # Show weaknesses
        weaknesses = eval_result.get('weaknesses', [])
        if weaknesses:
            print(f"\nWeaknesses:")
            for w in weaknesses[:3]:
                print(f"  ✗ {w}")
        
        # Show explanation
        explanation = eval_result.get('explanation', '')
        if explanation:
            print(f"\nExplanation: {explanation[:200]}...")
        
        timestamp = eval_result.get('evaluation_timestamp', 'N/A')
        if timestamp != 'N/A':
            print(f"\nTimestamp: {timestamp}")
    
    # Show statistics
    stats = runner.get_statistics()
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total Evaluations: {stats.get('total_evaluations', 0)}")
    print(f"Average Score: {stats.get('average_score', 0):.2f}/5")
    print(f"Queue Size: {stats.get('queue_size', 0)}")
    print(f"Worker Running: {stats.get('is_running', False)}")
    print(f"Worker Alive: {stats.get('worker_alive', False)}")
    print("=" * 70)


if __name__ == "__main__":
    main()