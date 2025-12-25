"""View and analyze evaluation results."""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from judge.evaluation_runner import EvaluationRunner

def main():
    runner = EvaluationRunner(eval_dir="evaluations/finance")
    
    # Get recent evaluations
    recent = runner.get_recent_evaluations(limit=10)
    
    print("=" * 70)
    print("RECENT EVALUATIONS")
    print("=" * 70)
    
    for i, eval_result in enumerate(recent, 1):
        print(f"\n{i}. Evaluation:")
        print(f"   Query: {eval_result['user_query'][:60]}...")
        print(f"   Overall: {eval_result['overall_score']}/5")
        print(f"   Accuracy: {eval_result['accuracy_score']}/5")
        print(f"   Safety: {eval_result['safety_score']}/5")
        print(f"   Timestamp: {eval_result.get('evaluation_timestamp', 'N/A')}")
    
    # Generate and display report
    print("\n")
    report = runner.generate_report()
    print(report)

if __name__ == "__main__":
    main()