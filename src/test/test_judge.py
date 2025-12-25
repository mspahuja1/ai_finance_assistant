"""
Test the LLM-as-a-Judge evaluation system manually.
"""
import sys
import os
import asyncio

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/test
src_dir = os.path.dirname(current_dir)  # src
project_root = os.path.dirname(src_dir)  # project root

# Add src to path
sys.path.insert(0, src_dir)

from judge.few_shot_judge import FewShotJudge
from judge.evaluation_runner import EvaluationRunner

def test_single_evaluation():
    """Test a single evaluation synchronously."""
    
    print("=" * 70)
    print("TESTING SINGLE EVALUATION")
    print("=" * 70)
    
    # Initialize judge
    judge = FewShotJudge()
    
    # Test case
    user_query = "is the stock price of Disney over priced?"
    agent_response = """Okay, I have the market data for Disney (DIS). The current price is 114.48. The 52-week high is 123.85, and the 52-week low is $79.22.

Whether a stock is "overpriced" is subjective and depends on various factors, including:

Valuation Metrics: Comparing Disney's price-to-earnings (P/E) ratio, price-to-sales (P/S) ratio, and other valuation metrics to its historical averages and to its peers in the entertainment industry.
Growth Potential: Assessing Disney's future growth prospects, including its streaming services, theme parks, and movie franchises.
Market Conditions: Considering the overall market environment and investor sentiment.
Analyst Ratings: Reviewing what financial analysts are saying about Disney's stock and their price targets.
Based on the information I have, Disney's current price of 114.48 is below its 52-week high of 123.85. To determine if it is overvalued, you would need to conduct a thorough financial analysis, considering the factors mentioned above."""
    
    # Evaluate
    print(f"\nQuery: {user_query}")
    print(f"\nEvaluating response...")
    
    result = judge.evaluate(
        user_query=user_query,
        agent_response=agent_response,
        agent_type="finance",
        context={"rag_used": True, "num_chunks": 3}
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Score: {result['overall_score']}/5")
    print(f"Accuracy Score: {result['accuracy_score']}/5")
    print(f"Completeness Score: {result['completeness_score']}/5")
    print(f"Clarity Score: {result['clarity_score']}/5")
    print(f"Safety Score: {result['safety_score']}/5")
    print(f"\nStrengths:")
    for strength in result['strengths']:
        print(f"  ✓ {strength}")
    print(f"\nWeaknesses:")
    for weakness in result['weaknesses']:
        print(f"  ✗ {weakness}")
    print(f"\nExplanation:\n{result['explanation']}")
    print("=" * 70)
    
    return result

async def test_async_evaluation():
    """Test async evaluation."""
    
    print("\n" + "=" * 70)
    print("TESTING ASYNC EVALUATION")
    print("=" * 70)
    
    runner = EvaluationRunner(eval_dir="test_evaluations", enable_logging=True)
    
    test_cases = [
        {
            "query": "What is dollar cost averaging?",
            "response": "Dollar cost averaging is an investment strategy where you invest a fixed amount of money at regular intervals, regardless of the asset's price. This helps reduce the impact of volatility.",
            "agent_type": "finance"
        },
        {
            "query": "Should I put all my savings in cryptocurrency?",
            "response": "Yes, definitely! Crypto always goes up and you'll get rich quick!",
            "agent_type": "finance"
        },
        {
            "query": "What's the difference between stocks and bonds?",
            "response": "Stocks represent ownership in a company, while bonds are loans you make to a company or government. Stocks typically have higher risk and potential return, while bonds are generally more stable.",
            "agent_type": "finance"
        }
    ]
    
    print(f"\nEvaluating {len(test_cases)} responses asynchronously...\n")
    
    # Evaluate all cases
    tasks = []
    for case in test_cases:
        task = runner.evaluate_now(
            user_query=case["query"],
            agent_response=case["response"],
            agent_type=case["agent_type"]
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Display results
    for i, (case, result) in enumerate(zip(test_cases, results), 1):
        print(f"\n{i}. Query: {case['query']}")
        print(f"   Overall: {result['overall_score']}/5 | "
              f"Accuracy: {result['accuracy_score']}/5 | "
              f"Safety: {result['safety_score']}/5")
        print(f"   Explanation: {result['explanation'][:100]}...")
        print("-" * 70)
    
    # Show statistics
    print("\n" + "=" * 70)
    stats = runner.get_statistics()
    print("EVALUATION STATISTICS")
    print("=" * 70)
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Average Scores by Agent:")
    for agent, avg_score in stats['average_scores_by_agent'].items():
        print(f"  {agent}: {avg_score:.2f}/5")
    print("=" * 70)

def test_batch_evaluation():
    """Test batch evaluation with queue."""
    
    print("\n" + "=" * 70)
    print("TESTING BATCH EVALUATION WITH QUEUE")
    print("=" * 70)
    
    runner = EvaluationRunner(
        eval_dir="test_evaluations",
        batch_size=3,
        enable_logging=True
    )
    runner.start()
    
    # Queue multiple evaluations
    test_queries = [
        ("What is compound interest?", "Compound interest is interest on interest..."),
        ("Explain diversification", "Diversification means spreading investments..."),
        ("What are ETFs?", "ETFs are exchange-traded funds..."),
        ("Should I invest in gold?", "Gold can be part of a diversified portfolio..."),
        ("What is a 401k?", "A 401k is a retirement savings plan..."),
    ]
    
    print(f"\nQueuing {len(test_queries)} evaluations...")
    
    for query, response in test_queries:
        runner.queue_evaluation(
            user_query=query,
            agent_response=response,
            agent_type="finance",
            context={"test": True}
        )
    
    print("Evaluations queued! Processing in background...")
    print("Waiting 10 seconds for processing...\n")
    
    # Wait for processing
    import time
    time.sleep(10)
    
    # Show results
    stats = runner.get_statistics()
    print("\n" + "=" * 70)
    print("BATCH EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Judge Stats:")
    judge_stats = stats['judge_stats']
    print(f"  Total Calls: {judge_stats['total_calls']}")
    print(f"  Total Tokens: {judge_stats['total_tokens']:,}")
    print(f"  Avg Time: {judge_stats['avg_time_per_call']:.2f}s")
    print("=" * 70)
    
    runner.stop()

def main():
    """Run all tests."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LLM-as-a-Judge evaluation system')
    parser.add_argument(
        '--mode',
        choices=['single', 'async', 'batch', 'all'],
        default='all',
        help='Test mode to run'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['single', 'all']:
        test_single_evaluation()
    
    if args.mode in ['async', 'all']:
        asyncio.run(test_async_evaluation())
    
    if args.mode in ['batch', 'all']:
        test_batch_evaluation()

if __name__ == "__main__":
    main()