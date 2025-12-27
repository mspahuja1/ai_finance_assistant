"""Diagnose import issues"""
import sys
import os

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

print("=" * 70)
print("DIAGNOSING IMPORT ISSUES")
print("=" * 70)

# Check directories
print(f"\nCurrent dir: {current_dir}")
print(f"Src dir: {src_dir}")
print(f"Python path includes src: {src_dir in sys.path}")

# Check if judge directory exists
judge_dir = os.path.join(src_dir, "judge")
print(f"\nJudge directory exists: {os.path.exists(judge_dir)}")
print(f"Judge directory: {judge_dir}")

# List files in judge directory
if os.path.exists(judge_dir):
    print("\nFiles in judge directory:")
    for file in os.listdir(judge_dir):
        print(f"  - {file}")

# Try importing base_judge
print("\n" + "-" * 70)
print("Testing base_judge import...")
try:
    from judge.base_judge import BaseJudge
    print("✅ BaseJudge imported successfully")
    print(f"   BaseJudge class: {BaseJudge}")
except Exception as e:
    print(f"❌ Failed to import BaseJudge: {e}")

# Try importing few_shot_judge
print("\n" + "-" * 70)
print("Testing few_shot_judge import...")
try:
    from judge import few_shot_judge
    print("✅ few_shot_judge module imported")
    print(f"   Module attributes: {dir(few_shot_judge)}")
    
    # Check if FewShotJudge class exists
    if hasattr(few_shot_judge, 'FewShotJudge'):
        print("✅ FewShotJudge class found in module")
        FewShotJudge = few_shot_judge.FewShotJudge
        print(f"   FewShotJudge class: {FewShotJudge}")
    else:
        print("❌ FewShotJudge class NOT found in module")
        
except Exception as e:
    print(f"❌ Failed to import few_shot_judge: {e}")
    import traceback
    traceback.print_exc()

# Try direct import
print("\n" + "-" * 70)
print("Testing direct import...")
try:
    from judge.few_shot_judge import FewShotJudge
    print("✅ FewShotJudge imported successfully")
    print(f"   Class: {FewShotJudge}")
except Exception as e:
    print(f"❌ Failed to import FewShotJudge directly: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)