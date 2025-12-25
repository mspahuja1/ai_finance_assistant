#!/usr/bin/env python
"""Test imports directly"""
import sys
import os

print("Starting import test...")

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

print(f"Python path: {sys.path[:3]}")

# Test 1: Import the module
print("\n1. Importing few_shot_judge module...")
try:
    import judge.few_shot_judge
    print(f"   ✅ Module imported")
    print(f"   Module file: {judge.few_shot_judge.__file__}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Check module contents
print("\n2. Checking module contents...")
module_contents = [x for x in dir(judge.few_shot_judge) if not x.startswith('_')]
print(f"   Public names: {module_contents}")

# Test 3: Check if FewShotJudge exists
print("\n3. Looking for FewShotJudge class...")
if 'FewShotJudge' in dir(judge.few_shot_judge):
    print("   ✅ FewShotJudge found")
    FewShotJudge = judge.few_shot_judge.FewShotJudge
    print(f"   Class type: {type(FewShotJudge)}")
else:
    print("   ❌ FewShotJudge NOT found")
    print(f"   Available: {module_contents}")
    exit(1)

# Test 4: Try to instantiate
print("\n4. Creating instance...")
try:
    instance = FewShotJudge()
    print(f"   ✅ Instance created")
    print(f"   Judge name: {instance.judge_name}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All tests passed!")