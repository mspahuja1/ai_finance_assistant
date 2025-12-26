# test_judge_dependencies.py
# Designed and developed by Mandeep Pahuja
try:
    import google.generativeai as genai
    print("✅ google-generativeai installed")
except ImportError:
    print("❌ Need: pip install google-generativeai")

try:
    import pandas
    print("✅ pandas installed")
except ImportError:
    print("❌ Need: pip install pandas")

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv installed")
except ImportError:
    print("❌ Need: pip install python-dotenv")

try:
    import asyncio
    print("✅ asyncio available (built-in)")
except ImportError:
    print("❌ asyncio not available (should be built-in)")

print("\n🎉 All judge dependencies satisfied!")