# #############################################################################

# Troubleshooting
=====================
If RAG doesn't seem to work:
1. Check if vector store loaded:
bashcat logs/finance_agent.log | grep "RAG is ENABLED"
2. Verify vector store exists:
bashls -la src/finance_faiss_index/
Should show index.faiss and index.pkl
3. Test import manually:
bashcd AI_Finance_assistant
python -c "from src.agents.finance_agent import finance_agent; print('✅ Import successful')"
4. Check .env file:
bashcat .env | grep GOOGLE_API_KEY
Make sure your API key is there.

# #############################################################################

Cleaning up log files -
=====================
1. List all log files
bash python cleanup_logs.py list
2. Clean (delete) all logs with backup
bash python cleanup_logs.py clean
3. Clean without backup (⚠️ use with caution!)
bash python cleanup_logs.py clean --no-backup
4. Truncate logs (empty but keep files)
bash python cleanup_logs.py truncate ******** I use this a lot ***********
5. Archive logs to ZIP
bash python cleanup_logs.py archive
6. Backup logs only
bash python cleanup_logs.py backup
7. Custom log directory
bash python cleanup_logs.py clean --log-dir /path/to/logs
Quick Shell Scripts
You can also create quick shell scripts:
clean_logs.sh (Unix/Mac/Linux)
bash#!/bin/bash
echo "🧹 Cleaning logs..."
rm -rf logs/*.log
echo "✅ All log files deleted"
Make it executable:
bashchmod +x clean_logs.sh
./clean_logs.sh

# #############################################################################

semantic cache for RAG
======================
Usage:
bash# View stats
python src/tools/manage_cache.py stats
# Clear cache
python src/tools/manage_cache.py clear

# #############################################################################

# Verify installation for langSmith
=====================
python -c "import langsmith; print(langsmith.__version__)"

# #############################################################################

# to find a word in any script - 
=====================
    bash
    cd /Users/mandeep/myprojects/ai_finance_assistant
    grep -n "CONTEXT\|{context}" src/agents/tax_agent.py

# #############################################################################

# Clear cache
==============
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# #############################################################################
# Run Feedback Dashboard
=====================
python src/tools/feedback_dashboard.py
```

Should show:
```
================================================================================
RECENT QUALITY SCORES - FINANCE Agent
================================================================================

Query: does a financial advisor need to go to college to learn fi...
Score: 4.5/5
Retries: 0
Time: 2024-12-26T22:21:50

Query: what is compound interest...
Score: 4.8/5
Retries: 0
Time: 2024-12-26T22:20:15

================================================================================
PERFORMANCE METRICS
================================================================================

FINANCE Agent: ✅
  Success Rate: 95.0%
  Avg Latency: 3.85s
  Avg Quality Score: 4.65/5


# #############################################################################

# Check if retry logic exists
if grep -q "is_obvious_failure" src/agents/finance_agent.py; then
    echo "✅ Retry logic IS implemented"
    echo ""
    echo "Location:"
    grep -n "def is_obvious_failure" src/agents/finance_agent.py
else
    echo "❌ Retry logic NOT found"
    echo ""
    echo "The retry logic needs to be added to finance_agent.py"
fi

# Show the retry logic section
sed -n '450,480p' src/agents/finance_agent.py
# #############################################################################





# #############################################################################





# #############################################################################





# #############################################################################






# #############################################################################