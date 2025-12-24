Troubleshooting
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


Cleaning up log files -
=====================
1. List all log files
bashpython cleanup_logs.py list
2. Clean (delete) all logs with backup
bashpython cleanup_logs.py clean
3. Clean without backup (⚠️ use with caution!)
bashpython cleanup_logs.py clean --no-backup
4. Truncate logs (empty but keep files)
bashpython cleanup_logs.py truncate
5. Archive logs to ZIP
bashpython cleanup_logs.py archive
6. Backup logs only
bashpython cleanup_logs.py backup
7. Custom log directory
bashpython cleanup_logs.py clean --log-dir /path/to/logs
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


semantic cache for RAG
======================
Usage:
bash# View stats
python src/tools/manage_cache.py stats
# Clear cache
python src/tools/manage_cache.py clear