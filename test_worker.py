"""Test if async feedback worker is running"""

import sys
import time
sys.path.insert(0, 'src')

print("Testing async feedback worker...")

try:
    from feedback.async_feedback import async_feedback
    
    print(f"✅ Import successful")
    print(f"Worker running: {async_feedback.is_running}")
    print(f"Worker thread: {async_feedback.worker_thread}")
    print(f"Thread alive: {async_feedback.worker_thread.is_alive() if async_feedback.worker_thread else 'N/A'}")
    print(f"Queue size: {async_feedback.queue.qsize()}")
    
    # Queue a test item
    print("\nQueueing test feedback...")
    async_feedback.queue_quality_check(
        query="Test query",
        response="This is a test response with enough content to be evaluated properly.",
        agent_type="finance",
        retry_count=0
    )
    
    print(f"Queue size after adding: {async_feedback.queue.qsize()}")
    
    # Wait for processing
    print("Waiting 5 seconds for processing...")
    time.sleep(5)
    
    print(f"Queue size after waiting: {async_feedback.queue.qsize()}")
    
    # Check if files were created
    import os
    quality_dir = "feedback/quality_scores"
    if os.path.exists(quality_dir):
        files = os.listdir(quality_dir)
        print(f"\n✅ Quality scores directory exists")
        print(f"Files in directory: {len(files)}")
        if files:
            print(f"Latest file: {files[-1]}")
        else:
            print("⚠️ No files created yet")
    else:
        print(f"\n❌ Quality scores directory doesn't exist: {quality_dir}")
    
    # Check metrics
    metrics_file = "feedback/performance_metrics/current_metrics.json"
    if os.path.exists(metrics_file):
        print(f"\n✅ Metrics file exists")
        with open(metrics_file) as f:
            content = f.read()
            print(f"Content length: {len(content)} chars")
    else:
        print(f"\n⚠️ Metrics file doesn't exist yet: {metrics_file}")
    
    print("\n✅ Test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
