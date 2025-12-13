import sys
import importlib
import logging
import os

# Configure logging: write to file
log_dir = "/Users/mandeep/myprojects/ai_finance_assistant/logs"
os.makedirs(log_dir, exist_ok=True)  # ensures the folder exists

log_path = os.path.join(log_dir, "env_check.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w")]
)


def check_package(package_name, import_name=None):
    """Try to import a package and log + print its version."""
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        msg = f"✅ {package_name}: {version}"
        print(msg)          # console output
        logging.info(msg)   # log file
        return True
    except ImportError:
        msg = f"❌ {package_name}: Not installed"
        print(msg)
        logging.error(msg)
        return False

def main():
    header = "=" * 60
    print(header)
    print("🔍 MCP + LangGraph Environment Verification")
    print(header)
    logging.info(header)
    logging.info("🔍 MCP + LangGraph Environment Verification")
    logging.info(header)

    all_good = True

    # Check Python version
    python_version = sys.version.split()[0]
    if sys.version_info >= (3, 8):
        msg = f"✅ Python: {python_version}"
        print(msg)
        logging.info(msg)
    else:
        msg = f"❌ Python: {python_version} (Need 3.8+)"
        print(msg)
        logging.error(msg)
        all_good = False

    # Check required packages
    packages = {
        "langchain": "langchain",
        "langgraph": "langgraph",
        "langchain-openai": "langchain_openai",
        "langchain-anthropic": "langchain_anthropic",
        "faiss-cpu": "faiss",
        "chromadb": "chromadb",
        "sentence-transformers": "sentence_transformers",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scikit-learn": "sklearn",
        "yfinance": "yfinance",
    }

    for pkg, import_name in packages.items():
        ok = check_package(pkg, import_name)
        if not ok:
            all_good = False

    summary = "🎉 All required dependencies are installed!" if all_good else "⚠️ Some dependencies are missing."
    print(summary)
    logging.info(summary if all_good else summary)

if __name__ == "__main__":
    main()