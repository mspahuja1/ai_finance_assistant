"""setup_rag.py - Build the finance knowledge base"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))  # tools folder
src_dir = os.path.dirname(current_dir)  # src folder
sys.path.insert(0, src_dir)

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables")

class FinanceRAG:
    """RAG system for financial knowledge"""
    
    def __init__(self, api_key: str):
        """Initialize RAG system with FAISS vector store"""
        logger.info("Initializing FinanceRAG system...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        logger.info("✅ FinanceRAG initialized")
        
    def load_wikipedia_data(self, topics: List[str]):
        """Load financial data from Wikipedia"""
        logger.info(f"📚 Loading Wikipedia articles for {len(topics)} topics...")
        
        all_docs = []
        success_count = 0
        
        for i, topic in enumerate(topics, 1):
            try:
                logger.info(f"  [{i}/{len(topics)}] Loading: {topic}")
                loader = WikipediaLoader(query=topic, load_max_docs=2)
                docs = loader.load()
                all_docs.extend(docs)
                success_count += 1
                logger.info(f"    ✅ Loaded {len(docs)} documents")
            except Exception as e:
                logger.error(f"    ❌ Failed to load '{topic}': {e}")
        
        logger.info(f"📊 Successfully loaded {success_count}/{len(topics)} topics")
        logger.info(f"📄 Total documents: {len(all_docs)}")
        
        return all_docs
    
    def create_vector_store(self, documents: List):
        """Create FAISS vector store from documents"""
        logger.info(f"🔨 Creating vector store from {len(documents)} documents...")
        
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        # Split documents into chunks
        logger.info("  Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"  ✅ Created {len(splits)} chunks")
        
        # Create FAISS vector store
        logger.info("  Creating FAISS index...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        logger.info("  ✅ FAISS index created successfully")
        
        return self.vector_store
    
    def save_vector_store(self, path: str):
        """Save FAISS index to disk"""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        logger.info(f"💾 Saving vector store to: {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        self.vector_store.save_local(path)
        logger.info(f"✅ Vector store saved successfully")


def main():
    """Main function to build the knowledge base"""
    
    logger.info("=" * 70)
    logger.info("🚀 STARTING FINANCE RAG KNOWLEDGE BASE CREATION")
    logger.info("=" * 70)
    
    # Initialize RAG
    rag = FinanceRAG(api_key)
    
    # Define financial topics to load from Wikipedia
    financial_topics = [
        "Personal finance",
        "Investment",
        "Stock market",
        "Bond (finance)",
        "Mutual fund",
        "Exchange-traded fund",
        "Retirement planning",
        "Portfolio (finance)",
        "Diversification (finance)",
        "Asset allocation",
        "Risk management",
        "Financial planning",
        "Compound interest",
        "Dollar cost averaging",
        "Index fund",
        "Value investing",
        "Dividend",
        "Capital gains",
        "Tax-advantaged account",
        "401k",
        "Individual retirement account",
        "Stock valuation",
        "Financial independence"
    ]
    
    logger.info(f"📋 Topics to load: {len(financial_topics)}")
    
    # Load data from Wikipedia
    docs = rag.load_wikipedia_data(financial_topics)
    
    if not docs:
        logger.error("❌ No documents loaded. Exiting.")
        return
    
    # Create vector store
    rag.create_vector_store(docs)
    
    # Save to src folder (one level up from tools folder)
    vector_store_path = os.path.join(src_dir, "finance_faiss_index")
    rag.save_vector_store(vector_store_path)
    
    logger.info("=" * 70)
    logger.info("✅ RAG KNOWLEDGE BASE CREATED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"📁 Location: {vector_store_path}")
    logger.info(f"📊 Total documents processed: {len(docs)}")
    logger.info("")
    logger.info("🎯 Next steps:")
    logger.info("   1. The finance agent will automatically use this knowledge base")
    logger.info("   2. Run your main application as usual")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)