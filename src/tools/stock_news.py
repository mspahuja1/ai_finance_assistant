"""Stock News Tool - Updated for new yfinance API structure"""
import time
import sys
import os
from typing import Annotated
from datetime import datetime

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from langchain_core.tools import tool
from tools.cache import market_cache
from utils.logging_config import create_logger

news_logger = create_logger("news_tool", "news_tool.log")

@tool
def get_stock_news(symbol: Annotated[str, "Stock ticker symbol (e.g., AAPL, RIV)"]) -> str:
    """Fetch recent news articles for a stock symbol"""
    
    request_id = f"news-{int(time.time() * 1000)}"
    news_logger.info(f"📰 NEWS REQUEST [{request_id}] - {symbol}")
    
    # Check cache
    cached_result = market_cache.get("get_stock_news", symbol=symbol)
    if cached_result:
        news_logger.info(f"⚡ CACHE HIT [{request_id}]")
        return cached_result
    
    start_time = time.time()
    
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        news_logger.info(f"DEBUG: Got {len(news) if news else 0} news items")
        
        if not news or len(news) == 0:
            news_logger.warning(f"No news returned for {symbol}")
            return f"No recent news found for {symbol}"
        
        result = f"📰 Recent News for {symbol}:\n\n"
        
        for i, article in enumerate(news[:5], 1):
            # Handle new nested structure
            # The actual content is in article['content']
            content = article.get('content', article)  # Fallback to article if no 'content' key
            
            # Extract fields from the nested content
            title = content.get('title', 'No title')
            
            # Publisher/Provider
            provider = content.get('provider', {})
            publisher = provider.get('displayName', 'Unknown') if isinstance(provider, dict) else 'Unknown'
            
            # Link - try multiple possible fields
            link = (content.get('clickThroughUrl') or 
                   content.get('previewUrl') or 
                   content.get('canonicalUrl', {}).get('url') if isinstance(content.get('canonicalUrl'), dict) else None or
                   '#')
            
            # Date - handle new format
            pub_date = content.get('pubDate') or content.get('displayTime')
            if pub_date:
                try:
                    # Parse ISO format: 2025-12-13T16:48:00Z
                    date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    date = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    date = pub_date[:16]  # Just take first 16 chars
            else:
                date = 'Recent'
            
            # Summary (optional)
            summary = content.get('summary', '')
            
            result += f"{i}. {title}\n"
            result += f"   Publisher: {publisher}\n"
            result += f"   Date: {date}\n"
            if summary and len(summary) > 0:
                # Limit summary to 150 chars
                summary_short = summary[:150] + "..." if len(summary) > 150 else summary
                result += f"   Summary: {summary_short}\n"
            result += f"   Link: {link}\n\n"
        
        # Cache the result
        market_cache.set("get_stock_news", result, symbol=symbol)
        
        elapsed = time.time() - start_time
        news_logger.info(f"✅ SUCCESS [{request_id}] - {elapsed:.3f}s - {len(news)} articles")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        news_logger.error(f"❌ ERROR [{request_id}] - {elapsed:.3f}s - {str(e)}")
        import traceback
        news_logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error fetching news: {str(e)}"