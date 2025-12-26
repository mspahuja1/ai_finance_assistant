"""Agents package"""
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

# --- Agent imports ---
from agents.router import router_agent
from agents.finance_agent import finance_agent
from agents.portfolio_agent import portfolio_agent
from agents.market_agent import market_agent, market_agent_should_continue
from agents.goal_agent import goal_agent
from agents.news_agent import news_agent, news_agent_should_continue
from agents.tax_agent import tax_agent



__all__ = [
    # Agents
    'router_agent',
    'finance_agent',
    'portfolio_agent',
    'market_agent',
    'market_agent_should_continue',
    'goal_agent',
    'news_agent',
    'news_agent_should_continue',
    'tax_agent',

]