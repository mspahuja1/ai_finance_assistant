"""
Multi-Agent Financial Advisory System using LangGraph 1.04
A modular framework with 6 specialized agents and RAG capabilities
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import operator

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state across all agents in the workflow.
    Tracks conversation history, routing decisions, and context.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    user_query: str
    context: dict  # Stores retrieved documents, user profile, etc.
    agent_responses: dict  # Tracks responses from each agent
    confidence_score: float  # Router confidence in agent selection


# ============================================================================
# AGENT BASE CLASS
# ============================================================================

class BaseFinancialAgent:
    """Base class for all financial agents with common RAG functionality"""
    
    def __init__(self, name: str, llm, retriever=None):
        self.name = name
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Override in subclasses to define agent-specific behavior"""
        return f"You are a {self.name} specialized in financial services."
    
    def retrieve_context(self, query: str) -> list:
        """RAG retrieval - fetch relevant documents"""
        if self.retriever:
            docs = self.retriever.get_relevant_documents(query)
            return [doc.page_content for doc in docs]
        return []
    
    def process(self, state: AgentState) -> AgentState:
        """Main processing logic for the agent"""
        query = state["user_query"]
        
        # RAG: Retrieve relevant context
        context_docs = self.retrieve_context(query)
        context_str = "\n\n".join(context_docs) if context_docs else "No additional context available."
        
        # Build prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"Context:\n{context_str}\n\nUser Query: {query}")
        ])
        
        # Generate response
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query})
        
        # Update state
        state["messages"].append(AIMessage(content=response, name=self.name))
        state["agent_responses"][self.name] = response
        
        return state


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class FinanceQAAgent(BaseFinancialAgent):
    """Handles general financial education queries"""
    
    def _get_system_prompt(self) -> str:
        return """You are a Finance Q&A Agent specializing in financial education.
        
Your role:
- Answer general questions about financial concepts, terminology, and principles
- Provide clear, educational explanations suitable for various knowledge levels
- Use examples and analogies to clarify complex topics
- Maintain accuracy and cite sources when providing specific data

Keep responses concise yet comprehensive."""


class PortfolioAnalysisAgent(BaseFinancialAgent):
    """Reviews and analyzes user portfolios"""
    
    def _get_system_prompt(self) -> str:
        return """You are a Portfolio Analysis Agent specializing in investment portfolio review.
        
Your role:
- Analyze portfolio composition, diversification, and risk exposure
- Identify potential issues like concentration risk or misalignment with goals
- Provide actionable insights for portfolio optimization
- Consider risk tolerance, time horizon, and financial goals
- Use data from the context to provide specific recommendations

Be thorough but accessible in your analysis."""


class MarketAnalysisAgent(BaseFinancialAgent):
    """Provides real-time market insights"""
    
    def _get_system_prompt(self) -> str:
        return """You are a Market Analysis Agent specializing in market insights and trends.
        
Your role:
- Analyze current market conditions and trends
- Provide context for recent market movements
- Explain sector performance and economic indicators
- Connect market events to potential portfolio impacts
- Use real-time data from the context when available

Focus on actionable insights rather than predictions."""


class GoalPlanningAgent(BaseFinancialAgent):
    """Assists with financial goal setting and planning"""
    
    def _get_system_prompt(self) -> str:
        return """You are a Goal Planning Agent specializing in financial goal setting.
        
Your role:
- Help users define and structure financial goals (retirement, education, home purchase)
- Calculate required savings rates and timelines
- Suggest appropriate investment strategies for different goal timeframes
- Break down complex long-term goals into actionable steps
- Consider inflation, expected returns, and risk factors

Be encouraging while maintaining realistic expectations."""


class NewsSynthesizerAgent(BaseFinancialAgent):
    """Summarizes and contextualizes financial news"""
    
    def _get_system_prompt(self) -> str:
        return """You are a News Synthesizer Agent specializing in financial news analysis.
        
Your role:
- Summarize complex financial news into digestible insights
- Contextualize news events within broader market trends
- Explain potential impacts on different investment types
- Filter noise from signal in financial media
- Connect news to user's potential interests or portfolio

Remain objective and avoid sensationalism."""


class TaxEducationAgent(BaseFinancialAgent):
    """Explains tax concepts and account types"""
    
    def _get_system_prompt(self) -> str:
        return """You are a Tax Education Agent specializing in tax concepts and account types.
        
Your role:
- Explain tax-advantaged accounts (401k, IRA, Roth IRA, HSA, 529)
- Clarify tax implications of different investment strategies
- Educate on tax-loss harvesting, capital gains, and deductions
- Compare tax treatment of different account types
- Provide general tax education (not personalized tax advice)

Always remind users to consult a tax professional for specific situations."""


# ============================================================================
# ROUTER AGENT
# ============================================================================

class RouterAgent:
    """Routes user queries to the most appropriate agent(s)"""
    
    def __init__(self, llm):
        self.llm = llm
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing agent for a financial advisory system. 
            Analyze the user query and determine which agent should handle it.
            
            Available agents:
            - finance_qa: General financial education and concepts
            - portfolio_analysis: Portfolio review and optimization
            - market_analysis: Market trends and insights
            - goal_planning: Financial goal setting and planning
            - news_synthesizer: Financial news summaries and context
            - tax_education: Tax concepts and account types
            
            Respond with ONLY the agent name, nothing else."""),
            ("human", "User query: {query}")
        ])
    
    def route(self, state: AgentState) -> AgentState:
        """Determine which agent should handle the query"""
        chain = self.routing_prompt | self.llm | StrOutputParser()
        next_agent = chain.invoke({"query": state["user_query"]}).strip().lower()
        
        # Validate agent name
        valid_agents = [
            "finance_qa", "portfolio_analysis", "market_analysis",
            "goal_planning", "news_synthesizer", "tax_education"
        ]
        
        if next_agent not in valid_agents:
            next_agent = "finance_qa"  # Default fallback
        
        state["next_agent"] = next_agent
        state["confidence_score"] = 0.85  # Placeholder - implement actual scoring
        
        return state


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

class FinancialAdvisoryWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self, llm, retrievers: dict = None):
        """
        Args:
            llm: Language model instance
            retrievers: Dict mapping agent names to their retrievers
        """
        self.llm = llm
        self.retrievers = retrievers or {}
        
        # Initialize agents
        self.agents = {
            "finance_qa": FinanceQAAgent("Finance Q&A", llm, self.retrievers.get("finance_qa")),
            "portfolio_analysis": PortfolioAnalysisAgent("Portfolio Analysis", llm, self.retrievers.get("portfolio_analysis")),
            "market_analysis": MarketAnalysisAgent("Market Analysis", llm, self.retrievers.get("market_analysis")),
            "goal_planning": GoalPlanningAgent("Goal Planning", llm, self.retrievers.get("goal_planning")),
            "news_synthesizer": NewsSynthesizerAgent("News Synthesizer", llm, self.retrievers.get("news_synthesizer")),
            "tax_education": TaxEducationAgent("Tax Education", llm, self.retrievers.get("tax_education")),
        }
        
        self.router = RouterAgent(llm)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router.route)
        for name, agent in self.agents.items():
            workflow.add_node(name, agent.process)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router to agents
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next_agent"],
            {name: name for name in self.agents.keys()}
        )
        
        # All agents lead to END
        for name in self.agents.keys():
            workflow.add_edge(name, END)
        
        return workflow.compile()
    
    def invoke(self, user_query: str, context: dict = None) -> dict:
        """
        Execute the workflow for a user query
        
        Args:
            user_query: The user's question
            context: Additional context (user profile, portfolio data, etc.)
        
        Returns:
            Final state with agent responses
        """
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "next_agent": "",
            "user_query": user_query,
            "context": context or {},
            "agent_responses": {},
            "confidence_score": 0.0
        }
        
        result = self.graph.invoke(initial_state)
        return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the framework"""
    
    # Initialize your LLM (placeholder - replace with actual LLM)
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    class MockLLM:
        """Mock LLM for demonstration"""
        def invoke(self, prompt):
            return "This is a mock response from the LLM."
    
    llm = MockLLM()
    
    # Initialize retrievers (placeholder - implement with your vector store)
    retrievers = {
        # "finance_qa": your_finance_qa_retriever,
        # "portfolio_analysis": your_portfolio_retriever,
        # etc.
    }
    
    # Create workflow
    workflow = FinancialAdvisoryWorkflow(llm, retrievers)
    
    # Example queries
    queries = [
        "What is dollar-cost averaging?",
        "Can you analyze my portfolio? I have 60% stocks and 40% bonds.",
        "What's happening in the market today?",
        "Help me plan for retirement in 30 years.",
        "Summarize the latest Fed meeting news.",
        "Explain the difference between Traditional and Roth IRA."
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = workflow.invoke(
            user_query=query,
            context={"user_profile": {"risk_tolerance": "moderate"}}
        )
        
        print(f"Routed to: {result['next_agent']}")
        print(f"Response: {result['agent_responses'].get(result['next_agent'], 'No response')}")


if __name__ == "__main__":
    print("Multi-Agent Financial Advisory System - LangGraph 1.04")
    print("="*60)
    example_usage()