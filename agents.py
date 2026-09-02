"""
Agent definitions for Competitor Analysis System
Defines three specialized agents: Research, Analysis, and Report
"""

import logging
import config
from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import (
    CompanyInfoTool,
    CompetitorSearchTool,
    CoverageReportSearchTool,
    DataProcessorTool,
    OfficialWebsiteSearchTool,
    PricingSearchTool,
    ReviewSearchTool,
)

logger = logging.getLogger(__name__)


def create_llm(temperature: float = 0.7):
    """Create and configure the LLM instance"""
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
        request_timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
    )


def create_research_agent(
    company_name: str,
    industry: str,
    analysis_depth: str = config.ANALYSIS_DEPTH,
    trusted_domains: list[str] | None = None,
) -> Agent:
    """
    Create Research Agent - Specialist in gathering competitor data
    
    Args:
        company_name: Name of the company being analyzed
        industry: Industry sector
        
    Returns:
        Agent: Configured research agent
    """
    logger.info(f"Creating Research Agent for {company_name} in {industry}")
    
    role = config.RESEARCH_AGENT_ROLE
    
    goal = config.RESEARCH_AGENT_GOAL.format(
        company_name=company_name,
        industry=industry
    )
    
    backstory = config.RESEARCH_AGENT_BACKSTORY
    
    depth_config = config.ANALYSIS_DEPTH_CONFIG.get(
        analysis_depth,
        config.ANALYSIS_DEPTH_CONFIG["standard"],
    )

    trusted_domains = trusted_domains or []
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[
            CompetitorSearchTool(max_results=depth_config["max_search_results"], allowed_domains=trusted_domains),
            CoverageReportSearchTool(allowed_domains=trusted_domains),
            OfficialWebsiteSearchTool(allowed_domains=trusted_domains),
            CompanyInfoTool(allowed_domains=trusted_domains),
            PricingSearchTool(allowed_domains=trusted_domains),
            ReviewSearchTool(allowed_domains=trusted_domains),
        ],
        llm=create_llm(temperature=0.5),
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        memory=True
    )
    
    logger.info("Research Agent created successfully")
    return agent


def create_analysis_agent(company_name: str, industry: str) -> Agent:
    """
    Create Analysis Agent - Expert in competitive analysis and SWOT
    
    Args:
        company_name: Name of the company being analyzed
        industry: Industry sector
        
    Returns:
        Agent: Configured analysis agent
    """
    logger.info(f"Creating Analysis Agent for {company_name} in {industry}")
    
    role = config.ANALYSIS_AGENT_ROLE
    
    goal = config.ANALYSIS_AGENT_GOAL.format(
        company_name=company_name,
        industry=industry
    )
    
    backstory = config.ANALYSIS_AGENT_BACKSTORY
    
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[DataProcessorTool()],
        llm=create_llm(temperature=0.4),
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        memory=True
    )
    
    logger.info("Analysis Agent created successfully")
    return agent


def create_report_agent(company_name: str, industry: str) -> Agent:
    """
    Create Report Agent - Specialist in synthesizing insights and recommendations
    
    Args:
        company_name: Name of the company being analyzed
        industry: Industry sector
        
    Returns:
        Agent: Configured report agent
    """
    logger.info(f"Creating Report Agent for {company_name} in {industry}")
    
    role = config.REPORT_AGENT_ROLE
    
    goal = config.REPORT_AGENT_GOAL.format(
        company_name=company_name,
        industry=industry
    )
    
    backstory = config.REPORT_AGENT_BACKSTORY
    
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[],  # Report agent synthesizes, doesn't need search tools
        llm=create_llm(temperature=0.6),
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        memory=True
    )
    
    logger.info("Report Agent created successfully")
    return agent


def create_product_agent(company_name: str, industry: str, our_product: str = "") -> Agent:
    """Create the product-comparison specialist."""
    product_context = our_product.strip() or company_name
    return Agent(
        role=config.PRODUCT_AGENT_ROLE,
        goal=config.PRODUCT_AGENT_GOAL.format(
            company_name=product_context,
            industry=industry,
        ),
        backstory=config.PRODUCT_AGENT_BACKSTORY,
        tools=[],
        llm=create_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        memory=True,
    )


def create_evaluator_agent(
    company_name: str,
    industry: str,
    trusted_domains: list[str] | None = None,
) -> Agent:
    """Create the evidence and quality-review specialist."""
    return Agent(
        role=config.EVALUATOR_AGENT_ROLE,
        goal=config.EVALUATOR_AGENT_GOAL.format(
            company_name=company_name,
            industry=industry,
        ),
        backstory=config.EVALUATOR_AGENT_BACKSTORY,
        tools=[
            CoverageReportSearchTool(allowed_domains=trusted_domains or []),
            OfficialWebsiteSearchTool(allowed_domains=trusted_domains or []),
            CompanyInfoTool(allowed_domains=trusted_domains or []),
        ],
        llm=create_llm(temperature=0.2),
        verbose=True,
        allow_delegation=False,
        max_iter=8,
        memory=True,
    )


def create_all_agents(
    company_name: str,
    industry: str,
    analysis_depth: str = config.ANALYSIS_DEPTH,
    our_product: str = "",
    trusted_domains: list[str] | None = None,
) -> dict:
    """
    Create all three agents for the competitor analysis system
    
    Args:
        company_name: Name of the company being analyzed
        industry: Industry sector
        
    Returns:
        dict: Dictionary containing all three agents
    """
    logger.info(f"Creating all agents for {company_name} in {industry}")
    
    agents = {
        "research": create_research_agent(company_name, industry, analysis_depth, trusted_domains),
        "analysis": create_analysis_agent(company_name, industry),
        "product": create_product_agent(company_name, industry, our_product),
        "evaluator": create_evaluator_agent(company_name, industry, trusted_domains),
        "report": create_report_agent(company_name, industry)
    }
    
    logger.info("All agents created successfully")
    return agents
