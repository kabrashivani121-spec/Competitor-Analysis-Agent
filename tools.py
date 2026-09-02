"""
Custom tools for Competitor Analysis System
Includes SerpAPI wrapper and data processing utilities
"""

import json
import logging
import re
from collections import Counter
from typing import Any, Dict
from urllib.parse import urlparse
import config
from serpapi import GoogleSearch
from langchain_core.tools import BaseTool
from pydantic.v1 import Field

from trusted_sources import domain_matches, normalize_domain, trusted_organic_results

logger = logging.getLogger(__name__)


class SearchAPIError(RuntimeError):
    """Raised when SerpAPI rejects a search request."""


def infer_official_website(results: Dict, company_name: str) -> str:
    """Infer an official site from a knowledge graph or repeated branded top results."""
    knowledge_graph_url = (results.get("knowledge_graph") or {}).get("website", "")
    if knowledge_graph_url:
        return knowledge_graph_url

    brand_terms = [
        token
        for token in re.findall(r"[a-z0-9]+", company_name.lower())
        if len(token) >= 3 and token not in {"company", "group", "holdings", "inc", "plc"}
    ]
    candidates = []
    for item in results.get("organic_results", [])[:8]:
        url = item.get("link", "")
        domain = normalize_domain(url)
        label = f"{item.get('title', '')} {item.get('source', '')}".lower()
        if domain and brand_terms and any(term in label for term in brand_terms):
            candidates.append((domain, url))

    counts = Counter(domain for domain, _ in candidates)
    for domain, count in counts.most_common():
        if count >= 2:
            first_url = next(url for candidate, url in candidates if candidate == domain)
            parsed = urlparse(first_url)
            return f"{parsed.scheme}://{parsed.netloc}/"
    return ""


def execute_serpapi_search(params: Dict) -> Dict:
    """Execute a bounded SerpAPI request and validate its response."""
    if not params.get("api_key"):
        raise SearchAPIError("SERPAPI_API_KEY is not configured")

    search = GoogleSearch(params)
    search.timeout = config.SERPAPI_TIMEOUT_SECONDS
    results = search.get_dict()

    if not isinstance(results, dict):
        raise SearchAPIError("SerpAPI returned an invalid response")
    if results.get("error"):
        raise SearchAPIError(str(results["error"]))

    return results


class CompetitorSearchTool(BaseTool):
    """Tool for searching competitor information using SerpAPI"""
    
    name: str = "Competitor Search Tool"
    description: str = """Search for competitor information, company details, pricing, and reviews.
    Input should be a search query string. Returns structured information about competitors."""
    
    api_key: str = Field(default=config.SERPAPI_API_KEY)
    max_results: int = Field(default=10)
    allowed_domains: list[str] = Field(default_factory=list)
    
    def _run(self, query: str) -> str:
        """Execute competitor search"""
        try:
            logger.info(f"Searching for: {query}")
            
            # Execute search
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": min(max(self.max_results * 3, 10), 30),
                "engine": "google"
            }
            
            results = execute_serpapi_search(params)
            
            # Process results
            processed_results = self._process_search_results(results)
            
            return json.dumps(processed_results, indent=2)
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}")
            return json.dumps({
                "error": str(e),
                "query": query,
                "results": []
            })
    
    def _process_search_results(self, results: Dict) -> Dict:
        """Process and structure search results"""
        accepted, rejected = trusted_organic_results(results, self.allowed_domains)
        processed = {
            "organic_results": [],
            "knowledge_graph": {},
            "trusted_source_policy": "recognized publishers and official websites only",
            "untrusted_results_excluded": len(rejected),
        }
        
        # Extract organic results
        if accepted:
            for result in accepted[:self.max_results]:
                processed["organic_results"].append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0),
                    "trust_classification": result.get("trust_classification", ""),
                })
        
        # Extract knowledge graph if available
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            if kg.get("website"):
                processed["knowledge_graph"] = {
                    "title": kg.get("title", ""),
                    "type": kg.get("type", ""),
                    "website": kg.get("website", ""),
                    "trust_classification": "official company / competitor website",
                }
        
        return processed


class CoverageReportSearchTool(CompetitorSearchTool):
    """Search recognized publishers for coverage and industry-outlook reports."""

    name: str = "Trusted Coverage and Industry Outlook Search"
    description: str = """Find initiating-coverage, equity-research, industry-outlook, market-outlook,
    annual-report, and regulatory-filing sources. Only recognized institutions and official company
    or regulator domains are returned. Input should be a company, competitor, or industry topic."""
    max_results: int = Field(default=8)

    def _run(self, topic: str) -> str:
        query = (
            f'"{topic}" ("initiating coverage" OR "equity research" OR '
            '"industry outlook" OR "market outlook" OR "annual report" OR "10-K")'
        )
        payload = json.loads(super()._run(query))
        if payload.get("error"):
            return json.dumps(payload, indent=2)

        ignored = {
            "annual", "coverage", "equity", "filing", "industry", "initiating",
            "investor", "market", "outlook", "report", "research", "sec", "website",
        }
        terms = [
            token
            for token in re.findall(r"[a-z0-9]+", topic.lower())
            if len(token) >= 4 and token not in ignored and not token.isdigit()
        ]
        if terms:
            original = payload.get("organic_results", [])
            primary_term = terms[0]
            payload["organic_results"] = [
                item
                for item in original
                if primary_term in " ".join(
                    str(item.get(field, "")).lower()
                    for field in ("title", "link", "snippet")
                )
            ]
            payload["irrelevant_results_excluded"] = (
                len(original) - len(payload["organic_results"])
            )
        return json.dumps(payload, indent=2)


class OfficialWebsiteSearchTool(BaseTool):
    """Discover and search an organization's verified official website."""

    name: str = "Official Company or Competitor Website Search"
    description: str = """Search a verified official company or competitor domain. Input must be
    `Company Name | research topic`, for example `Porsche | annual report competitors` or
    `BMW | pricing electric vehicles`. The official domain is taken from Google's knowledge graph;
    results outside that domain are rejected."""
    api_key: str = Field(default=config.SERPAPI_API_KEY)
    max_results: int = Field(default=8)
    allowed_domains: list[str] = Field(default_factory=list)

    def _run(self, request: str) -> str:
        try:
            company_name, separator, topic = request.partition("|")
            company_name = company_name.strip()
            topic = topic.strip() if separator else "annual report investor relations products competitors"
            if not company_name:
                raise ValueError("A company name is required before the | separator")

            identity_results = execute_serpapi_search({
                "q": f'"{company_name}" official website investor relations',
                "api_key": self.api_key,
                "engine": "google",
                "num": 10,
            })
            website = infer_official_website(identity_results, company_name)
            official_domain = normalize_domain(website)

            if not official_domain:
                accepted_identity, _ = trusted_organic_results(
                    identity_results,
                    self.allowed_domains,
                )
                if accepted_identity:
                    official_domain = normalize_domain(accepted_identity[0].get("link", ""))
                    website = accepted_identity[0].get("link", "")

            if not official_domain:
                return json.dumps({
                    "error": "No verified official domain was found; provide an official URL in the sidebar.",
                    "company_name": company_name,
                    "organic_results": [],
                }, indent=2)

            source_results = execute_serpapi_search({
                "q": f"site:{official_domain} {topic}",
                "api_key": self.api_key,
                "engine": "google",
                "num": min(max(self.max_results * 2, 10), 20),
            })
            accepted, rejected = trusted_organic_results(source_results, [official_domain])
            return json.dumps({
                "company_name": company_name,
                "official_website": website,
                "official_domain": official_domain,
                "trust_classification": "official company / competitor website",
                "organic_results": [
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "trust_classification": item.get("trust_classification", ""),
                    }
                    for item in accepted[:self.max_results]
                ],
                "untrusted_results_excluded": len(rejected),
            }, indent=2)
        except Exception as exc:
            logger.error("Official website search failed for '%s': %s", request, exc)
            return json.dumps({
                "error": str(exc),
                "request": request,
                "organic_results": [],
            }, indent=2)


class CompanyInfoTool(BaseTool):
    """Tool for getting detailed company information"""
    
    name: str = "Company Information Tool"
    description: str = """Get detailed information about a specific company including 
    description, website, industry, and key facts. Input should be the company name."""
    
    api_key: str = Field(default=config.SERPAPI_API_KEY)
    allowed_domains: list[str] = Field(default_factory=list)
    
    def _run(self, company_name: str) -> str:
        """Get company information"""
        try:
            logger.info(f"Getting info for company: {company_name}")
            
            # Search for company information
            params = {
                "q": f"{company_name} company information",
                "api_key": self.api_key,
                "engine": "google",
                "num": 15,
            }
            
            results = execute_serpapi_search(params)
            
            # Extract company info
            company_info = self._extract_company_info(results, company_name)
            
            return json.dumps(company_info, indent=2)
            
        except Exception as e:
            logger.error(f"Error getting company info for '{company_name}': {str(e)}")
            return json.dumps({
                "error": str(e),
                "company_name": company_name
            })
    
    def _extract_company_info(self, results: Dict, company_name: str) -> Dict:
        """Extract structured company information from search results"""
        info = {
            "name": company_name,
            "description": "",
            "website": "",
            "industry": "",
            "founded": "",
            "headquarters": "",
            "key_facts": []
        }
        
        # Use the knowledge graph only to identify the official website. Descriptive
        # evidence must still come from an accepted source URL.
        info["website"] = infer_official_website(results, company_name)
        official_domain = normalize_domain(info["website"])
        accepted, _ = trusted_organic_results(
            results,
            [*self.allowed_domains, *([official_domain] if official_domain else [])],
        )
        
        if accepted:
            preferred = next(
                (
                    item
                    for item in accepted
                    if official_domain
                    and domain_matches(normalize_domain(item.get("link", "")), official_domain)
                ),
                accepted[0],
            )
            info["description"] = preferred.get("snippet", "")
            info["website"] = info["website"] or preferred.get("link", "")
            info["sources"] = [{
                "title": preferred.get("title", ""),
                "url": preferred.get("link", ""),
                "trust_classification": preferred.get("trust_classification", ""),
            }]
        
        return info


class PricingSearchTool(BaseTool):
    """Tool for finding pricing information"""
    
    name: str = "Pricing Search Tool"
    description: str = """Search for pricing information for a specific company or product.
    Input should be the company/product name. Returns pricing details if available."""
    
    api_key: str = Field(default=config.SERPAPI_API_KEY)
    allowed_domains: list[str] = Field(default_factory=list)
    
    def _run(self, company_name: str) -> str:
        """Search for pricing information"""
        try:
            logger.info(f"Searching pricing for: {company_name}")
            
            # Search for pricing
            params = {
                "q": f"{company_name} pricing plans cost",
                "api_key": self.api_key,
                "engine": "google",
                "num": 15,
            }
            
            results = execute_serpapi_search(params)
            
            # Extract pricing info
            pricing_info = self._extract_pricing_info(results, company_name)
            
            return json.dumps(pricing_info, indent=2)
            
        except Exception as e:
            logger.error(f"Error searching pricing for '{company_name}': {str(e)}")
            return json.dumps({
                "error": str(e),
                "company_name": company_name
            })
    
    def _extract_pricing_info(self, results: Dict, company_name: str) -> Dict:
        """Extract pricing information from search results"""
        pricing = {
            "company_name": company_name,
            "pricing_found": False,
            "pricing_details": [],
            "sources": []
        }
        
        accepted, _ = trusted_organic_results(results, self.allowed_domains)
        if accepted:
            for result in accepted:
                snippet = result.get("snippet", "").lower()
                
                # Look for pricing indicators
                if any(keyword in snippet for keyword in ["price", "pricing", "$", "cost", "plan"]):
                    pricing["pricing_found"] = True
                    pricing["pricing_details"].append({
                        "source": result.get("title", ""),
                        "link": result.get("link", ""),
                        "description": result.get("snippet", ""),
                        "trust_classification": result.get("trust_classification", ""),
                    })
                    pricing["sources"].append(result.get("link", ""))
        
        return pricing


class ReviewSearchTool(BaseTool):
    """Tool for finding customer reviews and sentiment"""
    
    name: str = "Review Search Tool"
    description: str = """Search for customer reviews and feedback about a company or product.
    Input should be the company/product name. Returns review summaries and sentiment."""
    
    api_key: str = Field(default=config.SERPAPI_API_KEY)
    allowed_domains: list[str] = Field(default_factory=list)
    
    def _run(self, company_name: str) -> str:
        """Search for reviews"""
        try:
            logger.info(f"Searching reviews for: {company_name}")
            
            # Search for reviews
            params = {
                "q": f"{company_name} reviews customer feedback",
                "api_key": self.api_key,
                "engine": "google",
                "num": 15,
            }
            
            results = execute_serpapi_search(params)
            
            # Extract review info
            review_info = self._extract_review_info(results, company_name)
            
            return json.dumps(review_info, indent=2)
            
        except Exception as e:
            logger.error(f"Error searching reviews for '{company_name}': {str(e)}")
            return json.dumps({
                "error": str(e),
                "company_name": company_name
            })
    
    def _extract_review_info(self, results: Dict, company_name: str) -> Dict:
        """Extract review information from search results"""
        reviews = {
            "company_name": company_name,
            "reviews_found": False,
            "review_sources": [],
            "sentiment_indicators": []
        }
        
        accepted, _ = trusted_organic_results(results, self.allowed_domains)
        if accepted:
            for result in accepted:
                snippet = result.get("snippet", "")
                
                # Look for review indicators
                if any(keyword in snippet.lower() for keyword in ["review", "rating", "customer", "feedback"]):
                    reviews["reviews_found"] = True
                    reviews["review_sources"].append({
                        "source": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": snippet,
                        "trust_classification": result.get("trust_classification", ""),
                    })
        
        return reviews


class DataProcessorTool(BaseTool):
    """Tool for processing and structuring competitor data"""
    
    name: str = "Data Processor Tool"
    description: str = """Process raw competitor data into structured format.
    Input should be JSON string of competitor data. Returns cleaned and structured data."""
    
    def _run(
        self,
        data: str | dict | list | None = None,
        competitors: list[dict[str, Any]] | None = None,
    ) -> str:
        """Process competitor data"""
        try:
            if competitors is not None:
                data_dict = {"competitors": competitors}
            # Parse input data
            elif isinstance(data, str):
                try:
                    data_dict = json.loads(data)
                except json.JSONDecodeError:
                    data_dict = {"raw_data": data}
            elif data is None:
                data_dict = {}
            else:
                data_dict = data

            if isinstance(data_dict, list):
                data_dict = {"competitors": data_dict}
            
            # Structure the data
            processed = self._structure_data(data_dict)
            
            return json.dumps(processed, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return json.dumps({
                "error": str(e),
                "raw_data": str(data)
            })
    
    def _structure_data(self, data: Dict) -> Dict:
        """Structure data into standard format"""
        structured = {
            "competitors": [],
            "market_overview": {},
            "data_quality": "processed"
        }
        
        # Process competitor information
        if "competitors" in data:
            for comp in data["competitors"]:
                structured["competitors"].append({
                    "name": comp.get("name", ""),
                    "website": comp.get("website", ""),
                    "description": comp.get("description", ""),
                    "strengths": comp.get("strengths", []),
                    "weaknesses": comp.get("weaknesses", [])
                })
        
        return structured


# Create tool instances
competitor_search_tool = CompetitorSearchTool()
company_info_tool = CompanyInfoTool()
pricing_search_tool = PricingSearchTool()
review_search_tool = ReviewSearchTool()
data_processor_tool = DataProcessorTool()
