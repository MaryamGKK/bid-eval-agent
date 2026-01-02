# services/searcher.py
"""Dual search service using Tavily + SerpAPI with parallel execution."""

import requests
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from tavily import TavilyClient

from config import TAVILY_API_KEY, SERPAPI_API_KEY
from models import CompanyInsight, KeySignals


class Searcher:
    """Search company information using Tavily and SerpAPI in parallel."""
    
    def __init__(self):
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required. Please configure it in .env file.")
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.cache: Dict[str, CompanyInsight] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _get_default_insight(self, company: str) -> CompanyInsight:
        """Return a default insight when search fails."""
        return CompanyInsight(
            sources=[],
            key_signals=KeySignals(
                us_commercial_experience="Limited",
                project_scale_alignment="Medium",
                recent_negative_news=False
            ),
            confidence_score=0.3  # Low confidence since search failed
        )
    
    def search(self, company: str) -> CompanyInsight:
        """Search for company information and return structured insight."""
        if company in self.cache:
            return self.cache[company]
        
        try:
            insight = self._search_single(company)
            self.cache[company] = insight
            return insight
        except Exception:
            # Return default on any error
            return self._get_default_insight(company)
    
    def search_batch(self, companies: List[str]) -> Dict[str, CompanyInsight]:
        """Search multiple companies in parallel."""
        results = {}
        futures = {}
        
        for company in companies:
            if company in self.cache:
                results[company] = self.cache[company]
            else:
                futures[self._executor.submit(self._search_single, company)] = company
        
        # Wait for all futures with timeout
        try:
            for future in as_completed(futures, timeout=45):
                company = futures[future]
                try:
                    results[company] = future.result()
                    self.cache[company] = results[company]
                except Exception:
                    # Use default insight on failure
                    results[company] = self._get_default_insight(company)
        except FuturesTimeoutError:
            # Handle overall timeout - fill remaining with defaults
            for future, company in futures.items():
                if company not in results:
                    results[company] = self._get_default_insight(company)
        
        return results
    
    def _search_single(self, company: str) -> CompanyInsight:
        """Search a single company (both APIs in parallel)."""
        tavily_future = self._executor.submit(self._tavily_search, company)
        serp_future = self._executor.submit(self._serp_search, company)
        
        # Get results with timeout handling
        try:
            tavily_result = tavily_future.result(timeout=15)
        except Exception:
            tavily_result = {"snippets": [], "urls": [], "success": False}
        
        try:
            serp_result = serp_future.result(timeout=15)
        except Exception:
            serp_result = {"snippets": [], "urls": [], "success": False}
        
        return self._analyze_results(company, tavily_result, serp_result)
    
    def _tavily_search(self, company: str) -> Dict:
        """Search using Tavily API."""
        try:
            query = f"{company} construction company US projects commercial"
            res = self.tavily.search(query, max_results=5)
            results = res.get("results", [])
            return {
                "snippets": [r.get("content", "") for r in results],
                "urls": [r.get("url", "") for r in results],
                "success": True
            }
        except Exception:
            return {"snippets": [], "urls": [], "success": False}
    
    def _serp_search(self, company: str) -> Dict:
        """Search using SerpAPI."""
        if not SERPAPI_API_KEY:
            return {"snippets": [], "urls": [], "success": False}
        
        try:
            params = {
                "q": f"{company} construction company",
                "api_key": SERPAPI_API_KEY,
                "num": 5
            }
            res = requests.get("https://serpapi.com/search", params=params, timeout=10).json()
            organic = res.get("organic_results", [])
            return {
                "snippets": [r.get("snippet", "") for r in organic],
                "urls": [r.get("link", "") for r in organic],
                "success": True
            }
        except Exception:
            return {"snippets": [], "urls": [], "success": False}
    
    def _analyze_results(self, company: str, tavily: Dict, serp: Dict) -> CompanyInsight:
        """Analyze search results and extract key signals."""
        all_snippets = tavily.get("snippets", []) + serp.get("snippets", [])
        all_text = " ".join(all_snippets).lower()
        
        all_urls = tavily.get("urls", []) + serp.get("urls", [])
        sources = list(dict.fromkeys([u for u in all_urls if u]))[:5]
        
        us_experience = self._detect_us_experience(all_text)
        scale = self._detect_scale_alignment(all_text)
        negative_news = self._detect_negative_news(all_text)
        confidence = self._calculate_confidence(tavily, serp, len(all_snippets))
        
        return CompanyInsight(
            sources=sources,
            key_signals=KeySignals(
                us_commercial_experience=us_experience,
                project_scale_alignment=scale,
                recent_negative_news=negative_news
            ),
            confidence_score=round(confidence, 2)
        )
    
    def _detect_us_experience(self, text: str):
        """Detect US commercial experience from text."""
        us_indicators = [
            "united states", "usa", "u.s.", "american",
            "california", "new york", "texas", "florida",
            "los angeles", "chicago", "houston", "phoenix"
        ]
        limited_indicators = ["expanding to us", "entering us market", "limited us"]
        
        us_count = sum(1 for indicator in us_indicators if indicator in text)
        
        if any(ind in text for ind in limited_indicators):
            return "Limited"
        elif us_count >= 3:
            return True
        elif us_count >= 1:
            return "Limited"
        else:
            return False
    
    def _detect_scale_alignment(self, text: str) -> str:
        """Detect if company handles similar project scales."""
        large_indicators = ["billion", "major projects", "large scale", "mega project"]
        medium_indicators = ["million", "commercial", "renovation", "mid-size"]
        
        if any(ind in text for ind in large_indicators):
            return "High"
        elif any(ind in text for ind in medium_indicators):
            return "Medium"
        else:
            return "Low"
    
    def _detect_negative_news(self, text: str) -> bool:
        """Detect negative news signals."""
        negative_indicators = [
            "lawsuit", "litigation", "fraud", "scandal",
            "bankruptcy", "violation", "fine", "penalty",
            "investigation", "settlement"
        ]
        return any(ind in text for ind in negative_indicators)
    
    def _calculate_confidence(self, tavily: Dict, serp: Dict, snippet_count: int) -> float:
        """Calculate confidence score based on search quality."""
        base = 0.5
        
        if tavily.get("success"):
            base += 0.15
        if serp.get("success"):
            base += 0.15
        
        if snippet_count >= 5:
            base += 0.15
        elif snippet_count >= 3:
            base += 0.10
        elif snippet_count >= 1:
            base += 0.05
        
        return min(base, 0.95)
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache = {}
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
