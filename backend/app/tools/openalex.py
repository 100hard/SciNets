import httpx
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OPENALEX_API_URL = "https://api.openalex.org/works"

from app.config import config

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.ReadTimeout)
)
async def search_papers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for papers on OpenAlex with retry logic.
    """
    # GUARD: Block empty or garbage queries (Fix #2 from User)
    if not query or len(query.strip()) < 5:
        print(f"[OpenAlex] Aborting search for empty/short query: '{query}'")
        return []

    # GUARD: Simplify overly complex queries (Fix #4 from User - Reliability)
    # Recursion/Nesting in queries causes 500s from OpenAlex API
    # PATCH D: Aggressive simplification threshold (Lowered 350 -> 200) to catch heavy boolean queries
    if len(query) > 200 or query.count("(") > 4:
        print(f"[OpenAlex] Query too complex (len={len(query)}, nesting={query.count('(')}). Simplifying...")
        # Simplification: Strip special chars, keep words, join with AND
        import re
        # Remove operators and parens
        clean = re.sub(r'[()"\']', '', query)
        # Split and take distinct significant words (DEDUPLICATED via dict.fromkeys)
        # Fix: Previously "A OR B" became "A AND B" which is too restrictive if synonyms.
        # Ideally, we want the ORIGINAL query, but we don't have it here. 
        # Best effort: Take unique long words.
        all_words = [w for w in clean.split() if w.lower() not in ["and", "or", "not"] and len(w) > 3]
        
        # SEMANTIC PRESERVATION (Fix #3 from User - Quality)
        # Keep core topic anchors if present
        anchors = []
        # Common scientific/AI alignment terms to preserve
        for key in ["safety", "robust", "align", "multi", "agent", "reinforce", "learning", "alzheimer", "sleep", "cancer", "climate", "energy", "quantum", "carbon", "capture", "basalt"]:
            for w in all_words:
                if key in w.lower():
                    anchors.append(w)
        
        # Combine anchors + unique words, prioritizing anchors
        unique_words = list(dict.fromkeys(anchors + all_words))
        
        # Limit to top 4 keywords (Reduced from 7 to prevent 503 overloads)
        # Intersection of 4 broad terms is much faster than 7.
        query = " ".join(unique_words[:4]) 
        print(f"[OpenAlex] Simplified Query: {query}")

    # if config.DEMO_MODE: ... (Removed to allow real search)

    # PATCH A: ALWAYS use structured filter search
    # OpenAlex full-text 'search' param is proving unstable (503s).
    # 'abstract.search' is more performant and specific enough for our needs.
    
    # Use abstract search for content matching
    params = {
        "filter": f"abstract.search:{query},has_abstract:true",
        "per-page": limit,
        "sort": "relevance_score:desc"
    }
    
    headers = {
        "User-Agent": "SciNets/2.0 (mailto:scinets.auth@gmail.com)"
    }
    # Add API Key if present (Unlock higher limits / premium pool)
    if config.OPENALEX_API_KEY:
        headers["api_key"] = config.OPENALEX_API_KEY
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(OPENALEX_API_URL, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # PATCH B: Do NOT retry on 503 (Service Unavailable)
            if e.response.status_code == 503:
                print("[OpenAlex] Upstream overloaded (503). Skipping retries.")
                # Proceed to fallback below
            else:
                # For other errors (like 429), maybe retry or fall through
                print(f"[OpenAlex] HTTP Error {e.response.status_code}: {e}")
                
        except httpx.HTTPError as e:
             print(f"[OpenAlex] Primary query failed: {e}")
        # Proceed to fallback / result processing
        else:
             # Success block
             data = response.json()
             results = []
             for item in data.get("results", []):
                results.append({
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "publication_year": item.get("publication_year"),
                    "abstract": reconstruct_abstract(item.get("abstract_inverted_index")), 
                    "host_venue": (item.get("host_venue") or {}).get("display_name"),
                    "cited_by_count": item.get("cited_by_count"),
                    "landing_page_url": item.get("landing_page_url")
                })
             return results

        # PATCH C: Graceful Fallback
        # If we reached here, primary request failed or was 503.
        print("[OpenAlex] Attempting Graceful Fallback with ultra-simple keywords...")
        
        # Cool-down to let upstream recover if it was a 503
        import asyncio
        await asyncio.sleep(1.0)
        
        # Ultra-simple fallback: top 2 LONGEST keywords (Reduced from 3 to ensure hits)
        # We assume longer words are more content-heavy (Alzheimers vs does)
        clean_terms = query.replace("AND", "").replace("OR", "").replace("(", "").replace(")", "").split()
        # Filter common stopwords just in case
        stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "connect", "between", "with", "from", "into", "during", "including", "until", "against", "among", "throughout", "despite", "towards", "upon", "learning", "study", "using", "based", "approach", "method", "analysis", "review", "system"}
        significant_terms = [t for t in clean_terms if len(t) > 3 and t.lower() not in stopwords]
        
        # Sort by length descending to get "Alzheimer's", "deprivation"
        # Fix: Deduplicate significant terms first!
        unique_sig_terms = list(dict.fromkeys(significant_terms))
        smart_terms = sorted(unique_sig_terms, key=len, reverse=True)[:2]
        
        if not smart_terms:
             # If nothing left, take raw split
             smart_terms = list(dict.fromkeys(clean_terms))[:2]
             
        fallback_query = " ".join(smart_terms)

        print(f"[OpenAlex] Fallback Query: {fallback_query}")

        # Use ABSTRACT search for fallback (Title often too sparse for intersection)
        fallback_params = {
            "filter": f"abstract.search:{fallback_query},has_abstract:true",
            "per-page": limit,
            "sort": "relevance_score:desc"
        }
        
        try:
            response = await client.get(OPENALEX_API_URL, params=fallback_params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "publication_year": item.get("publication_year"),
                        "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
                        "host_venue": (item.get("host_venue") or {}).get("display_name"),
                        "cited_by_count": item.get("cited_by_count"),
                        "landing_page_url": item.get("landing_page_url")
                    })
                return results
        except Exception as e:
            print(f"[OpenAlex] Fallback also failed: {e}")
            
        # If everything fails, return empty list (don't crash the agent)
        return []

def reconstruct_abstract(inverted_index: Any) -> str:
    """
    Helper to reconstruct abstract.
    Handles:
      - String (already valid text from Crossref/S2)
      - Dict (OpenAlex Inverted Index)
      - None (Returns empty string)
    """
    if not inverted_index:
        return ""
        
    # FIX: If already a string, return as is
    if isinstance(inverted_index, str):
        return inverted_index
    
    # OpenAlex Inverted Index Case
    if isinstance(inverted_index, dict):
        try:
            word_index = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_index.append((pos, word))
            
            word_index.sort()
            return " ".join([word for _, word in word_index])
        except Exception as e:
            print(f"Error parsing inverted index: {e}")
            return ""

    return ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_paper_citations(paper_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch papers that cite the given paper (citation expansion).
    Uses OpenAlex's cites filter. Returns papers weighted by relevance.
    
    NOTE: This is OPTIONAL expansion - use sparingly to avoid bias toward dominant narratives.
    """
    # Extract OpenAlex ID from URL if needed
    if paper_id.startswith("https://"):
        paper_id = paper_id.split("/")[-1]
    
    params = {
        "filter": f"cites:{paper_id},has_abstract:true",
        "per-page": limit,
        "sort": "cited_by_count:desc"  # High-impact citations first
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(OPENALEX_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                paper = {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "publication_year": item.get("publication_year"),
                    "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
                    "host_venue": (item.get("host_venue") or {}).get("display_name"),
                    "cited_by_count": item.get("cited_by_count"),
                    "landing_page_url": item.get("landing_page_url"),
                    "source": "citation_expansion"  # Mark source for weighting
                }
                results.append(paper)
            return results
        except httpx.HTTPError as e:
            print(f"Error fetching citations from OpenAlex: {e}")
            raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_paper_details(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch details for a single paper by OpenAlex ID.
    Used when processing user-selected papers (which are passed as URLs/IDs).
    """
    # Normalize ID (remove URL prefix if present)
    clean_id = paper_id
    if paper_id.startswith("https://openalex.org/"):
        clean_id = paper_id.split("/")[-1]
    
    url = f"https://api.openalex.org/works/{clean_id}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                print(f"[OpenAlex] Paper not found: {paper_id}")
                return None
            response.raise_for_status()
            item = response.json()
            
            return {
                "id": item.get("id"),
                "title": item.get("title"),
                "publication_year": item.get("publication_year"),
                "abstract": reconstruct_abstract(item.get("abstract_inverted_index")), 
                "host_venue": (item.get("host_venue") or {}).get("display_name"),
                "cited_by_count": item.get("cited_by_count"),
                "landing_page_url": item.get("landing_page_url")
            }
        except httpx.HTTPError as e:
            print(f"Error fetching details from OpenAlex for {paper_id}: {e}")
            raise
