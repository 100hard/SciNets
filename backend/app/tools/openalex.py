import httpx
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OPENALEX_API_URL = "https://api.openalex.org/works"

from app.config import config

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
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
    if len(query) > 350 or query.count("(") > 6:
        print(f"[OpenAlex] Query too complex (len={len(query)}, nesting={query.count('(')}). Simplifying...")
        # Simplification: Strip special chars, keep words, join with AND
        import re
        # Remove operators and parens
        clean = re.sub(r'[()"\']', '', query)
        # Split and take distinct significant words
        words = [w for w in clean.split() if w.lower() not in ["and", "or", "not"] and len(w) > 3]
        # Limit to top 8 keywords to stay safe
        query = " AND ".join(words[:8])
        print(f"[OpenAlex] Simplified Query: {query}")

    # if config.DEMO_MODE: ... (Removed to allow real search)

    params = {
        "search": query,
        "filter": "has_abstract:true",
        "per-page": limit,
        "sort": "relevance_score:desc"
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
                    "abstract": item.get("abstract_inverted_index"), # OpenAlex returns inverted index, need to reconstruct or fetch text
                    "host_venue": (item.get("host_venue") or {}).get("display_name"),
                    "cited_by_count": item.get("cited_by_count"),
                    "landing_page_url": item.get("landing_page_url")
                }
                results.append(paper)
            return results
        except httpx.HTTPError as e:
            print(f"Error fetching from OpenAlex (Query: {query[:20]}...): {e}")
            raise  # Re-raise to trigger retry

def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """
    Helper to reconstruct abstract from OpenAlex inverted index.
    """
    if not inverted_index:
        return ""
    
    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
    
    word_index.sort()
    return " ".join([word for _, word in word_index])


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
                    "abstract": item.get("abstract_inverted_index"),
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
                "abstract": item.get("abstract_inverted_index"), 
                "host_venue": (item.get("host_venue") or {}).get("display_name"),
                "cited_by_count": item.get("cited_by_count"),
                "landing_page_url": item.get("landing_page_url")
            }
        except httpx.HTTPError as e:
            print(f"Error fetching details from OpenAlex for {paper_id}: {e}")
            raise
