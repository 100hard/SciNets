import httpx
from typing import List, Dict, Any

OPENALEX_API_URL = "https://api.openalex.org/works"

async def search_papers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for papers on OpenAlex.
    """
    params = {
        "search": query,
        "per-page": limit,
        "sort": "relevance_score:desc"
    }
    
    async with httpx.AsyncClient() as client:
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
                    "host_venue": item.get("host_venue", {}).get("display_name"),
                    "cited_by_count": item.get("cited_by_count"),
                    "landing_page_url": item.get("landing_page_url")
                }
                results.append(paper)
            return results
        except httpx.HTTPError as e:
            print(f"Error fetching from OpenAlex: {e}")
            return []

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
