
from typing import List, Dict, Any
import asyncio

# Import individual providers
# from app.tools.semantic_scholar import search_papers_s2 (Deprecated)
from app.tools.crossref import search_papers_crossref
from app.tools.openalex import search_papers as search_papers_openalex
from app.tools.openalex import reconstruct_abstract, get_paper_citations, get_paper_details

async def search_papers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Unified Search Paper function.
    Strategy: 
      1. Try Crossref (Primary - Stable Public API)
      2. If Crossref fails, Fallback to OpenAlex (OA)
    """
    print(f"[SearchFacade] Searching for: '{query[:50]}...'")
    
    # 1. Try Crossref
    try:
        results = await search_papers_crossref(query, limit=limit)
        
        # QUALITY CHECK: Crossref often returns metadata WITHOUT abstracts.
        # Strictness increased: We need a decent number of readable papers.
        # Threshold: At least 3 valid abstracts OR > 20% of result set.
        valid_abstracts = [r for r in results if r.get('abstract') and len(r.get('abstract')) > 100]
        
        # If we asked for 10 papers and got 1, that's bad UX. Force fallback.
        is_high_quality = len(valid_abstracts) >= 3 or (len(results) > 0 and len(valid_abstracts) / len(results) > 0.20)

        if results and is_high_quality:
            print(f"[SearchFacade] Crossref returned {len(results)} results ({len(valid_abstracts)} with abstracts). ACCEPTED.")
            return results
        else:
            print(f"[SearchFacade] Crossref result quality too low ({len(valid_abstracts)} valid abstracts out of {len(results)}). Falling back to OpenAlex...")
            
    except Exception as e:
        print(f"[SearchFacade] Crossref failed: {e}. Falling back to OpenAlex...")

    # 2. Fallback to OpenAlex
    try:
        return await search_papers_openalex(query, limit=limit)
    except Exception as e:
        print(f"[SearchFacade] OpenAlex also failed: {e}")
        return []

# Re-export helper functions from OpenAlex for compatibility
# (Eventually we should make these provider-agnostic too)
__all__ = ["search_papers", "reconstruct_abstract", "get_paper_citations", "get_paper_details"]
