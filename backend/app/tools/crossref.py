
import httpx
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re

from app.config import config

CROSSREF_API_URL = "https://api.crossref.org/works"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.ReadTimeout)
)
async def search_papers_crossref(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search papers on Crossref.
    Docs: https://github.com/CrossRef/rest-api-doc
    """
    if not query:
        return []

    if len(query) > 300:
        query = query[:300]

    params = {
        "query": query,
        "rows": limit,
        # Select specific fields to keep payload small
        "select": "DOI,title,abstract,created,publisher,URL,is-referenced-by-count,container-title",
        "sort": "relevance"
    }

    # POLITE POOL: Sending email puts us in the faster/reliable pool
    headers = {
        "User-Agent": f"SciNets/2.0 (mailto:{config.CONTACT_EMAIL})"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(CROSSREF_API_URL, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            results = []
            for item in items:
                # 1. Title (list or string)
                title = item.get("title")
                if isinstance(title, list) and title:
                    title = title[0]
                elif not title:
                    title = "Untitled"
                
                # 2. Year (Parse from 'created')
                # 'created': {'date-parts': [[2023, 5, 12]], ...}
                pub_year = None
                try:
                    pub_year = item.get("created", {}).get("date-parts", [[None]])[0][0]
                except:
                    pass

                # 3. Abstract (Often contains XML tags like <jats:p>)
                abstract_raw = item.get("abstract", "")
                abstract_clean = clean_crossref_abstract(abstract_raw)

                # 4. Venue
                venue = item.get("container-title")
                if isinstance(venue, list) and venue:
                    venue = venue[0]
                
                # Map to SciNets schema
                results.append({
                    "id": item.get("URL") or f"https://doi.org/{item.get('DOI')}",
                    "title": title,
                    "publication_year": pub_year,
                    "abstract": abstract_clean, 
                    "host_venue": venue,
                    "cited_by_count": item.get("is-referenced-by-count", 0),
                    "landing_page_url": item.get("URL")
                })
            
            return results

        except Exception as e:
            # Let the facade handle logging/fallback
            print(f"[Crossref] Search failed: {e}")
            raise e

def clean_crossref_abstract(raw_xml: str) -> str:
    """
    Remove JATS/XML tags from Crossref abstracts.
    Example: <jats:p>This is the abstract.</jats:p>
    """
    if not raw_xml:
        return ""
    
    # Remove all XML/HTML tags
    clean = re.sub(r'<[^>]+>', '', raw_xml)
    # Unescape entities if needed (basic ones)
    clean = clean.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return clean.strip()
