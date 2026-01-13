import asyncio

async def test():
    from app.tools.openalex import search_papers as openalex_search
    from app.main import CandidatePaper
    
    papers = await openalex_search("caffeine memory", limit=2)
    print(f"Found {len(papers)} papers")
    
    for paper in papers:
        print("---")
        title = paper.get("title")
        year = paper.get("publication_year")
        venue = paper.get("host_venue")
        abstract_data = paper.get("abstract")
        print(f"Title: {title}")
        print(f"Year: {year}")
        print(f"Venue: {venue}")
        print(f"Abstract is dict: {isinstance(abstract_data, dict)}")
        
        # Try creating CandidatePaper
        try:
            inverted_index = abstract_data
            abstract = ""
            if inverted_index and isinstance(inverted_index, dict):
                word_positions = []
                for word, positions in inverted_index.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)
            
            venue_str = venue if venue else "Unknown venue"
            
            c = CandidatePaper(
                id=paper.get("id", "test-id"),
                title=title or "Untitled",
                year=year or 2024,
                venue=venue_str,
                abstract=abstract[:500] if abstract else "No abstract available",
                rationale="Test"
            )
            print(f"CandidatePaper created successfully: {c.title[:50]}")
        except Exception as e:
            print(f"Error creating CandidatePaper: {e}")

asyncio.run(test())
