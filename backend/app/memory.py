import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.state import Insight
import uuid

class MemoryManager:
    def __init__(self):
        # Default to localhost for host-based execution (Windows/Mac)
        self._available = False
        
        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            print("[Memory] Vector Database disabled (optimization mode).")
            self.client = None
            return

        try:
            self.client = QdrantClient(url=qdrant_url)
            self.collection_name = "scinets_insights"
            self._init_collection()
            self._available = True
        except Exception as e:
            print(f"[Memory] Warning: Failed to initialize Qdrant ({e}). Memory features will be disabled.")
            self.client = None

    def is_available(self) -> bool:
        """Check if memory storage is available."""
        return self._available and self.client is not None

    def _init_collection(self):
        if not self.client: return
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            print(f"[Memory] Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=models.Distance.COSINE
                )
            )

    def store_insight(self, insight: Insight):
        """Stores an insight in Qdrant."""
        if not self.is_available():
            print("[Memory] Skipping store - Qdrant unavailable")
            return
            
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        
        vector = embeddings.embed_query(insight.content)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "content": insight.content,
                        "domain": insight.domain,
                        "confidence": insight.confidence,
                        "source": insight.source  # Track insight origin
                    }
                )
            ]
        )
        print(f"[Memory] Stored insight: {insight.content[:50]}...")

    def retrieve_insights(self, query: str, limit: int = 3) -> list[Insight]:
        """Retrieves relevant insights for a query."""
        if not self.is_available():
            return []
            
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        
        vector = embeddings.embed_query(query)
        
        try:
            try:
                # Try newer query_points API first (v1.7+)
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit
                ).points
            except AttributeError:
                 # Fallback to older search API
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=limit
                )
            
            insights = []
            for res in results:
                if res.score > 0.7: # Threshold
                    insights.append(Insight(
                        content=res.payload["content"],
                        domain=res.payload["domain"],
                        confidence=res.payload["confidence"],
                        source=res.payload.get("source")  # Include source if available
                    ))
            
            print(f"[Memory] Retrieved {len(insights)} insights.")
            return insights
        except Exception as e:
            print(f"[Memory] Search failed: {e}")
            return []
