import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.state import Insight
import uuid

class MemoryManager:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://qdrant:6333")
        )
        self.collection_name = "scinets_insights"
        self._init_collection()

    def _init_collection(self):
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
                        "confidence": insight.confidence
                    }
                )
            ]
        )
        print(f"[Memory] Stored insight: {insight.content[:50]}...")

    def retrieve_insights(self, query: str, limit: int = 3) -> list[Insight]:
        """Retrieves relevant insights for a query."""
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        
        vector = embeddings.embed_query(query)
        
        try:
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
                        confidence=res.payload["confidence"]
                    ))
            
            print(f"[Memory] Retrieved {len(insights)} insights.")
            return insights
        except Exception as e:
            print(f"[Memory] Search failed: {e}")
            return []
