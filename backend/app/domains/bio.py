from typing import List, Dict, Any
from app.domains.types import DomainPack

class BioDomain(DomainPack):
    @property
    def name(self) -> str:
        return "bio"
    
    @property
    def description(self) -> str:
        return "Biology, Genetics, and Bioinformatics"

    def get_hypothesis_prompt(self, context: str) -> str:
        return f"""
        You are an expert biologist. Based on the following literature summary, generate novel hypotheses.
        Focus on molecular mechanisms, genetic interactions, or therapeutic targets.
        
        Literature Context:
        {context}
        """

    def get_experiment_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "bio_sequence_analysis",
                "name": "Sequence Analysis",
                "description": "Analyze DNA/Protein sequences for motifs or alignment."
            }
        ]
