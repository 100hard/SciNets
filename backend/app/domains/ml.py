from typing import List, Dict, Any
from app.domains.types import DomainPack

class MLDomain(DomainPack):
    @property
    def name(self) -> str:
        return "ml"
    
    @property
    def description(self) -> str:
        return "Machine Learning, Deep Learning, and AI"

    def get_hypothesis_prompt(self, context: str) -> str:
        return f"""
        You are an AI researcher. Based on the following literature summary, generate novel hypotheses.
        Focus on model architecture, training techniques, or new applications.
        
        Literature Context:
        {context}
        """

    def get_experiment_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "ml_train_model",
                "name": "Train Model",
                "description": "Train a standard ML model (RF, SVM, MLP) on a dataset."
            }
        ]
