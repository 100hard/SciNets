from typing import Protocol, List, Dict, Any
from pydantic import BaseModel

class DomainPack(Protocol):
    @property
    def name(self) -> str:
        """Name of the domain (e.g., 'bio', 'ml')."""
        ...
    
    @property
    def description(self) -> str:
        """Short description for the orchestrator."""
        ...

    def get_hypothesis_prompt(self, context: str) -> str:
        """Returns a prompt for generating hypotheses in this domain."""
        ...

    def get_experiment_templates(self) -> List[Dict[str, Any]]:
        """Returns available experiment templates."""
        ...
