from app.domains.bio import BioDomain
from app.domains.ml import MLDomain
from app.domains.types import DomainPack
from typing import Dict

def get_domain_packs() -> Dict[str, DomainPack]:
    """
    Returns a dictionary of available domain packs.
    """
    return {
        "bio": BioDomain(),
        "ml": MLDomain()
    }
