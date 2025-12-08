"""Basic tests to verify pytest setup and imports."""

import pytest
from app.state import DiscoveryState, Hypothesis, Experiment


@pytest.mark.unit
def test_imports():
    """Verify all critical imports work."""
    from app.graph import create_graph
    from app.llm import get_llm
    from app.state import DiscoveryState
    
    assert create_graph is not None
    assert get_llm is not None
    assert DiscoveryState is not None


@pytest.mark.unit
def test_discovery_state_creation():
    """Test DiscoveryState can be created."""
    state = DiscoveryState(
        user_query="Test query",
        goal="discover",
        lens="none",
        speculation="medium",
        run_experiments=True
    )
    
    assert state.user_query == "Test query"
    assert state.goal == "discover"
    assert state.run_experiments is True


@pytest.mark.unit
def test_hypothesis_serialization():
    """Test that Hypothesis serializes to JSON correctly (prevents Pydantic issues)."""
    import json
    
    h = Hypothesis(
        id="test-123",
        text="Test hypothesis",
        domain_tags=["bio"],
        novelty_score=0.8,
        feasibility_score=0.7,
        testability_score=0.9
    )
    
    # This should not raise an error
    h_dict = h.model_dump()
    assert isinstance(h_dict, dict)
    
    # Should be JSON serializable
    json_str = json.dumps(h_dict)
    assert isinstance(json_str, str)
    
    # Should deserialize back
    h_reloaded = Hypothesis(**json.loads(json_str))
    assert h_reloaded.id == h.id
    assert h_reloaded.text == h.text
