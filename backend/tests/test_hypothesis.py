"""
Tests for the hypothesis agent.

These tests verify the hypothesis generation logic and graph exploration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.state import DiscoveryState, Hypothesis
from app.agents.hypothesis import hypothesis_node


@pytest.fixture
def mock_state():
    """Create a mock DiscoveryState for testing."""
    return DiscoveryState(
        user_query="What causes Alzheimer's disease?",
        goal="discover",
        lens="physics",
        speculation="high",
        run_experiments=True,
        literature={
            "summary": "Amyloid plaques and tau tangles are associated with Alzheimer's.",
            "papers": []
        },
        concept_graph={
            "nodes": ["Alzheimer's", "Amyloid", "Tau", "Neurodegeneration"],
            "edges": [
                {"source": "Amyloid", "target": "Alzheimer's", "relation": "causes"},
                {"source": "Tau", "target": "Alzheimer's", "relation": "causes"},
                {"source": "Neurodegeneration", "target": "Alzheimer's", "relation": "results_in"}
            ]
        }
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hypothesis_node_generates_hypotheses(mock_state):
    """Test that hypothesis_node generates hypotheses."""
    # Mock the LLM to return structured output
    with patch("app.agents.hypothesis.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=Mock(
                hypotheses=[
                    Hypothesis(
                        id="h1",
                        text="Hypothesis 1: Amyloid accumulation triggers tau pathology",
                        domain_tags=["bio"],
                        novelty_score=0.8,
                        feasibility_score=0.7,
                        testability_score=0.9
                    )
                ]
            )
        )
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Bridge analysis complete"))
        mock_get_llm.return_value = mock_llm
        
        result = await hypothesis_node(mock_state)
        
        assert "hypotheses" in result
        assert len(result["hypotheses"]) > 0
        assert result["hypotheses"][0].text is not None
        assert "selected_hypothesis_id" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hypothesis_node_handles_empty_graph():
    """Test that hypothesis_node handles empty concept graph gracefully."""
    state = DiscoveryState(
        user_query="Test query",
        goal="discover",
        literature={"summary": "Test summary"},
        concept_graph={"nodes": [], "edges": []}
    )
    
    with patch("app.agents.hypothesis.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=Mock(
                hypotheses=[
                    Hypothesis(
                        id="h1",
                        text="Fallback hypothesis",
                        domain_tags=["bio"],
                        novelty_score=0.5,
                        feasibility_score=0.5,
                        testability_score=0.5
                    )
                ]
            )
        )
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="No graph available"))
        mock_get_llm.return_value = mock_llm
        
        result = await hypothesis_node(state)
        
        # Should still return hypotheses even with empty graph
        assert "hypotheses" in result
        assert len(result["hypotheses"]) > 0


@pytest.mark.unit
def test_hypothesis_pydantic_validation():
    """Test that Hypothesis model validates correctly."""
    # Valid hypothesis
    h = Hypothesis(
        id="test-id",
        text="This is a test hypothesis",
        domain_tags=["bio", "chem"],
        novelty_score=0.75,
        feasibility_score=0.80,
        testability_score=0.90
    )
    
    assert h.id == "test-id"
    assert h.novelty_score == 0.75
    
    # Test Pydantic serialization (this was causing issues!)
    h_dict = h.model_dump()
    assert isinstance(h_dict, dict)
    assert h_dict["text"] == "This is a test hypothesis"
    
    # Test JSON serialization
    import json
    h_json = json.dumps(h_dict)
    assert isinstance(h_json, str)
