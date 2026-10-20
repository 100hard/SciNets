import pytest
from unittest.mock import AsyncMock, patch
from app.state import DiscoveryState, Hypothesis, EvidenceItem
from app.graph import create_graph

@pytest.fixture
def mock_chains():
    """Mock all LLM chains to avoid API calls."""
    with patch("app.agents.orchestrator.get_cheap_llm"), \
         patch("app.agents.literature.get_cheap_llm"), \
         patch("app.agents.hypothesis.get_llm"), \
         patch("app.agents.evidence.get_cheap_llm"), \
         patch("app.agents.critique.get_cheap_llm"), \
         patch("app.agents.decision.get_llm"):
        yield

@pytest.mark.asyncio
async def test_full_graph_execution_happy_path(mock_chains):
    """
    Integration test for the full discovery graph execution without external APIs.
    Simulates the flow: Planner -> Literature -> Hypothesis Preview (interrupt) -> Deep -> Evidence -> Critique -> Decision
    """
    with patch("app.graph.plan_node", new_callable=AsyncMock) as mock_plan, \
         patch("app.graph.literature_node", new_callable=AsyncMock) as mock_lit, \
         patch("app.graph.hypothesis_node", new_callable=AsyncMock) as mock_hyp, \
         patch("app.graph.evidence_node", new_callable=AsyncMock) as mock_ev, \
         patch("app.graph.critique_node", new_callable=AsyncMock) as mock_crit, \
         patch("app.graph.decision_node", new_callable=AsyncMock) as mock_dec:
        
        # Configure Mock Returns
        mock_plan.return_value = {"domain_tags": ["physics"], "plan": {"steps": ["all"]}}
        mock_lit.return_value = {"literature": {"summary": "Papers found"}}
        
        mock_hyp.return_value = {
            "hypotheses": [
                Hypothesis(
                    id="h1", 
                    text="Test Hypo", 
                    domain_tags=["physics"],
                    novelty_score=0.8,
                    feasibility_score=0.9,
                    testability_score=0.7
                )
            ],
            "selected_hypothesis_id": "h1"
        }
        
        mock_ev.return_value = {
            "hypotheses": [
                Hypothesis(
                    id="h1", 
                    text="Test Hypo", 
                    domain_tags=["physics"],
                    novelty_score=0.8,
                    feasibility_score=0.9,
                    testability_score=0.7,
                    evidence=[EvidenceItem(paper_id="p1", title="Evidence Paper", stance="support", strength=5, key_points=[])]
                )
            ]
        }
        
        mock_crit.return_value = {"critique": {"summary": "Solid reasoning"}}
        mock_dec.return_value = {"done": True}

        # Run Graph
        graph = create_graph()
        initial_state = DiscoveryState(
            user_query="Test efficient flow"
        )
        
        # Execute Phase 1 up to interrupt (hypothesis_preview)
        config = {"configurable": {"thread_id": "test_thread_1"}}
        events = []
        async for event in graph.astream(initial_state, config=config):
            events.append(event)
             
        assert mock_plan.called
        assert mock_lit.called
        assert mock_hyp.called

        # Resume after interrupt (hypothesis_deep -> evidence -> critique -> decision)
        async for event in graph.astream(None, config=config):
            events.append(event)

        assert mock_ev.called
        assert mock_crit.called
        assert mock_dec.called
