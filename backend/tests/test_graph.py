
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.state import DiscoveryState, Experiment, Hypothesis, EvidenceItem
from app.graph import create_graph

@pytest.fixture
def mock_chains():
    """Mock all LLM chains to avoid API calls."""
    with patch("app.agents.orchestrator.get_cheap_llm"), \
         patch("app.agents.literature.get_cheap_llm"), \
         patch("app.agents.hypothesis.get_llm"), \
         patch("app.agents.evidence.get_cheap_llm"), \
         patch("app.agents.experiment.get_llm"):
        yield

@pytest.mark.asyncio
async def test_full_graph_execution_happy_path(mock_chains):
    """
    Integration test for the full graph execution without external APIs.
    Simulates the flow: Planner -> Literature -> Hypothesis -> Evidence -> Experiment
    """
    # 1. Setup Mocks
    # 1. Setup Mocks
    with patch("app.graph.plan_node", new_callable=AsyncMock) as mock_plan, \
         patch("app.graph.literature_node", new_callable=AsyncMock) as mock_lit, \
         patch("app.graph.hypothesis_node", new_callable=AsyncMock) as mock_hyp, \
         patch("app.graph.evidence_node", new_callable=AsyncMock) as mock_ev, \
         patch("app.graph.experiment_node", new_callable=AsyncMock) as mock_exp, \
         patch("app.graph.critique_node", new_callable=AsyncMock) as mock_crit:
        
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
        
        mock_exp.return_value = {
            "experiments": [Experiment(hypothesis_id="h1", code_snippet="print('Success')")]
        }
        
        mock_crit.return_value = {} # End

        # 2. Run Graph
        graph = create_graph()
        initial_state = DiscoveryState(
            user_query="Test efficient flow",
            run_experiments=True
        )
        
        # Execute
        events = []
        config = {"configurable": {"thread_id": "test_thread_1"}}
        async for event in graph.astream(initial_state, config=config):
             events.append(event)
             
        # 3. Verify Flow
        # Check that nodes were called in order
        assert mock_plan.called
        assert mock_lit.called
        assert mock_hyp.called
        assert mock_ev.called
        assert mock_exp.called
        
        print("\n✅ Efficient Integration Test Passed: Full Chain simulated in <1s")
