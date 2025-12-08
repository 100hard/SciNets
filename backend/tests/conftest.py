"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared test fixtures
- Pytest hooks
- Test configuration
"""

import pytest
from typing import Generator
import asyncio


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a mock LLM response",
        "role": "assistant"
    }


@pytest.fixture
def sample_concept_graph():
    """Sample concept graph for testing."""
    return {
        "nodes": [
            "Concept A",
            "Concept B", 
            "Concept C",
            "Concept D"
        ],
        "edges": [
            {"source": "Concept A", "target": "Concept B", "relation": "causes"},
            {"source": "Concept B", "target": "Concept C", "relation": "leads_to"},
            {"source": "Concept C", "target": "Concept D", "relation": "inhibits"}
        ]
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
