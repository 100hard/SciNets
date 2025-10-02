from uuid import uuid4

from app.models.graph import (
    GraphEdge,
    GraphEdgeData,
    GraphEvidenceItem,
    GraphMeta,
    GraphNode,
    GraphNodeData,
    GraphNodeLink,
    GraphResponse,
    NodeType,
    RelationType,
)


def test_graph_response_supports_extended_node_types() -> None:
    concept_id = uuid4()
    material_id = uuid4()

    response = GraphResponse(
        nodes=[
            GraphNode(
                data=GraphNodeData(
                    id=f"{NodeType.CONCEPT}:{concept_id}",
                    type=NodeType.CONCEPT,
                    label="Photosynthesis",
                    entity_id=concept_id,
                    paper_count=2,
                    aliases=["photo synthesis"],
                    description="Biological concept",
                    top_links=[
                        GraphNodeLink(
                            id="link-1",
                            label="Chlorophyll",
                            type=NodeType.MATERIAL,
                            relation=RelationType.PROPOSES,
                            weight=0.7,
                        )
                    ],
                    evidence=[
                        GraphEvidenceItem(
                            paper_id=uuid4(),
                            paper_title="Energy Conversion",
                            snippet="Demonstrates chemical process.",
                            confidence=0.9,
                            relation=RelationType.REPORTS,
                        )
                    ],
                    metadata={"domain": "biology"},
                )
            )
        ],
        edges=[
            GraphEdge(
                data=GraphEdgeData(
                    id="edge-1",
                    source=f"{NodeType.MATERIAL}:{material_id}",
                    target=f"{NodeType.CONCEPT}:{concept_id}",
                    type=RelationType.COMPARES,
                    weight=0.4,
                    paper_count=1,
                    average_confidence=0.88,
                    metadata={"note": "cross-domain"},
                )
            )
        ],
        meta=GraphMeta(
            limit=25,
            node_count=1,
            edge_count=1,
            concept_count=1,
            paper_count=2,
            center_id=f"{NodeType.CONCEPT}:{concept_id}",
            center_type=NodeType.CONCEPT,
            filters={"types": ["concept", "material"], "relations": ["compares"]},
        ),
    )

    payload = response.model_dump()
    assert payload["nodes"][0]["data"]["type"] == "concept"
    assert payload["nodes"][0]["data"]["top_links"][0]["type"] == "material"
    assert payload["edges"][0]["data"]["type"] == "compares"
    assert payload["meta"]["center_type"] == "concept"
