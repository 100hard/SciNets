"""
Benchmark Queries for Experiment 1 Pilot.
5 queries spanning 3 domains, vetted from SciNets benchmark suite.
"""

PILOT_QUERIES = [
    {
        "id": "ML-1",
        "domain": "ml",
        "query": "How does loss landscape geometry influence generalization in overparameterized neural networks?",
        "search_terms": '"loss landscape" AND generalization AND "neural networks"',
        "description": "Connects loss surface curvature / flat minima to generalization bounds."
    },
    {
        "id": "ML-2",
        "domain": "ml",
        "query": "What mechanistic links connect implicit bias of stochastic gradient descent and flat minima in deep learning models?",
        "search_terms": '"implicit bias" AND "stochastic gradient descent" AND "flat minima"',
        "description": "Connects algorithmic SGD noise to loss surface minimum selection."
    },
    {
        "id": "BIO-1",
        "domain": "bio",
        "query": "How does sleep deprivation affect synaptic plasticity and memory consolidation in the hippocampus?",
        "search_terms": '"sleep deprivation" AND "synaptic plasticity" AND hippocampus',
        "description": "Connects sleep loss, intracellular signaling (cAMP/PKA/mTOR), and dendritic spine remodeling."
    },
    {
        "id": "BIO-3",
        "domain": "bio",
        "query": "What mechanistic pathways link gut microbiome dysbiosis to cognitive decline and mood regulation abnormalities?",
        "search_terms": '"gut microbiome" AND dysbiosis AND (cognition OR depression OR mood)',
        "description": "Connects microbial metabolites, vagus nerve signaling, and neuroinflammation."
    },
    {
        "id": "CLIM-1",
        "domain": "climate",
        "query": "What causal pathways connect deforestation, nutrient runoff, and coral reef degradation?",
        "search_terms": '"coral reef" AND (runoff OR deforestation)',
        "description": "Connects watershed land use, sediment/nutrient transport, and marine calcification/bleaching."
    }
]
