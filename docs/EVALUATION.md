# SciNets Evaluation Notes

SciNets has been tested primarily as a graph-backed scientific ideation system.

## Summary

The strongest behavior appears when the system:

- retrieves a manageable number of relevant papers
- builds a coherent concept graph
- reasons over shorter and clearer connection paths

## Observed Strengths

- Reliable graph construction from literature inputs
- Useful discovery of bridge concepts across disconnected areas
- Better performance on shorter reasoning chains

## Observed Limits

- Longer reasoning paths are harder for the generation layer to explain faithfully
- Higher novelty can trade off against grounding and clarity
- Complex synthesis quality depends heavily on model capacity

## Practical Takeaway

SciNets is most credible as a research exploration and ideation tool:

- strong for search, structuring, and surfacing hypotheses
- best when users keep a human-in-the-loop for validation
- more reliable on concise, explainable scientific chains than on long speculative ones
