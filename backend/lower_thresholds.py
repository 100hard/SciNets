#!/usr/bin/env python3

# Read the original file
with open('app/services/canonicalization.py', 'r') as f:
    content = f.read()

# Replace the thresholds with lower values
old_thresholds = """_SIMILARITY_THRESHOLDS: Dict[ConceptResolutionType, float] = {
    ConceptResolutionType.METHOD: 0.85,
    ConceptResolutionType.DATASET: 0.82,
    ConceptResolutionType.METRIC: 0.90,
    ConceptResolutionType.TASK: 0.80,
}"""

new_thresholds = """_SIMILARITY_THRESHOLDS: Dict[ConceptResolutionType, float] = {
    ConceptResolutionType.METHOD: 0.70,
    ConceptResolutionType.DATASET: 0.65,
    ConceptResolutionType.METRIC: 0.75,
    ConceptResolutionType.TASK: 0.65,
}"""

# Replace the thresholds
new_content = content.replace(old_thresholds, new_thresholds)

# Write the modified content back
with open('app/services/canonicalization.py', 'w') as f:
    f.write(new_content)

print("Successfully lowered similarity thresholds")
