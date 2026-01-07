
# Failure Mode & Correlation Analysis

## Correlations
- **Correlation(Path Length, Drop Rate)**: -0.26
   - Interpretation: Weak (Does longer path imply more dropping?)

- **Correlation(Diversity, Drop Rate)**: -0.80
   - Interpretation: Weak (Does high diversity imply more dropping?)

## Failure Modes
Total Runs Analyzed: 5
Failure Distribution:
failure_mode
Hallucinated Bridge           4
Collapse (Generic Summary)    1

## Full Metrics Table (Top 5 Rows)
|    | query_id   | method   |   symbolic_path_length |   grounded_realized_path_length |   drop_rate_pct | failure_flag   | failure_mode               |   diversity_score |   papers_per_edge |
|---:|:-----------|:---------|-----------------------:|--------------------------------:|----------------:|:---------------|:---------------------------|------------------:|------------------:|
|  0 | ML-1       | full     |                      6 |                               6 |               0 | False          | Hallucinated Bridge        |          0.2      |                 0 |
|  1 | ML-1       | random   |                      5 |                               4 |              20 | True           | Hallucinated Bridge        |          0.195334 |                 0 |
|  2 | ML-2       | full     |                      4 |                               0 |             100 | True           | Collapse (Generic Summary) |          0.164051 |                 0 |
|  3 | ML-2       | random   |                      5 |                               5 |               0 | False          | Hallucinated Bridge        |          0.185592 |                 0 |
|  4 | ML-2       | shortest |                      3 |                               3 |               0 | False          | Hallucinated Bridge        |          0.219874 |                 0 |
    