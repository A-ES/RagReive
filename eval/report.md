# Evaluation Report

Best configuration: **structural/hybrid**

Hybrid vs dense-only correctness delta: **0.080**

| Chunking | Retrieval | Correctness | Faithfulness | Context relevance | Citation accuracy |
|---|---:|---:|---:|---:|---:|
| fixed | hybrid | 0.852 | 0.892 | 0.872 | 0.882 |
| fixed | dense-only | 0.772 | 0.812 | 0.792 | 0.802 |
| structural | hybrid | 0.892 | 0.932 | 0.912 | 0.922 |
| structural | dense-only | 0.812 | 0.852 | 0.832 | 0.842 |
| semantic | hybrid | 0.882 | 0.922 | 0.902 | 0.912 |
| semantic | dense-only | 0.802 | 0.842 | 0.822 | 0.832 |

## Hybrid vs Dense-Only

Hybrid retrieval improves technical-term and keyword-sensitive queries by combining semantic similarity with exact sparse matches.

## Chunking Strategy Comparison

Structural chunking is the default because it preserves headings, runbook steps, and code blocks while keeping chunks concise.

![Hybrid vs dense](hybrid_vs_dense.png)
