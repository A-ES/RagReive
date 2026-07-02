# Retrieval Notes

Dense retrieval embeds the query and searches Qdrant for cosine-similar chunks.

Sparse retrieval uses BM25 tokenization over chunk text and is useful for endpoint names, exact configuration keys, and acronyms.

Hybrid retrieval calls both retrievers, then merges results with Reciprocal Rank Fusion. Setting `dense_only` to `true` bypasses sparse retrieval and RRF.
