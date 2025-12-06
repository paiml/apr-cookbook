# K-Means Clustering

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Implement k-means clustering with APR model storage.

## Run Command

```bash
cargo run --example create_apr_kmeans_clustering
```

## Code

```rust,ignore
{{#include ../../../../examples/creation/create_apr_kmeans_clustering.rs}}
```

## Key Concepts

1. **Centroids**: Cluster centers stored as [k, dims] tensor
2. **Assignment**: Nearest centroid based on Euclidean distance
3. **Convergence**: Iterative refinement until centroids stabilize
