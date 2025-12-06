# Linear Regression Model

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Create a linear regression model with weight and bias tensors.

## Run Command

```bash
cargo run --example create_apr_linear_regression
```

## Code

```rust,ignore
{{#include ../../../../examples/creation/create_apr_linear_regression.rs}}
```

## Key Concepts

1. **Weight Matrix**: Shape [input_dim, output_dim]
2. **Bias Vector**: Shape [output_dim]
3. **Prediction**: `y = Wx + b`

## Mathematical Background

Linear regression finds the best-fit line through data points by minimizing the mean squared error between predictions and actual values.
