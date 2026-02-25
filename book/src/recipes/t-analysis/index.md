# Category T: Analysis

Model analysis recipes mirroring the `apr` CLI analysis subcommands. These examples demonstrate inspection, validation, diffing, benchmarking, profiling, QA gates, oracle identification, canary testing, tree visualization, hex forensics, and error explanation.

## Inspection and Validation

| Recipe | Example | CLI Equivalent | Description |
|--------|---------|----------------|-------------|
| [Inspect](./inspect.md) | `analysis_inspect` | `apr inspect` | Model metadata and tensor listing |
| [Validate](./validate.md) | `analysis_validate` | `apr validate` | 100-point integrity validation |
| [Diff](./diff.md) | `analysis_diff` | `apr diff` | Weight-level model comparison |

## Performance

| Recipe | Example | CLI Equivalent | Description |
|--------|---------|----------------|-------------|
| [Bench](./bench.md) | `analysis_bench` | `apr bench` | Throughput benchmarking across batch sizes |
| [Profile](./profile.md) | `analysis_profile` | `apr profile` | Roofline model profiling |

## Quality Assurance

| Recipe | Example | CLI Equivalent | Description |
|--------|---------|----------------|-------------|
| [QA Gates](./qa-gates.md) | `analysis_qa_gates` | `apr qa` | 6-gate falsifiable QA for CI/CD |
| [Oracle](./oracle.md) | `analysis_oracle` | `apr oracle` | Model family identification |
| [Canary](./canary.md) | `analysis_canary` | `apr canary` | Canary regression testing |

## Forensics and Diagnostics

| Recipe | Example | CLI Equivalent | Description |
|--------|---------|----------------|-------------|
| [Tree](./tree.md) | `analysis_tree` | `apr tree` | Architecture visualization as ASCII tree |
| [Hex](./hex.md) | `analysis_hex` | `apr hex` | Format-aware binary forensics |
| [Explain](./explain.md) | `analysis_explain` | `apr explain` | Error code explanation system |
