# Compliance Audit

Full compliance audit pipeline for model deployment approval, composing five stages: inspect, oracle, qualify, QA, and report. Produces a structured audit report with pass/fail gates for governance sign-off.

## CLI Equivalent
```bash
N/A (composes apr inspect + apr oracle + apr qualify + apr qa)
```

## Key Concepts
- Five-stage compliance pipeline for deployment approval
- Governance gates: metadata, oracle scoring, qualification tiers
- Structured audit report generation

## Run
```bash
cargo run --example compliance_audit
```

## Source
[`examples/advanced/compliance_audit.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/advanced/compliance_audit.rs)
