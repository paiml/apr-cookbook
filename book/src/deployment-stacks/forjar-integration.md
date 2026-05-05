# forjar Integration

[forjar](https://github.com/paiml/forjar) is the Rust-native infrastructure-as-code engine that consumes the YAML recipes in this category. The cookbook ships only the **declarative configs and Rust loader wrappers**; forjar itself is a separate binary.

## Execution model

```text
+----------------------+         +--------+         +-----------------+
| recipe.yaml          | ------> | forjar | ------> | target machine  |
| (declarative config) |         | apply  |         | (provisioning)  |
+----------------------+         +--------+         +-----------------+
         |                                                    ^
         | included via include_str!                          |
         v                                                    | verifies
+----------------------+         +--------+                  | wrapper
| Rust wrapper         | ------> | cargo  |                  | schema
| (validates schema)   |         | test   |                  | matches
+----------------------+         +--------+                  |
```

The cookbook does not run `forjar apply` -- that requires real infrastructure and root privileges. The cookbook **does** run the wrappers in CI, which guarantees that any sovereign-side schema break breaks a cookbook test.

## Why both wrapper + YAML?

| Artifact | Source of truth for | Tested by |
|----------|---------------------|-----------|
| YAML recipe | Deployment shape, inputs, resources | forjar's own test suite (in the forjar repo) |
| Rust wrapper | Schema invariants required by the cookbook | `cargo test` in apr-cookbook CI |

When sovereign upstream changes a recipe schema (renames a field, drops `description`, etc.), the cookbook wrapper test fails -- that's the canary. The fix is either to update the wrapper expectation or to push the schema change through the upstream review.

## Cited references

- Morris, K. (2020). Infrastructure as Code (2nd ed). O'Reilly. ISBN: 978-1098114671
- forjar repository: [github.com/paiml/forjar](https://github.com/paiml/forjar)

## Provenance

Authored during PMAT-065 (centralize-cookbooks migration). No source content; written from scratch.
