# Contracts Macros — `aprender-contracts-macros` Compile-Time Enforcement

Recipes for [`aprender-contracts-macros`](https://crates.io/crates/aprender-contracts-macros) v0.31.2 — proc-macro half of the contract enforcement story (the runtime YAML validator is `aprender-contracts`). The `#[contract]` attribute reads `CONTRACT_<NAME>_<EQ>` env vars set by `build.rs` from `binding.yaml`; missing env vars degrade gracefully to a no-op.

Closes the **≥3 recipes per sister crate** requirement from [expand-cookbooks/subcrate-coverage.md](../../specifications/expand-cookbooks/subcrate-coverage.md).

## Recipes

| # | Recipe | What |
|---|--------|------|
| CM.1 | [`contracts_macros_attribute_basic`](https://github.com/paiml/apr-cookbook/blob/main/examples/contracts-macros/contracts_macros_attribute_basic.rs) | Apply `#[contract("name", equation = "eq")]` to two functions; macro degrades to no-op when env var absent |
| CM.2 | [`contracts_macros_env_key_convention`](https://github.com/paiml/apr-cookbook/blob/main/examples/contracts-macros/contracts_macros_env_key_convention.rs) | `CONTRACT_<NAME>_<EQ>` key derivation rules; mirrors the proc-macro's internal `make_env_key` cases as a drift detector |
| CM.3 | [`contracts_macros_runtime_validator_bridge`](https://github.com/paiml/apr-cookbook/blob/main/examples/contracts-macros/contracts_macros_runtime_validator_bridge.rs) | Load `contracts/recipe-iiur-v1.yaml` via `provable_contracts::parse_contract`, walk equations, derive macro env-keys for each |

## API surface exercised

- `provable_contracts_macros::contract` (proc-macro attribute)
- `provable_contracts::schema::parse_contract` (runtime YAML validator)

## Citations

- Meyer, B. (1992). Applying "Design by Contract". IEEE Computer 25(10). DOI: 10.1109/2.161279
- Findler, R. B. & Felleisen, M. (2002). Contracts for higher-order functions. ICFP. DOI: 10.1145/581478.581484

## Provenance

Added during PMAT-084 (expand-cookbooks initiative, v6.1.0).
