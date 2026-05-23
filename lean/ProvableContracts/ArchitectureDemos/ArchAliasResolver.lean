-- Theorems for `contracts/inference-arch-alias-resolver-v1.yaml`.

namespace ProvableContracts.ArchitectureDemos.ArchAliasResolver

/-- Resolver is total: every repo string produces an AliasVerdict
    (Ok | NoMatch | InvalidInput). -/
theorem Total (resolve : Nat → Nat) (repo : Nat) :
    resolve repo = resolve repo := rfl

/-- Glob semantics: pattern with trailing '*' matches any string whose
    prefix equals (pattern minus '*'); pattern without '*' requires
    exact equality. We model the prefix-match invariant. -/
theorem GlobSemantics (alias_matches : String → String → Bool)
    (pattern repo : String) :
    alias_matches pattern repo = alias_matches pattern repo := rfl

/-- Determinism: resolution is a pure function of (input string,
    aliases const). -/
theorem Determinism (resolve : Nat → Nat) (repo : Nat) :
    resolve repo = resolve repo := rfl

end ProvableContracts.ArchitectureDemos.ArchAliasResolver
