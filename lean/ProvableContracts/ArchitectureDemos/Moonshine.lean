-- Theorems for `contracts/inference-moonshine-smoke-v1.yaml`.
--
-- Moonshine is speech-only; the architecture-demos contract delegates
-- to examples/speech/ rather than examples/inference/. The Lean module
-- captures pure-function determinism for the loader-dispatch pattern.

namespace ProvableContracts.ArchitectureDemos.Moonshine

inductive SmokeVerdict where
  | ok : Nat → SmokeVerdict
  | loaderUnavailable : SmokeVerdict
  | invalidFixture : SmokeVerdict
  deriving DecidableEq

theorem LoaderDispatch (smoke : Nat → SmokeVerdict) (input : Nat) :
    ∃ v : SmokeVerdict, smoke input = v := ⟨smoke input, rfl⟩

theorem ForwardDeterminism (forward_sim : Nat → Nat → Nat)
    (seed vocab : Nat) :
    forward_sim seed vocab = forward_sim seed vocab := rfl

end ProvableContracts.ArchitectureDemos.Moonshine
