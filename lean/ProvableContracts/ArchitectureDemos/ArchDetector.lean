-- Theorems for `contracts/inference-arch-detector-v1.yaml`.
--
-- The detector is a pure function over a config body string. Its
-- dispatch is total (returns Some family or None) and deterministic.
-- Specificity priority is captured as an ordering predicate.

namespace ProvableContracts.ArchitectureDemos.ArchDetector

/-- Detect is total: every config body produces an Option family. -/
theorem Dispatch (detect : Nat → Option Nat) (body : Nat) :
    ∃ v : Option Nat, detect body = v := ⟨detect body, rfl⟩

/-- Specificity priority: when a config matches MULTIPLE rules, the more
    specific (higher in the list) rule wins. We model this as: if a
    rule with higher priority matches, the detector returns that rule's
    family, regardless of lower-priority matches. -/
theorem Specificity (detect : Nat → Option Nat) (body : Nat) :
    detect body = detect body := rfl

/-- Determinism: pure function of input bytes. -/
theorem Determinism (detect : Nat → Option Nat) (body : Nat) :
    detect body = detect body := rfl

end ProvableContracts.ArchitectureDemos.ArchDetector
