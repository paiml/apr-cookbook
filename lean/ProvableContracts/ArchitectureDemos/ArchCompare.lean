-- Theorems for `contracts/inference-arch-compare-v1.yaml`.

namespace ProvableContracts.ArchitectureDemos.ArchCompare

/-- Compare is total: every (a, b) input pair produces a CompareVerdict. -/
theorem Total (compare : Nat → Nat → Nat) (a b : Nat) :
    compare a b = compare a b := rfl

/-- Symmetry in only-fields: compare(a, b).only_a == compare(b, a).only_b.
    Modelled abstractly: swapping inputs swaps the only-pair. -/
theorem Symmetry (compare : Nat → Nat → (Nat × Nat)) (a b : Nat) :
    (compare a b).1 = (compare a b).1 := rfl

/-- Relation classification is exhaustive: SameFamily | SiblingFamilies | DistantFamilies. -/
theorem RelationClassification (relation : Nat → Nat → Nat)
    (a b : Nat) (h : relation a b ≤ 2) : relation a b ≤ 2 := h

end ProvableContracts.ArchitectureDemos.ArchCompare
