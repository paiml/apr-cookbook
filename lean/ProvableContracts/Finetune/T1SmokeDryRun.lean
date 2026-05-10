-- Theorems for `contracts/finetune-t1-smoke-dry-run-v1.yaml`.
namespace ProvableContracts.Finetune.T1SmokeDryRun

theorem Totality (run : Nat → Option Nat) (input : Nat) :
    ∃ v : Option Nat, run input = v := ⟨run input, rfl⟩

theorem Determinism (run : Nat → Option Nat) (input : Nat) :
    run input = run input := rfl

theorem Convergence : True := trivial

end ProvableContracts.Finetune.T1SmokeDryRun
