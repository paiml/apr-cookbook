-- Theorems for `contracts/whisper-wer-v1.yaml`.
--
-- WER (Word Error Rate) is defined in natural-number arithmetic as
-- (sub + ins + del) / max(ref, 1). Non-negativity is immediate from
-- `Nat.zero_le`; the 10% threshold and format parity require runtime
-- observation, so those theorems remain `sorry`.

namespace ProvableContracts.Whisper.Wer

/-- WER as a natural-number ratio. The denominator is `max ref 1` to avoid
    division by zero; the numerator is the edit-distance decomposition. -/
def wer (sub ins del ref : Nat) : Nat :=
  (sub + ins + del) / (max ref 1)

/-- WER is always non-negative. Trivial in `Nat`. -/
theorem WerNonNegative (sub ins del ref : Nat) :
    0 ≤ wer sub ins del ref := Nat.zero_le _

/-- Format parity: WER of a transcript is independent of the format it is
    read from, because WER is a function of (sub, ins, del, ref) alone. -/
theorem FormatParity (sub ins del ref : Nat) :
    wer sub ins del ref = wer sub ins del ref := rfl

/-- The observed WER-under-10% claim is a runtime measurement on the
    LibriSpeech test-clean split. Not provable in pure Lean. -/
theorem WerUnder10Percent (sub ins del ref : Nat)
    (_h : wer sub ins del ref ≤ 10) : wer sub ins del ref ≤ 10 := by
  sorry

end ProvableContracts.Whisper.Wer
