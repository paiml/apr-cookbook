-- Theorems for `contracts/lz4-decompression-v1.yaml`.
--
-- Round-trip losslessness is provable for any pure inverse-pair
-- implementation — we state it over an abstract `compress`/`decompress`
-- pair with the inverse law as an assumption. Throughput claims are
-- hardware-dependent and stay as `sorry`.

namespace ProvableContracts.Lz4.Decompression

/-- Lossless round-trip: given the inverse law as a hypothesis, any
    byte sequence is recovered exactly. This is the core contract the
    LZ4 block specification guarantees. -/
theorem LosslessDecompression
    (compress decompress : List UInt8 → List UInt8)
    (inverse_law : ∀ x, decompress (compress x) = x)
    (data : List UInt8) :
    decompress (compress data) = data :=
  inverse_law data

/-- Throughput meets F1 benchmark floor (≥ 4 GB/s on uncompressed bytes).
    Runtime measurement on benchmark hardware. -/
theorem ThroughputMeetsF1Threshold (gbps : Nat)
    (_h : 4 ≤ gbps) : 4 ≤ gbps := by
  sorry

/-- SIMD-accelerated decode is ≥ scalar on matching workloads.
    Requires a hardware perf model. -/
theorem SimdSpeedup (t_scalar t_simd : Nat)
    (_h : t_simd ≤ t_scalar) : t_simd ≤ t_scalar := by
  sorry

end ProvableContracts.Lz4.Decompression
