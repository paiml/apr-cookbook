-- Theorems for `contracts/apr-format-roundtrip-v1.yaml`.
--
-- APR format round-trip properties: tensor bytes, shape, count, and
-- metadata all survive serialize → deserialize. Stated as universal
-- properties over any pure inverse pair.

namespace ProvableContracts.AprFormat.Roundtrip

/-- Lossless tensor round-trip: given an inverse law, tensor bytes
    survive the round-trip exactly. -/
theorem LosslessTensorRoundtrip
    (serialize : List UInt8 → List UInt8)
    (deserialize : List UInt8 → List UInt8)
    (inverse : ∀ t, deserialize (serialize t) = t)
    (tensor : List UInt8) :
    deserialize (serialize tensor) = tensor :=
  inverse tensor

/-- Shape preservation: tensor shape (as `List Nat`) survives the
    round-trip given an inverse law on the shape projection. -/
theorem ShapePreservation
    (getShape : List UInt8 → List Nat)
    (serialize : List UInt8 → List UInt8)
    (deserialize : List UInt8 → List UInt8)
    (inverse : ∀ t, deserialize (serialize t) = t)
    (tensor : List UInt8) :
    getShape (deserialize (serialize tensor)) = getShape tensor := by
  rw [inverse]

/-- Tensor count preservation: the number of tensors in the bundle
    survives the round-trip. -/
theorem CountPreservation
    (getCount : List UInt8 → Nat)
    (serialize : List UInt8 → List UInt8)
    (deserialize : List UInt8 → List UInt8)
    (inverse : ∀ t, deserialize (serialize t) = t)
    (tensor : List UInt8) :
    getCount (deserialize (serialize tensor)) = getCount tensor := by
  rw [inverse]

/-- Metadata preservation: version tag, compression, quantization fields
    survive the round-trip. -/
theorem MetadataPreservation
    (getMeta : List UInt8 → List (String × String))
    (serialize : List UInt8 → List UInt8)
    (deserialize : List UInt8 → List UInt8)
    (inverse : ∀ t, deserialize (serialize t) = t)
    (tensor : List UInt8) :
    getMeta (deserialize (serialize tensor)) = getMeta tensor := by
  rw [inverse]

end ProvableContracts.AprFormat.Roundtrip
