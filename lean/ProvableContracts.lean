-- Root module for ProvableContracts.
-- Each submodule corresponds to one contract YAML under `contracts/`.
-- The lean.status field in each contract tracks whether the theorem body
-- is `:= by sorry` (status: sorry) or a real proof (status: proved).

import ProvableContracts.Recipe.Iiur
import ProvableContracts.Whisper.Wer
import ProvableContracts.Avx512.Matmul
import ProvableContracts.Mmap.Inference
import ProvableContracts.Aes256.GcmDecrypt
import ProvableContracts.FlashAttention
import ProvableContracts.Cli.Parity
import ProvableContracts.Lz4.Decompression
import ProvableContracts.Int4.Quantization
import ProvableContracts.Docs.Schema
import ProvableContracts.AprFormat.Roundtrip
