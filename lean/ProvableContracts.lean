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

-- Architecture demos (PMAT-300..320): 18 family-smoke + 6 cross-family
-- meta-recipes. All proofs are real (rfl/structural arithmetic) since
-- the underlying recipes are pure functions of their inputs.
import ProvableContracts.ArchitectureDemos.Llama
import ProvableContracts.ArchitectureDemos.Mistral
import ProvableContracts.ArchitectureDemos.Qwen2
import ProvableContracts.ArchitectureDemos.Qwen3
import ProvableContracts.ArchitectureDemos.Qwen3_5
import ProvableContracts.ArchitectureDemos.Phi
import ProvableContracts.ArchitectureDemos.Gemma
import ProvableContracts.ArchitectureDemos.Gpt2
import ProvableContracts.ArchitectureDemos.GptNeox
import ProvableContracts.ArchitectureDemos.Deepseek
import ProvableContracts.ArchitectureDemos.FalconH1
import ProvableContracts.ArchitectureDemos.Rwkv7
import ProvableContracts.ArchitectureDemos.Openelm
import ProvableContracts.ArchitectureDemos.Opt
import ProvableContracts.ArchitectureDemos.Mamba
import ProvableContracts.ArchitectureDemos.Bert
import ProvableContracts.ArchitectureDemos.Moonshine
import ProvableContracts.ArchitectureDemos.Whisper
import ProvableContracts.ArchitectureDemos.ArchDetector
import ProvableContracts.ArchitectureDemos.ArchSummary
import ProvableContracts.ArchitectureDemos.ArchCompare
import ProvableContracts.ArchitectureDemos.ArchQuirkAudit
import ProvableContracts.ArchitectureDemos.ArchAliasResolver
import ProvableContracts.ArchitectureDemos.ArchResolutionPipeline
