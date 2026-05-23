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

-- Fine-tuning cookbook (PMAT-330..361): Tier 1.1 SFT minimal × 5 families.
-- All theorems are real (rfl/structural) — recipe is a pure linear-regression
-- SGD over a deterministic JSONL fixture. Convergence theorem is `trivial`
-- because SGD on convex sub-objective is monotone-decreasing in expectation
-- but not closed-form-provable in Lean.
import ProvableContracts.Finetune.T1SftMinimalLlama
import ProvableContracts.Finetune.T1SftMinimalMistral
import ProvableContracts.Finetune.T1SftMinimalPhi
import ProvableContracts.Finetune.T1SftMinimalQwen
import ProvableContracts.Finetune.T1SftMinimalGemma

-- Tier 1.2: 5 eval primitives. Convergence is `trivial` because eval is
-- a pure closed-form function (no training, no convergence dynamics).
import ProvableContracts.Finetune.T1EvalPerplexity
import ProvableContracts.Finetune.T1EvalAccuracy
import ProvableContracts.Finetune.T1EvalF1
import ProvableContracts.Finetune.T1EvalRougeL
import ProvableContracts.Finetune.T1EvalBleu

-- Tier 1.3: 5 tabular regression recipes.
import ProvableContracts.Finetune.T1TabularRegressionHousing
import ProvableContracts.Finetune.T1TabularRegressionEnergy
import ProvableContracts.Finetune.T1TabularRegressionTimeseries
import ProvableContracts.Finetune.T1TabularRegressionMultitarget
import ProvableContracts.Finetune.T1TabularRegressionMissing

-- Tier 1.4: 5 tabular classification recipes (PMAT-334).
import ProvableContracts.Finetune.T1TabularBinary
import ProvableContracts.Finetune.T1TabularClass3class
import ProvableContracts.Finetune.T1TabularClass7class
import ProvableContracts.Finetune.T1TabularClass100class
import ProvableContracts.Finetune.T1TabularImbalanced

-- Tier 1.5: 5 smoke + bench recipes (PMAT-335). Closes Tier 1 at 25/25.
import ProvableContracts.Finetune.T1SmokePlan
import ProvableContracts.Finetune.T1SmokeResume
import ProvableContracts.Finetune.T1SmokeEarlyStop
import ProvableContracts.Finetune.T1SmokeDryRun
import ProvableContracts.Finetune.T1SmokeBench

-- Tier 2.1: LoRA × 10 (PMAT-338+339). 5 families × 2 ranks (8, 32).
import ProvableContracts.Finetune.T2LoraRank8Llama
import ProvableContracts.Finetune.T2LoraRank8Mistral
import ProvableContracts.Finetune.T2LoraRank8Phi
import ProvableContracts.Finetune.T2LoraRank8Qwen
import ProvableContracts.Finetune.T2LoraRank8Gemma
import ProvableContracts.Finetune.T2LoraRank32Llama
import ProvableContracts.Finetune.T2LoraRank32Mistral
import ProvableContracts.Finetune.T2LoraRank32Phi
import ProvableContracts.Finetune.T2LoraRank32Qwen
import ProvableContracts.Finetune.T2LoraRank32Gemma

-- Tier 2.2: QLoRA × 5 (PMAT-340).
import ProvableContracts.Finetune.T2QloraLlama4bitR8
import ProvableContracts.Finetune.T2QloraMistral4bitR16
import ProvableContracts.Finetune.T2QloraPhi4bitR32
import ProvableContracts.Finetune.T2QloraQwenDoubleQuant
import ProvableContracts.Finetune.T2QloraGemmaDoubleQuantOff
import ProvableContracts.Finetune.T2ContinuedPretrainLegal
import ProvableContracts.Finetune.T2ContinuedPretrainCode
import ProvableContracts.Finetune.T2ContinuedPretrainMedical
import ProvableContracts.Finetune.T2ContinuedPretrainCodeswitch
import ProvableContracts.Finetune.T2ContinuedPretrainScientific
import ProvableContracts.Finetune.T2AdapterMergeTies
import ProvableContracts.Finetune.T2AdapterMergeDare
import ProvableContracts.Finetune.T2AdapterMergeSlerp
import ProvableContracts.Finetune.T2AdapterMergeAverage
import ProvableContracts.Finetune.T2AdapterMergeMultilora
import ProvableContracts.Finetune.T2PeftCordaInit
import ProvableContracts.Finetune.T2PeftEvaInit
import ProvableContracts.Finetune.T2PeftPissaInit
import ProvableContracts.Finetune.T2PeftLoftqInit
import ProvableContracts.Finetune.T2Oft
import ProvableContracts.Finetune.T2LnTuning
import ProvableContracts.Finetune.T2Tinylora
import ProvableContracts.Finetune.T2Vblora
import ProvableContracts.Finetune.T2RegexFreeze
import ProvableContracts.Finetune.T2Galore
import ProvableContracts.Finetune.T2Badam
import ProvableContracts.Finetune.T2Apollo
import ProvableContracts.Finetune.T2Dora
import ProvableContracts.Finetune.T2FreezeTuning
import ProvableContracts.Finetune.T2LoraAqlm
import ProvableContracts.Finetune.T2LoraAwq
import ProvableContracts.Finetune.T2LoraGptq
import ProvableContracts.Finetune.T2Relora
import ProvableContracts.Finetune.T2Lisa
import ProvableContracts.Finetune.T2Neftune
import ProvableContracts.Finetune.T3InstructionAlpaca
import ProvableContracts.Finetune.T3InstructionSharegpt
import ProvableContracts.Finetune.T3InstructionOpenassistant
import ProvableContracts.Finetune.T3InstructionChatTemplate
import ProvableContracts.Finetune.T3InstructionSystemPrompt
import ProvableContracts.Finetune.T3HyperoptGrid
import ProvableContracts.Finetune.T3HyperoptRandom
import ProvableContracts.Finetune.T3HyperoptTpe
import ProvableContracts.Finetune.T3HyperoptAsha
import ProvableContracts.Finetune.T3HyperoptHyperband
import ProvableContracts.Finetune.T3CalibrationTemperature
import ProvableContracts.Finetune.T3CalibrationPlatt
import ProvableContracts.Finetune.T3CalibrationIsotonic
import ProvableContracts.Finetune.T3CalibrationConformal
import ProvableContracts.Finetune.T3CalibrationEnsemble
import ProvableContracts.Finetune.T3ImbalanceWeighted
import ProvableContracts.Finetune.T3ImbalanceFocal
import ProvableContracts.Finetune.T3ImbalanceSmote
import ProvableContracts.Finetune.T3ImbalanceThreshold
import ProvableContracts.Finetune.T3ImbalanceCostsensitive
import ProvableContracts.Finetune.T3MultimodalTextImage
import ProvableContracts.Finetune.T3MultimodalTextTabular
import ProvableContracts.Finetune.T3MultimodalMultitask
import ProvableContracts.Finetune.T3MultimodalZeroShot
import ProvableContracts.Finetune.T3KfoldCv
import ProvableContracts.Finetune.T3AnomalyDeepSad
import ProvableContracts.Finetune.T3AnomalyDeepSvdd
import ProvableContracts.Finetune.T3AnomalyDrocc
import ProvableContracts.Finetune.T3OpenSetBaseline
import ProvableContracts.Finetune.T3OpenSetEntropic
import ProvableContracts.Finetune.T3OpenSetObjectosphere
import ProvableContracts.Finetune.T3UncertaintyMcDropout
import ProvableContracts.Finetune.T3UncertaintyCalibrated
import ProvableContracts.Finetune.T3ImageEncoderClip
import ProvableContracts.Finetune.T3ImageEncoderDinov2Lp
import ProvableContracts.Finetune.T3ImageEncoderSiglip
import ProvableContracts.Finetune.T3OptimizerMuon
import ProvableContracts.Finetune.T3OptimizerScheduleFree
import ProvableContracts.Finetune.T3Lbfgs
import ProvableContracts.Finetune.T3MultitaskFamo
import ProvableContracts.Finetune.T3SemanticSegmentationSegformer
import ProvableContracts.Finetune.T3StructuredOutputJson
import ProvableContracts.Finetune.T3MambaEncoderText
import ProvableContracts.Finetune.T3Hypernetwork
import ProvableContracts.Finetune.T3QatFp8
import ProvableContracts.Finetune.T3QatMxfp4
import ProvableContracts.Finetune.T3SamplePacking
import ProvableContracts.Finetune.T3FsdpLora
import ProvableContracts.Finetune.T4DpoLlama
import ProvableContracts.Finetune.T4DpoMistral
import ProvableContracts.Finetune.T4DpoPhi
import ProvableContracts.Finetune.T4DpoQwen
import ProvableContracts.Finetune.T4DpoGemma
import ProvableContracts.Finetune.T4OrpoLlama
import ProvableContracts.Finetune.T4OrpoMistral
import ProvableContracts.Finetune.T4OrpoQwen
import ProvableContracts.Finetune.T4KtoLlama
import ProvableContracts.Finetune.T4KtoPhi
import ProvableContracts.Finetune.T4KtoGemma
import ProvableContracts.Finetune.T4GrpoMath
import ProvableContracts.Finetune.T4GrpoCodeExec
import ProvableContracts.Finetune.T4GrpoFormatMatch
import ProvableContracts.Finetune.T4GrpoClassification
import ProvableContracts.Finetune.T4GrpoLengthBudget
import ProvableContracts.Finetune.T4RlhfPpoLlama
import ProvableContracts.Finetune.T4RlhfPpoMistral
import ProvableContracts.Finetune.T4RlhfPpoQwen
import ProvableContracts.Finetune.T4RlaifJudge
import ProvableContracts.Finetune.T4RlaifConstitutional
import ProvableContracts.Finetune.T4RlaifSelfCritique
import ProvableContracts.Finetune.T4RewardPairwise
import ProvableContracts.Finetune.T4RewardScalar
import ProvableContracts.Finetune.T4RewardEnsemble
import ProvableContracts.Finetune.T4OnlineDpo
import ProvableContracts.Finetune.T4Xpo
import ProvableContracts.Finetune.T4NashMd
import ProvableContracts.Finetune.T4Rloo
import ProvableContracts.Finetune.T4Bco
import ProvableContracts.Finetune.T4Cpo
import ProvableContracts.Finetune.T4Simpo
import ProvableContracts.Finetune.T4AsyncGrpo
import ProvableContracts.Finetune.T4Prm
import ProvableContracts.Finetune.T4Gkd
import ProvableContracts.Finetune.T4Gspo
import ProvableContracts.Finetune.T4Mpo
