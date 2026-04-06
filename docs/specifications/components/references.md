# Peer-Reviewed References

---

## Core Technical

[1] Matsakis, N. & Klock, F. (2014). *The Rust Programming Language*. ACM SIGPLAN Notices. DOI: 10.1145/2663171.2663188

[2] Jung, R. et al. (2017). *RustBelt: Securing the Foundations of the Rust Programming Language*. POPL 2017. DOI: 10.1145/3158154

[3] Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR 2018. arXiv:1712.05877

[4] Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020. DOI: 10.18653/v1/2020.emnlp-demos.6

[5] Gerganov, G. (2023). *GGUF: A Format for Large Language Model Weights*. llama.cpp Technical Report.

[6] Haas, A. et al. (2017). *Bringing the Web up to Speed with WebAssembly*. PLDI 2017. DOI: 10.1145/3062341.3062363

[7] McInnes, L. et al. (2018). *UMAP: Uniform Manifold Approximation and Projection*. arXiv:1802.03426

[8] Pichon-Pharabod, J. & Sewell, P. (2021). *WebAssembly SIMD: A Portable Performance Enhancement*. OOPSLA 2021.

---

## Compression & Performance

[9] Collet, Y. (2023). *LZ4 - Extremely Fast Compression*. GitHub.

[10] Collet, Y. & Kucherawy, M. (2021). *Zstandard Compression and the 'application/zstd' Media Type*. RFC 8878.

[11] Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*. NeurIPS 2022. arXiv:2205.14135

[12] Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023. arXiv:2210.17323

[13] Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS 2022. arXiv:2208.07339

[14] Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML 2023. (Whisper)

[15] Chen, T. et al. (2018). *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*. OSDI 2018.

---

## Toyota Way & Lean Manufacturing

[16] Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN: 978-0915299140

[17] Womack, J.P. & Jones, D.T. (1996). *Lean Thinking: Banish Waste and Create Wealth*. Simon & Schuster. ISBN: 978-0743249270

[18] Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS.

[19] Schwartz, R. et al. (2020). *Green AI*. Communications of the ACM, 63(12), 54-63.

[20] Deming, W. E. (1986). *Out of the Crisis*. MIT Press. ISBN: 978-0262541152

---

## ML Engineering

[21] Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE 2019. DOI: 10.1109/ICSE-SEIP.2019.00042

[22] Paleyes, A., Urma, R., & Lawrence, N.D. (2022). *Challenges in Deploying Machine Learning: A Survey of Case Studies*. ACM Computing Surveys. DOI: 10.1145/3533378

[23] Patterson, D. et al. (2022). *Carbon Emissions and Large Neural Network Training*. arXiv:2104.10350

[24] Myers, B. et al. (2023). *Declarative Machine Learning Systems*. Communications of the ACM, 66(3), 84-93.

[25] Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. ISBN: 978-1449373320

---

## Falsifiability & Scientific Method

[26] Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge. ISBN: 978-0415278447

---

## Security & Testing

[27] Shrimpton, T. & Terashima, R.S. (2020). *A Provable Security Analysis of TLS*. Journal of Cryptology, 33(2), 449-488.

[28] Anderson, R. (2020). *Security Engineering*. Wiley, 3rd Edition. ISBN: 978-1119642787

[29] Ohm, M. et al. (2020). *Backstabber's Knife Collection: A Review of Open Source Software Supply Chain Attacks*. DIMVA 2020.

[30] Lehmann, D. et al. (2020). *Everything Old is New Again: Binary Security of WebAssembly*. USENIX Security 2020.

[31] Wheeler, D. (2021). *Fully Countering Trusting Trust through Diverse Double-Compiling*. ACSAC.

[32] Claessen, K. & Hughes, J. (2000). *QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs*. ICFP 2000. DOI: 10.1145/351240.351266

---

## Parity & POC Repos

These companion repositories provide head-to-head benchmarks and proof-of-concept deployments referenced by cookbook recipes.

### Parity Benchmarks (APR vs. competitors)

[P1] paiml/qwen-coder-deploy — 5-runtime inference benchmark (Ollama, llama.cpp, vLLM, realizr, realizr-wgpu). [GitHub](https://github.com/paiml/qwen-coder-deploy)

[P2] paiml/candle-vs-apr — Candle vs realizr GGUF inference (realizr 1.63x faster). [GitHub](https://github.com/paiml/candle-vs-apr)

[P3] paiml/qwen-train-canary — Training throughput comparison across 5 runtimes. [GitHub](https://github.com/paiml/qwen-train-canary)

[P4] paiml/apr-leaderboard — HuggingFace leaderboard proving APR binary matches Python benchmarks (HumanEval, MBPP). [GitHub](https://github.com/paiml/apr-leaderboard)

### POC Deployments

[POC1] paiml/sovereign-ai-cookbook — Full sovereign stack: 17 Rust components, 10 deployment stacks. [GitHub](https://github.com/paiml/sovereign-ai-cookbook)

[POC2] paiml/whisper.apr — Production Whisper in pure Rust, WASM-first speech-to-text. [GitHub](https://github.com/paiml/whisper.apr)

[POC3] paiml/tiny-model-ground-truth — Token-identical greedy outputs across all apr subcommands and formats. [GitHub](https://github.com/paiml/tiny-model-ground-truth)

[POC4] paiml/apr-model-qa-playbook — Structured QA playbook for model validation with apr. [GitHub](https://github.com/paiml/apr-model-qa-playbook)

### Competing Runtimes (upstream references)

[CR1] ggerganov/llama.cpp — C++ LLM inference, GGUF format origin. [GitHub](https://github.com/ggerganov/llama.cpp)

[CR2] ollama/ollama — Local AI runtime. [GitHub](https://github.com/ollama/ollama)

[CR3] vllm-project/vllm — High-concurrency GPU serving. [GitHub](https://github.com/vllm-project/vllm)

[CR4] huggingface/candle — Minimalist Rust ML framework. [GitHub](https://github.com/huggingface/candle)

[CR5] Mozilla-Ocho/llamafile — Single-file LLM distribution. [GitHub](https://github.com/Mozilla-Ocho/llamafile)

[CR6] huggingface/text-generation-inference — HF inference server (maintenance mode). [GitHub](https://github.com/huggingface/text-generation-inference)

[CR7] huggingface/safetensors — Secure tensor format (complementary). [GitHub](https://github.com/huggingface/safetensors)
