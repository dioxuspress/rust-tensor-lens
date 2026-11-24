# Rust Tensor Lens

Zero-backend **Pure-Rust + Dioxus 0.7 WASM** interactive laboratory for LLM math, quantization, tokenization, and GPU memory budgeting.

- **Live Demo**: `https://dioxuspress.github.io/rust-tensor-lens/`
- **Deep-Dive Case Study**: `https://rustineverything.app/case/rust-tensor-lens`

## Interactive Labs (100% Client-Side Rust/WASM)

1. **RoPE & YaRN Extrapolation Lab (`src/math/rope.rs`)**
   - Compares **Standard RoPE**, **Position Interpolation (PI)**, **NTK-Aware**, and **YaRN** across relative-distance attention inner product decay `S(Δm)`, per-pair log wavelength spectrum `log₁₀(λ_i)`, YaRN attention temperature `√(1/t)`, and 2D complex-plane phasor rotation.
2. **INT8 / INT4 / NF4 / BitNet 1.58-bit Quantization Lab (`src/math/quant.rs`)**
   - Deterministic `8x8` Gaussian weight tiles with controllable outlier spikes (`1σ..16σ`) and block group sizes (`64 / 32 / 16 / 8`), rendering per-element error heatmaps `|w - ŵ|`, normalized quantization staircases, and live SQNR / RMSE / Cosine Similarity comparisons (using exact QLoRA 16-level NF4 codebook and BitNet `absmean` ternary quantization).
3. **Byte-Level BPE Tokenizer & Merge Stepper (`src/math/bpe.rs`)**
   - Real greedy lowest-rank Byte-Level BPE encoder starting from raw UTF-8 bytes (`0..255`), complete with an interactive merge-step scrubber to compare Rust code, Chinese AI terminology, and JSON token efficiency.
4. **LLM VRAM & KV-Cache Budget Calculator (`src/math/vram.rs`)**
   - Models `Llama-3.1 8B`, `Qwen2.5 7B/32B`, `Llama-3.3 70B`, `Gemma-3 27B`, and `DeepSeek-V3 (671B MoE · MLA)` across **Inference**, **LoRA / QLoRA**, and **Full SFT (AdamW)**, including FP16/FP8/INT4 KV-Cache, Gradient Checkpointing, `1K -> 128K` scaling curves, and roofline decode throughput estimates across `RTX 4090 / 5090 / L40S / A100 / H100 / M4 Max`.

## Local Development & Verification

```bash
# 1. Run pure-Rust numerical unit tests
cargo test

# 2. Check formatting, Clippy (host + wasm32)
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --target wasm32-unknown-unknown -- -D warnings

# 3. Launch live WASM dev server
dx serve --platform web

# 4. Build optimized static bundle
dx build --release --platform web
```

## Architecture Principles

- **Zero backend / zero telemetry**: every formula executes locally in compiled WebAssembly.
- **Zero `'unsafe-eval'` CSP compatible**: no `document::eval`, no runtime JS template evaluation, zero `node_modules` build dependency.
- **Pure-Rust math core**: all numerical algorithms live in `src/math/` with host-executable unit tests, decoupled from Dioxus UI rendering in `src/labs/`.

## License

MIT — part of the [DioxusPress](https://github.com/dioxuspress) & [Rust in Everything](https://rustineverything.app) ecosystem.
