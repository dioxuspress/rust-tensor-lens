#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPrecision {
  Fp16,
  Fp8,
  Int4Nf4,
  BitNet158,
}

impl WeightPrecision {
  pub fn label(self) -> &'static str {
    match self {
      Self::Fp16 => "BF16 / FP16 (16-bit)",
      Self::Fp8 => "FP8 / INT8 (8-bit)",
      Self::Int4Nf4 => "INT4 / NF4 (~4.5-bit)",
      Self::BitNet158 => "BitNet 1.58-bit (~2-bit)",
    }
  }

  pub fn bytes_per_param(self) -> f64 {
    match self {
      Self::Fp16 => 2.0,
      Self::Fp8 => 1.0,
      Self::Int4Nf4 => 0.5625, // 4-bit weights + FP16 group scales
      Self::BitNet158 => 0.25, // packed ternary + FP16 block scales
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrecision {
  Fp16,
  Fp8,
  Int4,
}

impl KvPrecision {
  pub fn label(self) -> &'static str {
    match self {
      Self::Fp16 => "FP16 KV",
      Self::Fp8 => "FP8 KV",
      Self::Int4 => "INT4 KV",
    }
  }

  pub fn bytes_per_elem(self) -> f64 {
    match self {
      Self::Fp16 => 2.0,
      Self::Fp8 => 1.0,
      Self::Int4 => 0.55,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadMode {
  Inference,
  LoraFinetune,
  FullFinetune,
}

impl WorkloadMode {
  pub fn label(self) -> &'static str {
    match self {
      Self::Inference => "Inference",
      Self::LoraFinetune => "LoRA / QLoRA",
      Self::FullFinetune => "Full SFT (AdamW)",
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelPreset {
  pub id: &'static str,
  pub name: &'static str,
  pub total_params_b: f64,
  pub active_params_b: f64,
  pub num_layers: usize,
  pub hidden_size: usize,
  pub num_q_heads: usize,
  pub num_kv_heads: usize,
  pub head_dim: usize,
  /// Some(latent_dim + rope_dim) for DeepSeek MLA architectures; None for standard MHA/GQA.
  pub mla_kv_dim: Option<usize>,
}

pub const MODEL_PRESETS: &[ModelPreset] = &[
  ModelPreset {
    id: "llama3-8b",
    name: "Llama-3.1 8B",
    total_params_b: 8.03,
    active_params_b: 8.03,
    num_layers: 32,
    hidden_size: 4_096,
    num_q_heads: 32,
    num_kv_heads: 8,
    head_dim: 128,
    mla_kv_dim: None,
  },
  ModelPreset {
    id: "qwen25-7b",
    name: "Qwen2.5 7B",
    total_params_b: 7.62,
    active_params_b: 7.62,
    num_layers: 28,
    hidden_size: 3_584,
    num_q_heads: 28,
    num_kv_heads: 4,
    head_dim: 128,
    mla_kv_dim: None,
  },
  ModelPreset {
    id: "qwen25-32b",
    name: "Qwen2.5 32B",
    total_params_b: 32.76,
    active_params_b: 32.76,
    num_layers: 64,
    hidden_size: 5_120,
    num_q_heads: 40,
    num_kv_heads: 8,
    head_dim: 128,
    mla_kv_dim: None,
  },
  ModelPreset {
    id: "llama3-70b",
    name: "Llama-3.3 70B",
    total_params_b: 70.6,
    active_params_b: 70.6,
    num_layers: 80,
    hidden_size: 8_192,
    num_q_heads: 64,
    num_kv_heads: 8,
    head_dim: 128,
    mla_kv_dim: None,
  },
  ModelPreset {
    id: "deepseek-v3",
    name: "DeepSeek-V3 (671B MLA)",
    total_params_b: 671.0,
    active_params_b: 37.0,
    num_layers: 61,
    hidden_size: 7_168,
    num_q_heads: 128,
    num_kv_heads: 128,
    head_dim: 128,
    mla_kv_dim: Some(576), // 512 compressed latent + 64 decoupled RoPE
  },
  ModelPreset {
    id: "gemma3-27b",
    name: "Gemma-3 27B",
    total_params_b: 27.4,
    active_params_b: 27.4,
    num_layers: 62,
    hidden_size: 5_376,
    num_q_heads: 32,
    num_kv_heads: 16,
    head_dim: 128,
    mla_kv_dim: None,
  },
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardwareSpec {
  pub name: &'static str,
  pub vram_gb: f64,
  pub bandwidth_gbs: f64,
}

pub const HARDWARE_SPECS: &[HardwareSpec] = &[
  HardwareSpec {
    name: "RTX 4090",
    vram_gb: 24.0,
    bandwidth_gbs: 1_008.0,
  },
  HardwareSpec {
    name: "RTX 5090",
    vram_gb: 32.0,
    bandwidth_gbs: 1_792.0,
  },
  HardwareSpec {
    name: "L40S",
    vram_gb: 48.0,
    bandwidth_gbs: 864.0,
  },
  HardwareSpec {
    name: "A100 80G",
    vram_gb: 80.0,
    bandwidth_gbs: 2_039.0,
  },
  HardwareSpec {
    name: "H100 SXM",
    vram_gb: 80.0,
    bandwidth_gbs: 3_350.0,
  },
  HardwareSpec {
    name: "M4 Max 128G",
    vram_gb: 96.0, // default 75% wired GPU pool on macOS
    bandwidth_gbs: 546.0,
  },
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VramParams {
  pub preset_index: usize,
  pub mode: WorkloadMode,
  pub weight_prec: WeightPrecision,
  pub kv_prec: KvPrecision,
  pub batch_size: usize,
  pub seq_len: usize,
  pub lora_rank: usize,
  pub grad_checkpointing: bool,
}

impl Default for VramParams {
  fn default() -> Self {
    Self {
      preset_index: 0,
      mode: WorkloadMode::Inference,
      weight_prec: WeightPrecision::Int4Nf4,
      kv_prec: KvPrecision::Fp16,
      batch_size: 4,
      seq_len: 16_384,
      lora_rank: 32,
      grad_checkpointing: true,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardwareFit {
  pub spec: HardwareSpec,
  pub utilization_pct: f64,
  pub min_cards: usize,
  pub est_decode_toks_per_sec: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContextCurvePoint {
  pub seq_len: usize,
  pub total_gb: f64,
  pub kv_gb: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VramAnalysis {
  pub model: ModelPreset,
  pub weights_gb: f64,
  pub kv_cache_gb: f64,
  pub optimizer_and_grads_gb: f64,
  pub activations_gb: f64,
  pub runtime_overhead_gb: f64,
  pub total_gb: f64,
  pub kv_bytes_per_token: f64,
  pub hardware_fits: Vec<HardwareFit>,
  pub context_curve: Vec<ContextCurvePoint>,
}

const GIB: f64 = 1_073_741_824.0;

fn kv_bytes_per_token(model: ModelPreset, kv_prec: KvPrecision) -> f64 {
  let per_layer_elems = match model.mla_kv_dim {
    Some(mla_dim) => mla_dim as f64,
    None => 2.0 * (model.num_kv_heads * model.head_dim) as f64,
  };
  model.num_layers as f64 * per_layer_elems * kv_prec.bytes_per_elem()
}

fn compute_breakdown(params: VramParams, seq_len_override: usize) -> (f64, f64, f64, f64, f64) {
  let model = MODEL_PRESETS
    .get(params.preset_index)
    .copied()
    .unwrap_or(MODEL_PRESETS[0]);
  let total_params = model.total_params_b * 1e9;
  let batch = params.batch_size.max(1) as f64;
  let seq = seq_len_override.max(256) as f64;

  let weights_bytes = match params.mode {
    WorkloadMode::FullFinetune => total_params * 2.0, // BF16 model weights
    WorkloadMode::Inference | WorkloadMode::LoraFinetune => {
      total_params * params.weight_prec.bytes_per_param()
    }
  };

  let kv_bytes = match params.mode {
    WorkloadMode::Inference => kv_bytes_per_token(model, params.kv_prec) * batch * seq,
    WorkloadMode::LoraFinetune | WorkloadMode::FullFinetune => 0.0,
  };

  let opt_grad_bytes = match params.mode {
    WorkloadMode::Inference => 0.0,
    WorkloadMode::LoraFinetune => {
      // LoRA applied on Q, K, V, O + Gate, Up, Down (~7 linear projections per layer).
      let r = params.lora_rank.clamp(4, 256) as f64;
      let trainable_params = model.num_layers as f64 * 7.0 * 2.0 * model.hidden_size as f64 * r;
      // FP16 adapter weights + FP16 grads + FP32 AdamW (master + m + v = 12B) = 16B/trainable param.
      trainable_params * 16.0
    }
    WorkloadMode::FullFinetune => {
      // BF16 gradients (2B) + FP32 master/m/v AdamW states (12B) = 14B/param.
      total_params * 14.0
    }
  };

  let activations_bytes = match params.mode {
    WorkloadMode::Inference => {
      // Transient hidden workspace during prefill/decode chunking.
      batch * seq.min(4096.0) * model.hidden_size as f64 * 6.0
    }
    WorkloadMode::LoraFinetune | WorkloadMode::FullFinetune => {
      // FlashAttention-2 training activation footprint (without O(S^2) attention materialization).
      let layers = model.num_layers as f64;
      let base = batch * seq * model.hidden_size as f64 * 20.0;
      if params.grad_checkpointing {
        base * layers.sqrt()
      } else {
        base * layers
      }
    }
  };

  let runtime_overhead_gb = 0.65 + 0.015 * (weights_bytes / GIB);

  (
    weights_bytes / GIB,
    kv_bytes / GIB,
    opt_grad_bytes / GIB,
    activations_bytes / GIB,
    runtime_overhead_gb,
  )
}

pub fn analyze_vram(params: VramParams) -> VramAnalysis {
  let model = MODEL_PRESETS
    .get(params.preset_index)
    .copied()
    .unwrap_or(MODEL_PRESETS[0]);
  let (weights_gb, kv_cache_gb, optimizer_and_grads_gb, activations_gb, runtime_overhead_gb) =
    compute_breakdown(params, params.seq_len);
  let total_gb =
    weights_gb + kv_cache_gb + optimizer_and_grads_gb + activations_gb + runtime_overhead_gb;
  let per_tok = kv_bytes_per_token(model, params.kv_prec);

  // Active bytes read per decode step (memory-bandwidth bound roofline with ~72% achievable HBM efficiency).
  let active_weight_gb = (model.active_params_b * 1e9 * params.weight_prec.bytes_per_param()) / GIB;
  let step_read_gb = (active_weight_gb + kv_cache_gb).max(0.1);

  let hardware_fits = HARDWARE_SPECS
    .iter()
    .map(|&spec| {
      let utilization_pct = (total_gb / spec.vram_gb) * 100.0;
      let min_cards = (total_gb / (spec.vram_gb * 0.92)).ceil().max(1.0) as usize;
      let effective_bw = spec.bandwidth_gbs * 0.72 * min_cards as f64;
      let est_decode_toks_per_sec = if params.mode == WorkloadMode::Inference {
        (effective_bw / step_read_gb) * params.batch_size.max(1) as f64
      } else {
        0.0
      };
      HardwareFit {
        spec,
        utilization_pct,
        min_cards,
        est_decode_toks_per_sec,
      }
    })
    .collect();

  let curve_lengths = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072];
  let context_curve = curve_lengths
    .iter()
    .map(|&len| {
      let (w, kv, og, act, ovh) = compute_breakdown(params, len);
      ContextCurvePoint {
        seq_len: len,
        total_gb: w + kv + og + act + ovh,
        kv_gb: kv,
      }
    })
    .collect();

  VramAnalysis {
    model,
    weights_gb,
    kv_cache_gb,
    optimizer_and_grads_gb,
    activations_gb,
    runtime_overhead_gb,
    total_gb,
    kv_bytes_per_token: per_tok,
    hardware_fits,
    context_curve,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn llama3_8b_int4_fits_on_24gb_rtx4090() {
    let res = analyze_vram(VramParams {
      preset_index: 0,
      mode: WorkloadMode::Inference,
      weight_prec: WeightPrecision::Int4Nf4,
      kv_prec: KvPrecision::Fp16,
      batch_size: 1,
      seq_len: 8_192,
      ..VramParams::default()
    });
    assert!(res.total_gb < 10.0, "got {:.2} GB", res.total_gb);
    assert_eq!(res.hardware_fits[0].min_cards, 1);
  }

  #[test]
  fn deepseek_mla_has_much_smaller_kv_than_standard_70b() {
    let llama70 = analyze_vram(VramParams {
      preset_index: 3, // Llama-3.3 70B
      kv_prec: KvPrecision::Fp16,
      ..VramParams::default()
    });
    let deepseek = analyze_vram(VramParams {
      preset_index: 4, // DeepSeek-V3 MLA
      kv_prec: KvPrecision::Fp16,
      ..VramParams::default()
    });
    assert!(deepseek.kv_bytes_per_token < llama70.kv_bytes_per_token * 0.3);
  }

  #[test]
  fn gradient_checkpointing_reduces_training_activations() {
    let no_ckpt = analyze_vram(VramParams {
      mode: WorkloadMode::LoraFinetune,
      grad_checkpointing: false,
      seq_len: 4_096,
      ..VramParams::default()
    });
    let with_ckpt = analyze_vram(VramParams {
      mode: WorkloadMode::LoraFinetune,
      grad_checkpointing: true,
      seq_len: 4_096,
      ..VramParams::default()
    });
    assert!(with_ckpt.activations_gb < no_ckpt.activations_gb * 0.3);
  }
}
