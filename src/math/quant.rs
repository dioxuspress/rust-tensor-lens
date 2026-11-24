use std::f64::consts::TAU;

/// Dettmers et al. (QLoRA) exact 16-level NormalFloat4 (NF4) codebook.
pub const NF4_CODEBOOK: [f64; 16] = [
  -1.0,
  -0.696_192_800_998_687_7,
  -0.525_073_051_452_636_7,
  -0.394_917_488_098_144_53,
  -0.284_441_381_692_886_35,
  -0.184_773_430_228_233_34,
  -0.091_050_036_251_544_95,
  0.0,
  0.079_580_299_556_255_34,
  0.160_930_201_411_247_25,
  0.246_112_301_945_686_34,
  0.337_915_241_718_292_24,
  0.440_709_829_330_444_34,
  0.562_617_003_917_694_1,
  0.722_956_836_223_602_3,
  1.0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
  SymInt8,
  SymInt4,
  AsymInt4,
  Nf4,
  BitNet158,
}

impl QuantScheme {
  pub const ALL: [Self; 5] = [
    Self::SymInt8,
    Self::SymInt4,
    Self::AsymInt4,
    Self::Nf4,
    Self::BitNet158,
  ];

  pub fn label(self) -> &'static str {
    match self {
      Self::SymInt8 => "Sym INT8",
      Self::SymInt4 => "Sym INT4",
      Self::AsymInt4 => "Asym INT4",
      Self::Nf4 => "NF4 (QLoRA)",
      Self::BitNet158 => "BitNet b1.58",
    }
  }

  pub fn nominal_bits(self) -> f64 {
    match self {
      Self::SymInt8 => 8.0,
      Self::SymInt4 | Self::AsymInt4 | Self::Nf4 => 4.0,
      Self::BitNet158 => 1.585,
    }
  }

  pub fn effective_bits(self, group_size: usize) -> f64 {
    let gs = group_size.max(8) as f64;
    let metadata_bits = match self {
      Self::AsymInt4 => 20.0, // FP16 scale + 4-bit zero-point
      Self::SymInt8 | Self::SymInt4 | Self::Nf4 | Self::BitNet158 => 16.0, // FP16 scale
    };
    self.nominal_bits() + metadata_bits / gs
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantParams {
  pub scheme: QuantScheme,
  pub group_size: usize,
  pub outlier_sigma: f64,
  pub outlier_count: usize,
  pub seed: u64,
}

impl Default for QuantParams {
  fn default() -> Self {
    Self {
      scheme: QuantScheme::Nf4,
      group_size: 32,
      outlier_sigma: 6.5,
      outlier_count: 2,
      seed: 42,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantCell {
  pub row: usize,
  pub col: usize,
  pub original: f64,
  pub dequantized: f64,
  pub abs_error: f64,
  pub is_outlier: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchemeReport {
  pub scheme: QuantScheme,
  pub mse: f64,
  pub rmse: f64,
  pub sqnr_db: f64,
  pub cosine_sim: f64,
  pub max_abs_err: f64,
  pub effective_bits: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferPoint {
  pub input: f64,
  pub output: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantAnalysis {
  pub cells: Vec<QuantCell>,
  pub max_cell_error: f64,
  pub selected_report: SchemeReport,
  pub comparison: Vec<SchemeReport>,
  pub transfer_curve: Vec<TransferPoint>,
}

/// Deterministic xorshift64* PRNG for reproducible Gaussian weight tiles without extra crates.
struct Rng64(u64);

impl Rng64 {
  fn new(seed: u64) -> Self {
    let init = if seed == 0 {
      0x9E37_79B9_7F4A_7C15
    } else {
      seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    };
    Self(init)
  }

  fn next_u64(&mut self) -> u64 {
    let mut x = self.0;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    self.0 = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
  }

  fn next_unit_open(&mut self) -> f64 {
    let v = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
    v.clamp(1e-12, 1.0 - 1e-12)
  }

  fn next_normal(&mut self) -> f64 {
    let u1 = self.next_unit_open();
    let u2 = self.next_unit_open();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
  }
}

fn nearest_nf4(norm: f64) -> f64 {
  let mut best = NF4_CODEBOOK[0];
  let mut best_dist = (norm - best).abs();
  for &level in &NF4_CODEBOOK[1..] {
    let dist = (norm - level).abs();
    if dist < best_dist {
      best_dist = dist;
      best = level;
    }
  }
  best
}

pub fn quantize_dequantize_block(
  weights: &[f64],
  scheme: QuantScheme,
  group_size: usize,
) -> Vec<f64> {
  let gs = group_size.clamp(8, weights.len().max(8));
  let mut out = Vec::with_capacity(weights.len());

  for chunk in weights.chunks(gs) {
    match scheme {
      QuantScheme::SymInt8 => {
        let absmax = chunk.iter().fold(0.0_f64, |m, &w| m.max(w.abs()));
        let scale = (absmax / 127.0).max(1e-12);
        for &w in chunk {
          let q = (w / scale).round().clamp(-127.0, 127.0);
          out.push(q * scale);
        }
      }
      QuantScheme::SymInt4 => {
        let absmax = chunk.iter().fold(0.0_f64, |m, &w| m.max(w.abs()));
        let scale = (absmax / 7.0).max(1e-12);
        for &w in chunk {
          let q = (w / scale).round().clamp(-7.0, 7.0);
          out.push(q * scale);
        }
      }
      QuantScheme::AsymInt4 => {
        let min_v = chunk.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scale = ((max_v - min_v) / 15.0).max(1e-12);
        let zero_point = (-min_v / scale).round().clamp(0.0, 15.0);
        for &w in chunk {
          let q = (w / scale + zero_point).round().clamp(0.0, 15.0);
          out.push((q - zero_point) * scale);
        }
      }
      QuantScheme::Nf4 => {
        let absmax = chunk
          .iter()
          .fold(0.0_f64, |m, &w| m.max(w.abs()))
          .max(1e-12);
        for &w in chunk {
          let norm = (w / absmax).clamp(-1.0, 1.0);
          out.push(nearest_nf4(norm) * absmax);
        }
      }
      QuantScheme::BitNet158 => {
        let absmean =
          (chunk.iter().map(|w| w.abs()).sum::<f64>() / chunk.len().max(1) as f64).max(1e-12);
        for &w in chunk {
          let q = (w / absmean).round().clamp(-1.0, 1.0);
          out.push(q * absmean);
        }
      }
    }
  }

  out
}

fn evaluate_scheme(
  weights: &[f64],
  dequant: &[f64],
  scheme: QuantScheme,
  group_size: usize,
) -> SchemeReport {
  let n = weights.len().max(1) as f64;
  let mut err_sq = 0.0;
  let mut sig_sq = 0.0;
  let mut deq_sq = 0.0;
  let mut dot = 0.0;
  let mut max_abs_err = 0.0_f64;

  for (&w, &dq) in weights.iter().zip(dequant.iter()) {
    let diff = (w - dq).abs();
    err_sq += diff * diff;
    sig_sq += w * w;
    deq_sq += dq * dq;
    dot += w * dq;
    if diff > max_abs_err {
      max_abs_err = diff;
    }
  }

  let mse = err_sq / n;
  let rmse = mse.sqrt();
  let sqnr_db = if err_sq <= 1e-15 {
    60.0
  } else {
    10.0 * (sig_sq / err_sq).log10()
  };
  let denom = (sig_sq * deq_sq).sqrt();
  let cosine_sim = if denom > 1e-12 {
    (dot / denom).clamp(-1.0, 1.0)
  } else {
    0.0
  };

  SchemeReport {
    scheme,
    mse,
    rmse,
    sqnr_db,
    cosine_sim,
    max_abs_err,
    effective_bits: scheme.effective_bits(group_size),
  }
}

pub fn analyze_quantization(params: QuantParams) -> QuantAnalysis {
  const DIM: usize = 8;
  const TOTAL: usize = DIM * DIM;

  let mut rng = Rng64::new(params.seed);
  let mut weights = Vec::with_capacity(TOTAL);
  let mut outlier_flags = [false; TOTAL];

  for _ in 0..TOTAL {
    weights.push(rng.next_normal());
  }

  let outlier_slots = [11usize, 37, 52, 20, 46, 5];
  let count = params.outlier_count.min(outlier_slots.len());
  for (idx, &slot) in outlier_slots.iter().take(count).enumerate() {
    let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
    weights[slot] = sign * params.outlier_sigma.clamp(1.0, 16.0);
    outlier_flags[slot] = true;
  }

  let mut comparison = Vec::with_capacity(QuantScheme::ALL.len());
  let mut selected_dequant = Vec::new();
  let mut selected_report = None;

  for scheme in QuantScheme::ALL {
    let deq = quantize_dequantize_block(&weights, scheme, params.group_size);
    let report = evaluate_scheme(&weights, &deq, scheme, params.group_size);
    if scheme == params.scheme {
      selected_dequant = deq;
      selected_report = Some(report);
    }
    comparison.push(report);
  }

  let mut cells = Vec::with_capacity(TOTAL);
  let mut max_cell_error = 1e-6_f64;
  for idx in 0..TOTAL {
    let original = weights[idx];
    let dequantized = selected_dequant.get(idx).copied().unwrap_or(0.0);
    let abs_error = (original - dequantized).abs();
    if abs_error > max_cell_error {
      max_cell_error = abs_error;
    }
    cells.push(QuantCell {
      row: idx / DIM,
      col: idx % DIM,
      original,
      dequantized,
      abs_error,
      is_outlier: outlier_flags[idx],
    });
  }

  // Build normalized [-1, 1] transfer curve for the selected scheme.
  let steps = 80usize;
  let mut probe = Vec::with_capacity(steps + 1);
  for i in 0..=steps {
    let x = -1.0 + 2.0 * (i as f64 / steps as f64);
    probe.push(x);
  }
  let mapped = quantize_dequantize_block(&probe, params.scheme, probe.len());
  let transfer_curve = probe
    .into_iter()
    .zip(mapped)
    .map(|(input, output)| TransferPoint { input, output })
    .collect();

  let fallback_report = comparison[0];

  QuantAnalysis {
    cells,
    max_cell_error,
    selected_report: selected_report.unwrap_or(fallback_report),
    comparison,
    transfer_curve,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn int8_outperforms_int4_and_bitnet_on_sqnr() {
    let res = analyze_quantization(QuantParams {
      outlier_count: 0,
      group_size: 64,
      ..QuantParams::default()
    });
    let int8 = res
      .comparison
      .iter()
      .find(|r| r.scheme == QuantScheme::SymInt8)
      .unwrap();
    let int4 = res
      .comparison
      .iter()
      .find(|r| r.scheme == QuantScheme::SymInt4)
      .unwrap();
    let bitnet = res
      .comparison
      .iter()
      .find(|r| r.scheme == QuantScheme::BitNet158)
      .unwrap();
    assert!(int8.sqnr_db > int4.sqnr_db + 15.0);
    assert!(int4.sqnr_db > bitnet.sqnr_db);
  }

  #[test]
  fn smaller_group_size_reduces_outlier_damage() {
    let coarse = analyze_quantization(QuantParams {
      scheme: QuantScheme::SymInt4,
      group_size: 64,
      outlier_sigma: 12.0,
      outlier_count: 1,
      seed: 7,
    });
    let fine = analyze_quantization(QuantParams {
      scheme: QuantScheme::SymInt4,
      group_size: 16,
      outlier_sigma: 12.0,
      outlier_count: 1,
      seed: 7,
    });
    assert!(fine.selected_report.sqnr_db > coarse.selected_report.sqnr_db);
  }

  #[test]
  fn nf4_beats_uniform_int4_on_gaussian_weights() {
    let res = analyze_quantization(QuantParams {
      outlier_count: 0,
      group_size: 64,
      seed: 99,
      ..QuantParams::default()
    });
    let nf4 = res
      .comparison
      .iter()
      .find(|r| r.scheme == QuantScheme::Nf4)
      .unwrap();
    let sym4 = res
      .comparison
      .iter()
      .find(|r| r.scheme == QuantScheme::SymInt4)
      .unwrap();
    assert!(nf4.sqnr_db > sym4.sqnr_db);
  }
}
