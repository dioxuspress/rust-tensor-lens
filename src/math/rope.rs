use std::f64::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeScaling {
  Standard,
  PositionInterpolation,
  NtkAware,
  Yarn,
}

impl RopeScaling {
  pub fn label(self) -> &'static str {
    match self {
      Self::Standard => "Standard RoPE",
      Self::PositionInterpolation => "PI (Linear)",
      Self::NtkAware => "NTK-Aware",
      Self::Yarn => "YaRN",
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
  pub head_dim: usize,
  pub base_theta: f64,
  pub train_ctx: usize,
  pub target_ctx: usize,
  pub scaling: RopeScaling,
  pub yarn_alpha: f64,
  pub yarn_beta: f64,
  pub inspect_pos: usize,
  pub inspect_pair: usize,
}

impl Default for RopeParams {
  fn default() -> Self {
    Self {
      head_dim: 128,
      base_theta: 500_000.0,
      train_ctx: 8_192,
      target_ctx: 65_536,
      scaling: RopeScaling::Yarn,
      yarn_alpha: 1.0,
      yarn_beta: 32.0,
      inspect_pos: 2_048,
      inspect_pair: 8,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairSpectrum {
  pub pair_index: usize,
  pub base_inv_freq: f64,
  pub scaled_inv_freq: f64,
  pub base_wavelength: f64,
  pub scaled_wavelength: f64,
  pub yarn_ramp: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayPoint {
  pub distance: usize,
  pub baseline: f64,
  pub scaled: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhasorSample {
  pub pair_index: usize,
  pub base_angle: f64,
  pub scaled_angle: f64,
  pub base_x: f64,
  pub base_y: f64,
  pub scaled_x: f64,
  pub scaled_y: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RopeAnalysis {
  pub scale_factor: f64,
  pub attention_temp_scale: f64,
  pub spectrum: Vec<PairSpectrum>,
  pub decay_curve: Vec<DecayPoint>,
  pub phasor: PhasorSample,
  pub high_freq_pairs: usize,
  pub mid_freq_pairs: usize,
  pub low_freq_pairs: usize,
  pub baseline_oob_rms: f64,
  pub scaled_oob_rms: f64,
}

pub fn analyze_rope(params: RopeParams) -> RopeAnalysis {
  let head_dim = params.head_dim.clamp(8, 256) & !1;
  let num_pairs = (head_dim / 2).max(4);
  let base_theta = params.base_theta.clamp(1_000.0, 10_000_000.0);
  let train_ctx = params.train_ctx.max(512);
  let target_ctx = params.target_ctx.max(train_ctx);
  let scale = (target_ctx as f64 / train_ctx as f64).max(1.0);
  let alpha = params.yarn_alpha.clamp(0.25, 8.0);
  let beta = params.yarn_beta.max(alpha + 1.0);

  let ntk_base = if scale > 1.0 && head_dim > 2 {
    let exponent = head_dim as f64 / (head_dim as f64 - 2.0);
    base_theta * scale.powf(exponent)
  } else {
    base_theta
  };

  let mut spectrum = Vec::with_capacity(num_pairs);
  let mut high_freq_pairs = 0usize;
  let mut mid_freq_pairs = 0usize;
  let mut low_freq_pairs = 0usize;

  for i in 0..num_pairs {
    let exp = (2 * i) as f64 / head_dim as f64;
    let base_inv_freq = 1.0 / base_theta.powf(exp);
    let base_wavelength = TAU / base_inv_freq;
    let rotations = train_ctx as f64 / base_wavelength;
    let ramp = ((rotations - alpha) / (beta - alpha)).clamp(0.0, 1.0);

    if ramp >= 0.999 {
      high_freq_pairs += 1;
    } else if ramp <= 0.001 {
      low_freq_pairs += 1;
    } else {
      mid_freq_pairs += 1;
    }

    let scaled_inv_freq = match params.scaling {
      RopeScaling::Standard => base_inv_freq,
      RopeScaling::PositionInterpolation => base_inv_freq / scale,
      RopeScaling::NtkAware => 1.0 / ntk_base.powf(exp),
      RopeScaling::Yarn => (1.0 - ramp) * (base_inv_freq / scale) + ramp * base_inv_freq,
    };
    let scaled_wavelength = TAU / scaled_inv_freq;

    spectrum.push(PairSpectrum {
      pair_index: i,
      base_inv_freq,
      scaled_inv_freq,
      base_wavelength,
      scaled_wavelength,
      yarn_ramp: ramp,
    });
  }

  // Sample relative distances from 0 to target_ctx with dense early sampling.
  let sample_count = 96usize;
  let mut decay_curve = Vec::with_capacity(sample_count + 1);
  let mut base_oob_sq = 0.0;
  let mut scaled_oob_sq = 0.0;
  let mut oob_count = 0usize;

  for step in 0..=sample_count {
    let t = step as f64 / sample_count as f64;
    // Slight quadratic concentration near short-to-mid distances while covering full target_ctx.
    let distance = ((t * 0.35 + t * t * 0.65) * target_ctx as f64).round() as usize;
    let mut base_sum = 0.0;
    let mut scaled_sum = 0.0;
    for pair in &spectrum {
      base_sum += (distance as f64 * pair.base_inv_freq).cos();
      scaled_sum += (distance as f64 * pair.scaled_inv_freq).cos();
    }
    let baseline = base_sum / num_pairs as f64;
    let scaled = scaled_sum / num_pairs as f64;

    if distance > train_ctx {
      base_oob_sq += baseline * baseline;
      scaled_oob_sq += scaled * scaled;
      oob_count += 1;
    }

    decay_curve.push(DecayPoint {
      distance,
      baseline,
      scaled,
    });
  }

  let pair_idx = params.inspect_pair.min(num_pairs.saturating_sub(1));
  let selected = &spectrum[pair_idx];
  let pos = params.inspect_pos.min(target_ctx) as f64;
  let base_angle = pos * selected.base_inv_freq;
  let scaled_angle = pos * selected.scaled_inv_freq;

  let attention_temp_scale = match params.scaling {
    RopeScaling::Yarn if scale > 1.0 => 0.1 * scale.ln() + 1.0,
    _ => 1.0,
  };

  let baseline_oob_rms = if oob_count > 0 {
    (base_oob_sq / oob_count as f64).sqrt()
  } else {
    0.0
  };
  let scaled_oob_rms = if oob_count > 0 {
    (scaled_oob_sq / oob_count as f64).sqrt()
  } else {
    0.0
  };

  RopeAnalysis {
    scale_factor: scale,
    attention_temp_scale,
    spectrum,
    decay_curve,
    phasor: PhasorSample {
      pair_index: pair_idx,
      base_angle,
      scaled_angle,
      base_x: base_angle.cos(),
      base_y: base_angle.sin(),
      scaled_x: scaled_angle.cos(),
      scaled_y: scaled_angle.sin(),
    },
    high_freq_pairs,
    mid_freq_pairs,
    low_freq_pairs,
    baseline_oob_rms,
    scaled_oob_rms,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn zero_distance_has_unit_inner_product() {
    let res = analyze_rope(RopeParams::default());
    let first = &res.decay_curve[0];
    assert_eq!(first.distance, 0);
    assert!((first.baseline - 1.0).abs() < 1e-12);
    assert!((first.scaled - 1.0).abs() < 1e-12);
  }

  #[test]
  fn yarn_preserves_highest_frequency_and_scales_lowest() {
    let params = RopeParams {
      head_dim: 128,
      base_theta: 10_000.0,
      train_ctx: 4_096,
      target_ctx: 32_768,
      scaling: RopeScaling::Yarn,
      ..RopeParams::default()
    };
    let res = analyze_rope(params);
    let hi = &res.spectrum[0];
    let lo = &res.spectrum[res.spectrum.len() - 1];
    assert!((hi.scaled_inv_freq - hi.base_inv_freq).abs() < 1e-9);
    assert!((lo.scaled_inv_freq - lo.base_inv_freq / 8.0).abs() < 1e-9);
    assert!(res.attention_temp_scale > 1.2);
  }

  #[test]
  fn pi_scales_every_pair_uniformly() {
    let params = RopeParams {
      train_ctx: 4_096,
      target_ctx: 16_384,
      scaling: RopeScaling::PositionInterpolation,
      ..RopeParams::default()
    };
    let res = analyze_rope(params);
    for pair in &res.spectrum {
      assert!((pair.scaled_inv_freq * 4.0 - pair.base_inv_freq).abs() < 1e-12);
    }
  }
}
