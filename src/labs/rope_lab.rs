use dioxus::prelude::*;

use crate::components::controls::{MetricItem, MetricStrip, RangeControl};
use crate::math::rope::{RopeParams, RopeScaling, analyze_rope};

fn format_ctx(n: usize) -> String {
  if n >= 1024 && n.is_multiple_of(1024) {
    format!("{}K", n / 1024)
  } else {
    format!("{n}")
  }
}

#[component]
pub fn RopeLab(zh: bool) -> Element {
  let mut params = use_signal(RopeParams::default);
  let analysis = use_memo(move || analyze_rope(params()));

  let p = params();
  let res = analysis();

  // Build SVG polylines for relative-distance inner product decay.
  let plot_w = 680.0_f64;
  let plot_h = 240.0_f64;
  let pad_l = 48.0_f64;
  let pad_r = 16.0_f64;
  let pad_t = 18.0_f64;
  let pad_b = 34.0_f64;
  let inner_w = plot_w - pad_l - pad_r;
  let inner_h = plot_h - pad_t - pad_b;
  let max_dist = p.target_ctx.max(1) as f64;

  let map_x = |d: usize| pad_l + (d as f64 / max_dist).clamp(0.0, 1.0) * inner_w;
  let map_y = |v: f64| {
    // Range [-0.35, 1.05]
    let norm = ((v - (-0.35)) / 1.40).clamp(0.0, 1.0);
    pad_t + (1.0 - norm) * inner_h
  };

  let baseline_pts = res
    .decay_curve
    .iter()
    .map(|pt| format!("{:.1},{:.1}", map_x(pt.distance), map_y(pt.baseline)))
    .collect::<Vec<_>>()
    .join(" ");
  let scaled_pts = res
    .decay_curve
    .iter()
    .map(|pt| format!("{:.1},{:.1}", map_x(pt.distance), map_y(pt.scaled)))
    .collect::<Vec<_>>()
    .join(" ");

  let train_line_x = map_x(p.train_ctx);
  let zero_line_y = map_y(0.0);

  // Build log-wavelength spectrum bars across dimension pairs.
  let spec_w = 680.0_f64;
  let spec_h = 155.0_f64;
  let spec_inner_w = spec_w - pad_l - pad_r;
  let spec_inner_h = spec_h - 16.0 - 28.0;
  let max_log_wl = res
    .spectrum
    .iter()
    .map(|s| s.scaled_wavelength.max(s.base_wavelength).log10())
    .fold(4.0_f64, f64::max)
    .max(4.0);
  let min_log_wl = 0.5_f64;
  let train_log_y = 16.0
    + (1.0
      - (((p.train_ctx as f64).log10() - min_log_wl) / (max_log_wl - min_log_wl)).clamp(0.0, 1.0))
      * spec_inner_h;

  // Phasor circle (2D rotation of selected pair at inspect_pos).
  let cx = 110.0_f64;
  let cy = 110.0_f64;
  let radius = 78.0_f64;
  let bx = cx + res.phasor.base_x * radius;
  let by = cy - res.phasor.base_y * radius;
  let sx = cx + res.phasor.scaled_x * radius;
  let sy = cy - res.phasor.scaled_y * radius;

  let num_pairs = (p.head_dim / 2).max(4);
  let selected_spec = res.spectrum.get(res.phasor.pair_index).cloned();

  let metrics = vec![
    MetricItem {
      label: if zh {
        "外推倍率 s".into()
      } else {
        "Scale Factor s".into()
      },
      value: format!("{:.2}x", res.scale_factor),
      sub: format!(
        "{} -> {}",
        format_ctx(p.train_ctx),
        format_ctx(p.target_ctx)
      ),
      tone: "tone-accent",
    },
    MetricItem {
      label: if zh {
        "YaRN 注意力温度 √(1/t)".into()
      } else {
        "YaRN Temp √(1/t)".into()
      },
      value: format!("{:.3}", res.attention_temp_scale),
      sub: if zh {
        "softmax logits 补偿".into()
      } else {
        "Attention entropy fix".into()
      },
      tone: "tone-teal",
    },
    MetricItem {
      label: if zh {
        "高 / 过渡 / 低频维对".into()
      } else {
        "Hi / Ramp / Lo Pairs".into()
      },
      value: format!(
        "{} / {} / {}",
        res.high_freq_pairs, res.mid_freq_pairs, res.low_freq_pairs
      ),
      sub: format!("{} pairs (d={})", num_pairs, p.head_dim),
      tone: "",
    },
    MetricItem {
      label: if zh {
        "越界区间内积 RMS".into()
      } else {
        "OOB Decay RMS".into()
      },
      value: format!("{:.3} vs {:.3}", res.scaled_oob_rms, res.baseline_oob_rms),
      sub: if zh {
        "外推后 vs 原始 RoPE".into()
      } else {
        "Scaled vs Baseline".into()
      },
      tone: "tone-amber",
    },
  ];

  rsx! {
    div { class: "lab-intro",
      h1 {
        if zh {
          "RoPE 与 YaRN 长上下文旋转频谱实验室"
        } else {
          "RoPE & YaRN Long-Context Frequency Lab"
        }
      }
      p {
        if zh {
          "实时对比 Standard RoPE、线性插值 (PI)、NTK-Aware 与 YaRN 在跨上下文窗口外推时的相对位置内积衰减、对数波长谱与二维相量旋转。"
        } else {
          "Compare Standard RoPE, Position Interpolation, NTK-Aware, and YaRN across relative-distance decay, log-wavelength spectrum, and 2D phasor rotation."
        }
      }
    }

    div { class: "lab-workbench",
      aside { class: "panel",
        div { class: "panel-heading",
          h2 { if zh { "模型与外推参数" } else { "Scaling Controls" } }
          span { class: "panel-caption", "d_head={p.head_dim}" }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "快捷预设" } else { "Model Presets" } }
          div { class: "chip-row",
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| params.set(RopeParams {
                head_dim: 128,
                base_theta: 500_000.0,
                train_ctx: 8_192,
                target_ctx: 65_536,
                scaling: RopeScaling::Yarn,
                inspect_pos: 2_048,
                inspect_pair: 8,
                ..RopeParams::default()
              }),
              "Llama-3 8K->64K"
            }
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| params.set(RopeParams {
                head_dim: 128,
                base_theta: 1_000_000.0,
                train_ctx: 32_768,
                target_ctx: 131_072,
                scaling: RopeScaling::Yarn,
                inspect_pos: 16_384,
                inspect_pair: 14,
                ..RopeParams::default()
              }),
              "Qwen2.5 32K->128K"
            }
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| params.set(RopeParams {
                head_dim: 64,
                base_theta: 10_000.0,
                train_ctx: 4_096,
                target_ctx: 32_768,
                scaling: RopeScaling::NtkAware,
                inspect_pos: 1_024,
                inspect_pair: 6,
                ..RopeParams::default()
              }),
              "Classic 10K Base"
            }
          }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "外推策略" } else { "Extrapolation Method" } }
          div { class: "chip-row",
            for mode in [
              RopeScaling::Standard,
              RopeScaling::PositionInterpolation,
              RopeScaling::NtkAware,
              RopeScaling::Yarn,
            ] {
              button {
                key: "{mode.label()}",
                r#type: "button",
                class: "chip-btn",
                aria_pressed: if p.scaling == mode { "true" } else { "false" },
                onclick: move |_| params.write().scaling = mode,
                "{mode.label()}"
              }
            }
          }
        }

        RangeControl {
          id: "rope-base".to_string(),
          label: if zh { "基频 θ_base (log10)".to_string() } else { "RoPE Base θ (log10)".to_string() },
          value_text: format!("{:.0}", p.base_theta),
          min: 3.5,
          max: 6.5,
          step: 0.1,
          value: p.base_theta.log10(),
          hint: Some(if zh {
            "Llama-2=10K，Llama-3=500K，Qwen2.5=1M".to_string()
          } else {
            "Llama-2: 10K, Llama-3: 500K, Qwen2.5: 1M".to_string()
          }),
          on_change: move |v: f64| params.write().base_theta = 10.0_f64.powf(v),
        }

        RangeControl {
          id: "rope-train-ctx".to_string(),
          label: if zh { "预训练窗口 L_train".to_string() } else { "Training Context L_train".to_string() },
          value_text: format_ctx(p.train_ctx),
          min: 11.0,
          max: 15.0,
          step: 1.0,
          value: (p.train_ctx as f64).log2(),
          on_change: move |v: f64| {
            let next_train = (2.0_f64.powf(v)).round() as usize;
            let mut w = params.write();
            w.train_ctx = next_train;
            if w.target_ctx < next_train {
              w.target_ctx = next_train;
            }
          },
        }

        RangeControl {
          id: "rope-target-ctx".to_string(),
          label: if zh { "目标推理窗口 L_target".to_string() } else { "Target Context L_target".to_string() },
          value_text: format_ctx(p.target_ctx),
          min: 12.0,
          max: 17.0,
          step: 1.0,
          value: (p.target_ctx as f64).log2(),
          on_change: move |v: f64| {
            let next_target = (2.0_f64.powf(v)).round() as usize;
            let mut w = params.write();
            w.target_ctx = next_target.max(w.train_ctx);
          },
        }

        RangeControl {
          id: "rope-pos".to_string(),
          label: if zh { "相量探测位置 m".to_string() } else { "Phasor Position m".to_string() },
          value_text: format!("pos {}", p.inspect_pos),
          min: 0.0,
          max: p.target_ctx as f64,
          step: 64.0,
          value: p.inspect_pos.min(p.target_ctx) as f64,
          on_change: move |v: f64| params.write().inspect_pos = v.round() as usize,
        }

        RangeControl {
          id: "rope-pair".to_string(),
          label: if zh { "观测维度对索引 i".to_string() } else { "Dimension Pair Index i".to_string() },
          value_text: format!("pair #{} / {}", p.inspect_pair, num_pairs.saturating_sub(1)),
          min: 0.0,
          max: num_pairs.saturating_sub(1) as f64,
          step: 1.0,
          value: p.inspect_pair.min(num_pairs.saturating_sub(1)) as f64,
          on_change: move |v: f64| params.write().inspect_pair = v.round() as usize,
        }

        div { class: "formula-note",
          if zh {
            "YaRN 核心思路：高频通道（波长远小于 L_train）保持原频率以保护局部语法精度，低频通道按比例 s 插值，过渡区以斜坡 γ(r) 平滑融合。"
          } else {
            "YaRN keeps high-frequency pairs unscaled to preserve local syntax while interpolating low-frequency pairs by s across ramp γ(r)."
          }
          br {}
          code { "θ'_i = (1 - γ) · (θ_i / s) + γ · θ_i" }
        }
      }

      div { class: "panel-stack",
        MetricStrip { items: metrics }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "相对位置内积衰减曲线 S(Δm)" } else { "Relative-Distance Inner Product S(Δm)" } }
            span { class: "panel-caption", "0 .. {format_ctx(p.target_ctx)}" }
          }
          div { class: "plot-frame",
            svg {
              class: "plot-svg",
              view_box: "0 0 680 240",
              role: "img",
              "aria-label": if zh { "RoPE 相对距离内积衰减对比曲线" } else { "RoPE relative distance decay comparison chart" },

              // Horizontal reference grid
              for tick in [1.0_f64, 0.5, 0.0, -0.25] {
                line {
                  x1: "{pad_l}",
                  x2: "{plot_w - pad_r}",
                  y1: "{map_y(tick):.1}",
                  y2: "{map_y(tick):.1}",
                  stroke: "var(--grid-line)",
                  stroke_width: "1",
                }
                text {
                  x: "{pad_l - 8.0}",
                  y: "{map_y(tick) + 4.0:.1}",
                  text_anchor: "end",
                  fill: "var(--ink-faint)",
                  font_size: "11",
                  font_family: "var(--font-mono)",
                  "{tick:.2}"
                }
              }

              // Zero axis
              line {
                x1: "{pad_l}",
                x2: "{plot_w - pad_r}",
                y1: "{zero_line_y:.1}",
                y2: "{zero_line_y:.1}",
                stroke: "var(--border-strong)",
                stroke_width: "1",
                stroke_dasharray: "4 4",
              }

              // Training context boundary
              line {
                x1: "{train_line_x:.1}",
                x2: "{train_line_x:.1}",
                y1: "{pad_t}",
                y2: "{plot_h - pad_b}",
                stroke: "var(--signal-amber)",
                stroke_width: "1.5",
                stroke_dasharray: "5 3",
              }
              text {
                x: "{(train_line_x + 6.0).min(plot_w - 110.0):.1}",
                y: "{pad_t + 12.0}",
                fill: "var(--signal-amber)",
                font_size: "11",
                font_family: "var(--font-mono)",
                "L_train = {format_ctx(p.train_ctx)}"
              }

              // Baseline curve
              polyline {
                fill: "none",
                stroke: "var(--ink-faint)",
                stroke_width: "1.75",
                stroke_dasharray: "4 3",
                points: "{baseline_pts}",
              }

              // Scaled curve
              polyline {
                fill: "none",
                stroke: "var(--accent)",
                stroke_width: "2.4",
                points: "{scaled_pts}",
              }

              // X-axis labels
              for frac in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
                text {
                  x: "{pad_l + frac * inner_w:.1}",
                  y: "{plot_h - 10.0}",
                  text_anchor: if frac == 0.0 { "start" } else if frac == 1.0 { "end" } else { "middle" },
                  fill: "var(--ink-faint)",
                  font_size: "11",
                  font_family: "var(--font-mono)",
                  "{format_ctx((frac * max_dist).round() as usize)}"
                }
              }
            }
            div { class: "legend-row",
              span { class: "legend-item",
                span { class: "legend-swatch", style: "background: var(--accent);" }
                "{p.scaling.label()}"
              }
              span { class: "legend-item",
                span { class: "legend-swatch", style: "background: var(--ink-faint);" }
                if zh { "原始 Standard RoPE 基线" } else { "Unscaled RoPE baseline" }
              }
              span { class: "legend-item",
                span { class: "legend-swatch", style: "background: var(--signal-amber);" }
                if zh { "训练窗口上限 L_train" } else { "Training length L_train" }
              }
            }
          }
        }

        div { class: "split-visual",
          section { class: "panel",
            div { class: "panel-heading",
              h2 { if zh { "通道对数波长谱 log₁₀(λ_i)" } else { "Per-Pair Log Wavelength Spectrum" } }
              span { class: "panel-caption", "pair 0 .. {num_pairs.saturating_sub(1)}" }
            }
            div { class: "plot-frame",
              svg {
                class: "plot-svg",
                view_box: "0 0 680 155",
                role: "img",
                "aria-label": if zh { "通道对数波长分布柱状图" } else { "Log wavelength distribution across dimension pairs" },

                line {
                  x1: "{pad_l}",
                  x2: "{spec_w - pad_r}",
                  y1: "{train_log_y:.1}",
                  y2: "{train_log_y:.1}",
                  stroke: "var(--signal-amber)",
                  stroke_width: "1.4",
                  stroke_dasharray: "4 3",
                }
                text {
                  x: "{pad_l - 6.0}",
                  y: "{train_log_y + 4.0:.1}",
                  text_anchor: "end",
                  fill: "var(--signal-amber)",
                  font_size: "10",
                  font_family: "var(--font-mono)",
                  "L_tr"
                }

                for pair in &res.spectrum {
                  {
                    let slot_w = spec_inner_w / num_pairs as f64;
                    let x = pad_l + pair.pair_index as f64 * slot_w + 1.0;
                    let w = (slot_w - 2.0).max(2.0);
                    let log_v = pair.scaled_wavelength.log10();
                    let h = (((log_v - min_log_wl) / (max_log_wl - min_log_wl)).clamp(0.03, 1.0)) * spec_inner_h;
                    let y = 16.0 + spec_inner_h - h;
                    let is_sel = pair.pair_index == res.phasor.pair_index;
                    let fill = if is_sel {
                      "var(--signal-crimson)"
                    } else if pair.yarn_ramp >= 0.99 {
                      "var(--accent)"
                    } else if pair.yarn_ramp <= 0.01 {
                      "var(--signal-teal)"
                    } else {
                      "var(--signal-amber)"
                    };
                    rsx! {
                      rect {
                        key: "{pair.pair_index}",
                        x: "{x:.1}",
                        y: "{y:.1}",
                        width: "{w:.1}",
                        height: "{h:.1}",
                        fill: "{fill}",
                        rx: "1",
                      }
                    }
                  }
                }

                text {
                  x: "{pad_l}",
                  y: "{spec_h - 8.0}",
                  fill: "var(--ink-faint)",
                  font_size: "11",
                  font_family: "var(--font-mono)",
                  if zh { "高频通道 (i=0)" } else { "High freq (i=0)" }
                }
                text {
                  x: "{spec_w - pad_r}",
                  y: "{spec_h - 8.0}",
                  text_anchor: "end",
                  fill: "var(--ink-faint)",
                  font_size: "11",
                  font_family: "var(--font-mono)",
                  if zh { "低频通道 (i={num_pairs.saturating_sub(1)})" } else { "Low freq (i={num_pairs.saturating_sub(1)})" }
                }
              }
            }
          }

          section { class: "panel",
            div { class: "panel-heading",
              h2 { if zh { "2D 旋转相量盘" } else { "2D Rotation Phasor" } }
              span { class: "panel-caption", "pair #{res.phasor.pair_index}" }
            }
            div { class: "plot-frame",
              svg {
                class: "plot-svg",
                view_box: "0 0 220 220",
                role: "img",
                "aria-label": if zh { "二维平面旋转相量对比" } else { "2D complex plane rotation phasor" },

                circle {
                  cx: "{cx}",
                  cy: "{cy}",
                  r: "{radius}",
                  fill: "none",
                  stroke: "var(--border-strong)",
                  stroke_width: "1.2",
                }
                line {
                  x1: "{cx - radius - 10.0}",
                  x2: "{cx + radius + 10.0}",
                  y1: "{cy}",
                  y2: "{cy}",
                  stroke: "var(--grid-line)",
                  stroke_width: "1",
                }
                line {
                  x1: "{cx}",
                  x2: "{cx}",
                  y1: "{cy - radius - 10.0}",
                  y2: "{cy + radius + 10.0}",
                  stroke: "var(--grid-line)",
                  stroke_width: "1",
                }
                // Baseline vector
                line {
                  x1: "{cx}",
                  y1: "{cy}",
                  x2: "{bx:.1}",
                  y2: "{by:.1}",
                  stroke: "var(--ink-faint)",
                  stroke_width: "2",
                  stroke_dasharray: "4 3",
                }
                circle { cx: "{bx:.1}", cy: "{by:.1}", r: "4", fill: "var(--ink-faint)" }
                // Scaled vector
                line {
                  x1: "{cx}",
                  y1: "{cy}",
                  x2: "{sx:.1}",
                  y2: "{sy:.1}",
                  stroke: "var(--signal-crimson)",
                  stroke_width: "2.5",
                }
                circle { cx: "{sx:.1}", cy: "{sy:.1}", r: "5", fill: "var(--signal-crimson)" }
              }
              if let Some(spec) = selected_spec {
                p { class: "control-hint",
                  "λ_base = {spec.base_wavelength:.1} tok | λ_scaled = {spec.scaled_wavelength:.1} tok | γ = {spec.yarn_ramp:.2}"
                }
              }
            }
          }
        }
      }
    }
  }
}
