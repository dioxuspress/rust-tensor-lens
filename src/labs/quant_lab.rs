use dioxus::prelude::*;

use crate::components::controls::{MetricItem, MetricStrip, RangeControl};
use crate::math::quant::{QuantParams, QuantScheme, analyze_quantization};

fn heat_class(abs_err: f64, max_err: f64, is_outlier: bool) -> &'static str {
  let ratio = (abs_err / max_err.max(1e-6)).clamp(0.0, 1.0);
  match (ratio, is_outlier) {
    (r, true) if r >= 0.80 => "heatmap-cell heat-5 is-outlier",
    (r, true) if r >= 0.60 => "heatmap-cell heat-4 is-outlier",
    (r, true) if r >= 0.42 => "heatmap-cell heat-3 is-outlier",
    (r, true) if r >= 0.25 => "heatmap-cell heat-2 is-outlier",
    (r, true) if r >= 0.10 => "heatmap-cell heat-1 is-outlier",
    (_, true) => "heatmap-cell heat-0 is-outlier",
    (r, false) if r >= 0.80 => "heatmap-cell heat-5",
    (r, false) if r >= 0.60 => "heatmap-cell heat-4",
    (r, false) if r >= 0.42 => "heatmap-cell heat-3",
    (r, false) if r >= 0.25 => "heatmap-cell heat-2",
    (r, false) if r >= 0.10 => "heatmap-cell heat-1",
    (_, false) => "heatmap-cell heat-0",
  }
}

#[component]
pub fn QuantLab(zh: bool) -> Element {
  let mut params = use_signal(QuantParams::default);
  let analysis = use_memo(move || analyze_quantization(params()));

  let p = params();
  let res = analysis();
  let sel = res.selected_report;

  // Transfer curve SVG mapping [-1, 1] -> [-1, 1]
  let svg_size = 240.0_f64;
  let pad = 28.0_f64;
  let inner = svg_size - pad * 2.0;
  let tx = |x: f64| pad + ((x + 1.0) * 0.5).clamp(0.0, 1.0) * inner;
  let ty = |y: f64| pad + (1.0 - ((y + 1.0) * 0.5).clamp(0.0, 1.0)) * inner;

  let transfer_pts = res
    .transfer_curve
    .iter()
    .map(|pt| format!("{:.1},{:.1}", tx(pt.input), ty(pt.output)))
    .collect::<Vec<_>>()
    .join(" ");

  let metrics = vec![
    MetricItem {
      label: if zh {
        "信噪比 SQNR".into()
      } else {
        "Signal-to-Noise (SQNR)".into()
      },
      value: format!("{:.2} dB", sel.sqnr_db),
      sub: if zh {
        "越高代表精度保留越好".into()
      } else {
        "Higher preserves more precision".into()
      },
      tone: if sel.sqnr_db >= 22.0 {
        "tone-teal"
      } else if sel.sqnr_db >= 14.0 {
        "tone-amber"
      } else {
        "tone-crimson"
      },
    },
    MetricItem {
      label: if zh {
        "余弦相似度".into()
      } else {
        "Cosine Similarity".into()
      },
      value: format!("{:.5}", sel.cosine_sim),
      sub: format!("RMSE {:.4}", sel.rmse),
      tone: "tone-accent",
    },
    MetricItem {
      label: if zh {
        "最大绝对误差".into()
      } else {
        "Max Abs Error".into()
      },
      value: format!("{:.4}", sel.max_abs_err),
      sub: format!("MSE {:.5}", sel.mse),
      tone: "tone-crimson",
    },
    MetricItem {
      label: if zh {
        "有效位宽 (含缩放因子)".into()
      } else {
        "Effective Bits / Param".into()
      },
      value: format!("{:.2} bit", sel.effective_bits),
      sub: format!("group_size = {}", p.group_size),
      tone: "",
    },
  ];

  rsx! {
    div { class: "lab-intro",
      h1 {
        if zh {
          "INT8 / INT4 / NF4 / BitNet 1.58-bit 量化误差热力图"
        } else {
          "INT8 / INT4 / NF4 / BitNet 1.58-bit Quantization Error Lab"
        }
      }
      p {
        if zh {
          "向 8×8 高斯权重块注入不同强度的激活/权重离群值 (Outliers)，实时观察分组大小 (Group Size) 与码本分布如何决定量化误差与 SQNR。"
        } else {
          "Inject outlier spikes into an 8x8 Gaussian weight tile and inspect how group size and codebook spacing govern quantization error and SQNR."
        }
      }
    }

    div { class: "lab-workbench",
      aside { class: "panel",
        div { class: "panel-heading",
          h2 { if zh { "量化配置" } else { "Quantizer Settings" } }
          span { class: "panel-caption", "{p.scheme.label()}" }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "典型场景预设" } else { "Scenario Presets" } }
          div { class: "chip-row",
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| params.set(QuantParams {
                scheme: QuantScheme::Nf4,
                group_size: 64,
                outlier_sigma: 1.0,
                outlier_count: 0,
                seed: 42,
              }),
              if zh { "纯高斯权重" } else { "Clean Gaussian" }
            }
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| params.set(QuantParams {
                scheme: QuantScheme::SymInt4,
                group_size: 64,
                outlier_sigma: 10.0,
                outlier_count: 2,
                seed: 42,
              }),
              if zh { "离群值坍塌 (10σ)" } else { "10σ Outlier Collapse" }
            }
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| params.set(QuantParams {
                scheme: QuantScheme::BitNet158,
                group_size: 32,
                outlier_sigma: 4.0,
                outlier_count: 1,
                seed: 7,
              }),
              "BitNet 1.58b"
            }
          }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "量化方案" } else { "Quantization Scheme" } }
          div { class: "chip-row",
            for scheme in QuantScheme::ALL {
              button {
                key: "{scheme.label()}",
                r#type: "button",
                class: "chip-btn",
                aria_pressed: if p.scheme == scheme { "true" } else { "false" },
                onclick: move |_| params.write().scheme = scheme,
                "{scheme.label()}"
              }
            }
          }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "分组量化粒度 (Group Size)" } else { "Group Size" } }
          div { class: "chip-row",
            for gs in [64usize, 32, 16, 8] {
              button {
                key: "{gs}",
                r#type: "button",
                class: "chip-btn",
                aria_pressed: if p.group_size == gs { "true" } else { "false" },
                onclick: move |_| params.write().group_size = gs,
                "g={gs}"
              }
            }
          }
        }

        RangeControl {
          id: "quant-outlier-sigma".to_string(),
          label: if zh { "离群值幅度 (×σ)".to_string() } else { "Outlier Magnitude (×σ)".to_string() },
          value_text: format!("{:.1}σ", p.outlier_sigma),
          min: 1.0,
          max: 16.0,
          step: 0.5,
          value: p.outlier_sigma,
          hint: Some(if zh {
            "模拟大模型通道中远超 ±3σ 的极端离群权重".to_string()
          } else {
            "Simulates emergent channel outliers beyond ±3σ".to_string()
          }),
          on_change: move |v: f64| params.write().outlier_sigma = v,
        }

        RangeControl {
          id: "quant-outlier-count".to_string(),
          label: if zh { "离群元素数量".to_string() } else { "Injected Outliers".to_string() },
          value_text: format!("{} / 64", p.outlier_count),
          min: 0.0,
          max: 6.0,
          step: 1.0,
          value: p.outlier_count as f64,
          on_change: move |v: f64| params.write().outlier_count = v.round() as usize,
        }

        RangeControl {
          id: "quant-seed".to_string(),
          label: if zh { "权重块随机种子".to_string() } else { "Tile PRNG Seed".to_string() },
          value_text: format!("#{}", p.seed),
          min: 1.0,
          max: 100.0,
          step: 1.0,
          value: p.seed as f64,
          on_change: move |v: f64| params.write().seed = v.round() as u64,
        }

        div { class: "formula-note",
          if zh {
            "为什么 NF4 优于均匀 INT4？因为 Transformer 权重服从零均值正态分布，NF4 把 16 个码点按标准正态等概率分位排布，在零附近更密、尾部更疏；而小 Group Size 能把离群值对缩放因子 scale 的污染隔离在局部子块。"
          } else {
            "NF4 places 16 quantization levels at equal-probability quantiles of N(0,1), allocating denser resolution near zero while smaller group sizes isolate outlier-induced scale inflation."
          }
        }
      }

      div { class: "panel-stack",
        MetricStrip { items: metrics }

        div { class: "split-visual",
          section { class: "panel",
            div { class: "panel-heading",
              h2 { if zh { "8×8 权重块逐元素误差热力图 |w - ŵ|" } else { "8x8 Tile Error Heatmap |w - ŵ|" } }
              span { class: "panel-caption", if zh { "琥珀边框 = 注入离群值" } else { "Amber ring = outlier" } }
            }
            div { class: "heatmap-grid",
              for cell in &res.cells {
                div {
                  key: "{cell.row}-{cell.col}",
                  class: heat_class(cell.abs_error, res.max_cell_error, cell.is_outlier),
                  title: "w={cell.original:.3}, deq={cell.dequantized:.3}, |err|={cell.abs_error:.4}",
                  span { class: "heatmap-err", "{cell.abs_error:.2}" }
                  span { class: "heatmap-orig", "{cell.original:+.1}" }
                }
              }
            }
          }

          section { class: "panel",
            div { class: "panel-heading",
              h2 { if zh { "归一化量化阶梯映射" } else { "Normalized Transfer Curve" } }
              span { class: "panel-caption", "{p.scheme.label()}" }
            }
            div { class: "plot-frame",
              svg {
                class: "plot-svg",
                view_box: "0 0 240 240",
                role: "img",
                "aria-label": if zh { "量化阶梯映射曲线" } else { "Quantization staircase transfer function" },

                // Axes
                line { x1: "{pad}", x2: "{svg_size - pad}", y1: "{ty(0.0):.1}", y2: "{ty(0.0):.1}", stroke: "var(--grid-line)", stroke_width: "1" }
                line { x1: "{tx(0.0):.1}", x2: "{tx(0.0):.1}", y1: "{pad}", y2: "{svg_size - pad}", stroke: "var(--grid-line)", stroke_width: "1" }
                // Ideal y = x diagonal
                line {
                  x1: "{tx(-1.0):.1}",
                  y1: "{ty(-1.0):.1}",
                  x2: "{tx(1.0):.1}",
                  y2: "{ty(1.0):.1}",
                  stroke: "var(--ink-faint)",
                  stroke_width: "1.2",
                  stroke_dasharray: "4 3",
                }
                // Quantized staircase
                polyline {
                  fill: "none",
                  stroke: "var(--accent)",
                  stroke_width: "2.3",
                  points: "{transfer_pts}",
                }
                text { x: "{pad}", y: "{svg_size - 8.0}", fill: "var(--ink-faint)", font_size: "10", font_family: "var(--font-mono)", "-1.0" }
                text { x: "{svg_size - pad}", y: "{svg_size - 8.0}", text_anchor: "end", fill: "var(--ink-faint)", font_size: "10", font_family: "var(--font-mono)", "+1.0" }
              }
              p { class: "control-hint",
                if zh {
                  "虚线为理想恒等映射 y=x，蓝线为当前码本阶梯。"
                } else {
                  "Dashed line: ideal y=x; solid curve: active quantizer staircase."
                }
              }
            }
          }
        }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "全方案同张量实时横向对比" } else { "Live Cross-Scheme Comparison (Same Tensor)" } }
            span { class: "panel-caption", "group_size={p.group_size}" }
          }
          div { class: "table-scroll",
            table { class: "data-table",
              thead {
                tr {
                  th { if zh { "量化方案" } else { "Scheme" } }
                  th { class: "num", if zh { "有效位宽" } else { "Eff. Bits" } }
                  th { class: "num", "SQNR (dB)" }
                  th { class: "num", "RMSE" }
                  th { class: "num", if zh { "最大误差" } else { "Max |Err|" } }
                  th { class: "num", if zh { "余弦相似度" } else { "Cosine Sim" } }
                }
              }
              tbody {
                for row in &res.comparison {
                  tr {
                    key: "{row.scheme.label()}",
                    class: if row.scheme == p.scheme { "is-active" } else { "" },
                    td { "{row.scheme.label()}" }
                    td { class: "num", "{row.effective_bits:.2}" }
                    td { class: "num", "{row.sqnr_db:.2}" }
                    td { class: "num", "{row.rmse:.4}" }
                    td { class: "num", "{row.max_abs_err:.4}" }
                    td { class: "num", "{row.cosine_sim:.5}" }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
