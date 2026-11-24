use dioxus::prelude::*;

use crate::components::controls::{MetricItem, MetricStrip, RangeControl};
use crate::math::vram::{
  KvPrecision, MODEL_PRESETS, VramParams, WeightPrecision, WorkloadMode, analyze_vram,
};

fn format_seq(n: usize) -> String {
  if n >= 1024 && n.is_multiple_of(1024) {
    format!("{}K", n / 1024)
  } else {
    format!("{n}")
  }
}

#[component]
pub fn VramLab(zh: bool) -> Element {
  let mut params = use_signal(VramParams::default);
  let analysis = use_memo(move || analyze_vram(params()));

  let p = params();
  let res = analysis();
  let denom = res.total_gb.max(0.01);

  let pct_weights = (res.weights_gb / denom) * 100.0;
  let pct_kv = (res.kv_cache_gb / denom) * 100.0;
  let pct_opt = (res.optimizer_and_grads_gb / denom) * 100.0;
  let pct_act = (res.activations_gb / denom) * 100.0;
  let pct_ovh = (res.runtime_overhead_gb / denom) * 100.0;

  // Context length scaling chart (1K .. 128K)
  let plot_w = 680.0_f64;
  let plot_h = 215.0_f64;
  let pad_l = 52.0_f64;
  let pad_r = 18.0_f64;
  let pad_t = 18.0_f64;
  let pad_b = 32.0_f64;
  let inner_w = plot_w - pad_l - pad_r;
  let inner_h = plot_h - pad_t - pad_b;

  let max_y = res
    .context_curve
    .iter()
    .map(|pt| pt.total_gb)
    .fold(24.0_f64, f64::max)
    .max(24.0)
    * 1.12;
  let num_pts = res.context_curve.len().max(2);
  let cx = |idx: usize| pad_l + (idx as f64 / (num_pts - 1) as f64) * inner_w;
  let cy = |gb: f64| pad_t + (1.0 - (gb / max_y).clamp(0.0, 1.0)) * inner_h;

  let total_line = res
    .context_curve
    .iter()
    .enumerate()
    .map(|(i, pt)| format!("{:.1},{:.1}", cx(i), cy(pt.total_gb)))
    .collect::<Vec<_>>()
    .join(" ");
  let kv_line = res
    .context_curve
    .iter()
    .enumerate()
    .map(|(i, pt)| format!("{:.1},{:.1}", cx(i), cy(pt.kv_gb)))
    .collect::<Vec<_>>()
    .join(" ");

  let kv_kb_per_tok = res.kv_bytes_per_token / 1024.0;
  let kv_arch_tag = if let Some(mla) = res.model.mla_kv_dim {
    format!("MLA latent={mla}")
  } else {
    format!(
      "GQA {}Q/{}KV",
      res.model.num_q_heads, res.model.num_kv_heads
    )
  };

  let metrics = vec![
    MetricItem {
      label: if zh {
        "总显存需求".into()
      } else {
        "Total VRAM Budget".into()
      },
      value: format!("{:.2} GiB", res.total_gb),
      sub: format!("{} · {}", res.model.name, p.mode.label()),
      tone: if res.total_gb <= 24.0 {
        "tone-teal"
      } else if res.total_gb <= 80.0 {
        "tone-accent"
      } else {
        "tone-crimson"
      },
    },
    MetricItem {
      label: if zh {
        "模型权重显存".into()
      } else {
        "Model Weights".into()
      },
      value: format!("{:.2} GiB", res.weights_gb),
      sub: format!("{:.1}% of total", pct_weights),
      tone: "tone-accent",
    },
    MetricItem {
      label: if zh {
        "KV-Cache 占用".into()
      } else {
        "KV-Cache Footprint".into()
      },
      value: format!("{:.2} GiB", res.kv_cache_gb),
      sub: format!("{kv_kb_per_tok:.1} KiB/tok ({kv_arch_tag})"),
      tone: "tone-teal",
    },
    MetricItem {
      label: if zh {
        "优化器 / 激活显存".into()
      } else {
        "Opt + Grads + Act".into()
      },
      value: format!("{:.2} GiB", res.optimizer_and_grads_gb + res.activations_gb),
      sub: format!(
        "Opt {:.1}G + Act {:.1}G",
        res.optimizer_and_grads_gb, res.activations_gb
      ),
      tone: "tone-amber",
    },
  ];

  rsx! {
    div { class: "lab-intro",
      h1 {
        if zh {
          "LLM 显存预算与 KV-Cache 吞吐算术器"
        } else {
          "LLM VRAM & KV-Cache Budget Calculator"
        }
      }
      p {
        if zh {
          "面向主流开源大模型（含 GQA 与 DeepSeek-V3 MLA 压缩），精确测算推理、LoRA/QLoRA 与全参微调的显存构成、单卡余量及显存带宽屋顶线吞吐。"
        } else {
          "Estimate exact memory breakdown, multi-GPU card counts, and memory-bandwidth roofline decode speed across GQA and DeepSeek MLA models."
        }
      }
    }

    div { class: "lab-workbench",
      aside { class: "panel",
        div { class: "panel-heading",
          h2 { if zh { "模型与工作负载配置" } else { "Workload Configuration" } }
          span { class: "panel-caption", "{res.model.total_params_b:.1}B" }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "选择模型架构" } else { "Model Architecture" } }
          div { class: "chip-row",
            for (idx, preset) in MODEL_PRESETS.iter().enumerate() {
              button {
                key: "{preset.id}",
                r#type: "button",
                class: "chip-btn",
                aria_pressed: if p.preset_index == idx { "true" } else { "false" },
                onclick: move |_| params.write().preset_index = idx,
                "{preset.name}"
              }
            }
          }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "运行模式" } else { "Workload Mode" } }
          div { class: "chip-row",
            for mode in [
              WorkloadMode::Inference,
              WorkloadMode::LoraFinetune,
              WorkloadMode::FullFinetune,
            ] {
              button {
                key: "{mode.label()}",
                r#type: "button",
                class: "chip-btn",
                aria_pressed: if p.mode == mode { "true" } else { "false" },
                onclick: move |_| params.write().mode = mode,
                "{mode.label()}"
              }
            }
          }
        }

        if p.mode != WorkloadMode::FullFinetune {
          div { class: "control-group",
            span { class: "control-label", if zh { "权重精度" } else { "Weight Precision" } }
            div { class: "chip-row",
              for prec in [
                WeightPrecision::Fp16,
                WeightPrecision::Fp8,
                WeightPrecision::Int4Nf4,
                WeightPrecision::BitNet158,
              ] {
                button {
                  key: "{prec.label()}",
                  r#type: "button",
                  class: "chip-btn",
                  aria_pressed: if p.weight_prec == prec { "true" } else { "false" },
                  onclick: move |_| params.write().weight_prec = prec,
                  "{prec.label()}"
                }
              }
            }
          }
        }

        if p.mode == WorkloadMode::Inference {
          div { class: "control-group",
            span { class: "control-label", if zh { "KV-Cache 量化精度" } else { "KV-Cache Precision" } }
            div { class: "chip-row",
              for kv in [KvPrecision::Fp16, KvPrecision::Fp8, KvPrecision::Int4] {
                button {
                  key: "{kv.label()}",
                  r#type: "button",
                  class: "chip-btn",
                  aria_pressed: if p.kv_prec == kv { "true" } else { "false" },
                  onclick: move |_| params.write().kv_prec = kv,
                  "{kv.label()}"
                }
              }
            }
          }
        }

        RangeControl {
          id: "vram-batch".to_string(),
          label: if zh { "并发 Batch Size".to_string() } else { "Batch Size".to_string() },
          value_text: format!("B = {}", p.batch_size),
          min: 1.0,
          max: 32.0,
          step: 1.0,
          value: p.batch_size as f64,
          on_change: move |v: f64| params.write().batch_size = v.round() as usize,
        }

        RangeControl {
          id: "vram-seq".to_string(),
          label: if zh { "上下文长度 Sequence Length".to_string() } else { "Context Length".to_string() },
          value_text: format_seq(p.seq_len),
          min: 10.0,
          max: 17.0,
          step: 1.0,
          value: (p.seq_len as f64).log2(),
          on_change: move |v: f64| params.write().seq_len = (2.0_f64.powf(v)).round() as usize,
        }

        if p.mode == WorkloadMode::LoraFinetune {
          RangeControl {
            id: "vram-lora-rank".to_string(),
            label: if zh { "LoRA 秩 (Rank r)".to_string() } else { "LoRA Rank r".to_string() },
            value_text: format!("r = {}", p.lora_rank),
            min: 8.0,
            max: 128.0,
            step: 8.0,
            value: p.lora_rank as f64,
            on_change: move |v: f64| params.write().lora_rank = v.round() as usize,
          }
        }

        if p.mode != WorkloadMode::Inference {
          div { class: "control-group",
            label { class: "toggle-row", r#for: "vram-ckpt",
              span { class: "control-label",
                if zh { "启用 Gradient Checkpointing (激活重算)" } else { "Gradient Checkpointing" }
              }
              input {
                id: "vram-ckpt",
                r#type: "checkbox",
                checked: p.grad_checkpointing,
                onchange: move |evt| params.write().grad_checkpointing = evt.checked(),
              }
            }
          }
        }

        div { class: "formula-note",
          if zh {
            "对比技巧：切换 Llama-3.3 70B（GQA）与 DeepSeek-V3（MLA），观察单 token KV-Cache 从 320 KiB 骤降至 ~68 KiB，长上下文并发上限直接跃升数倍。"
          } else {
            "Try comparing Llama-3.3 70B (GQA) against DeepSeek-V3 (MLA): per-token FP16 KV-Cache drops from 320 KiB to ~68 KiB."
          }
        }
      }

      div { class: "panel-stack",
        MetricStrip { items: metrics }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "显存构成堆叠分解" } else { "Memory Allocation Breakdown" } }
            span { class: "panel-caption", "{res.total_gb:.2} GiB" }
          }
          div { class: "vram-bar-track", role: "img", aria_label: "VRAM breakdown bar",
            if pct_weights > 0.2 {
              div { class: "vram-seg seg-weights", style: "inline-size: {pct_weights:.2}%;", title: "Weights {res.weights_gb:.2} GiB" }
            }
            if pct_kv > 0.2 {
              div { class: "vram-seg seg-kv", style: "inline-size: {pct_kv:.2}%;", title: "KV-Cache {res.kv_cache_gb:.2} GiB" }
            }
            if pct_opt > 0.2 {
              div { class: "vram-seg seg-opt", style: "inline-size: {pct_opt:.2}%;", title: "Opt/Grads {res.optimizer_and_grads_gb:.2} GiB" }
            }
            if pct_act > 0.2 {
              div { class: "vram-seg seg-act", style: "inline-size: {pct_act:.2}%;", title: "Activations {res.activations_gb:.2} GiB" }
            }
            if pct_ovh > 0.2 {
              div { class: "vram-seg seg-ovh", style: "inline-size: {pct_ovh:.2}%;", title: "Runtime {res.runtime_overhead_gb:.2} GiB" }
            }
          }
          div { class: "legend-row",
            span { class: "legend-item",
              span { class: "legend-swatch seg-weights" }
              if zh { "权重 {res.weights_gb:.2}G" } else { "Weights {res.weights_gb:.2}G" }
            }
            span { class: "legend-item",
              span { class: "legend-swatch seg-kv" }
              "KV-Cache {res.kv_cache_gb:.2}G"
            }
            span { class: "legend-item",
              span { class: "legend-swatch seg-opt" }
              if zh { "优化器+梯度 {res.optimizer_and_grads_gb:.2}G" } else { "Opt+Grads {res.optimizer_and_grads_gb:.2}G" }
            }
            span { class: "legend-item",
              span { class: "legend-swatch seg-act" }
              if zh { "激活 {res.activations_gb:.2}G" } else { "Activations {res.activations_gb:.2}G" }
            }
            span { class: "legend-item",
              span { class: "legend-swatch seg-ovh" }
              if zh { "运行时 {res.runtime_overhead_gb:.2}G" } else { "Runtime {res.runtime_overhead_gb:.2}G" }
            }
          }
        }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "上下文长度扩展显存曲线 (1K -> 128K)" } else { "VRAM Growth vs. Context Length (1K -> 128K)" } }
            span { class: "panel-caption", "B={p.batch_size}" }
          }
          div { class: "plot-frame",
            svg {
              class: "plot-svg",
              view_box: "0 0 680 215",
              role: "img",
              "aria-label": if zh { "显存随上下文长度增长曲线" } else { "VRAM growth across sequence lengths" },

              // Reference GPU lines (24GB, 80GB)
              for (cap, label) in [(24.0_f64, "24G (4090)"), (80.0_f64, "80G (A100/H100)")] {
                if cap <= max_y {
                  line {
                    x1: "{pad_l}",
                    x2: "{plot_w - pad_r}",
                    y1: "{cy(cap):.1}",
                    y2: "{cy(cap):.1}",
                    stroke: "var(--signal-amber)",
                    stroke_width: "1.2",
                    stroke_dasharray: "5 3",
                  }
                  text {
                    x: "{plot_w - pad_r}",
                    y: "{cy(cap) - 4.0:.1}",
                    text_anchor: "end",
                    fill: "var(--signal-amber)",
                    font_size: "10",
                    font_family: "var(--font-mono)",
                    "{label}"
                  }
                }
              }

              // Y ticks
              for frac in [0.0_f64, 0.5, 1.0] {
                {
                  let v = frac * max_y;
                  rsx! {
                    line {
                      x1: "{pad_l}",
                      x2: "{plot_w - pad_r}",
                      y1: "{cy(v):.1}",
                      y2: "{cy(v):.1}",
                      stroke: "var(--grid-line)",
                      stroke_width: "1",
                    }
                    text {
                      x: "{pad_l - 8.0}",
                      y: "{cy(v) + 4.0:.1}",
                      text_anchor: "end",
                      fill: "var(--ink-faint)",
                      font_size: "10",
                      font_family: "var(--font-mono)",
                      "{v:.0}G"
                    }
                  }
                }
              }

              // KV curve (when inference)
              if p.mode == WorkloadMode::Inference {
                polyline {
                  fill: "none",
                  stroke: "var(--signal-teal)",
                  stroke_width: "2",
                  stroke_dasharray: "4 3",
                  points: "{kv_line}",
                }
              }

              // Total curve
              polyline {
                fill: "none",
                stroke: "var(--accent)",
                stroke_width: "2.5",
                points: "{total_line}",
              }

              // Dots & X labels
              for (i, pt) in res.context_curve.iter().enumerate() {
                circle {
                  key: "{pt.seq_len}",
                  cx: "{cx(i):.1}",
                  cy: "{cy(pt.total_gb):.1}",
                  r: if pt.seq_len == p.seq_len { "5" } else { "3" },
                  fill: if pt.seq_len == p.seq_len { "var(--signal-crimson)" } else { "var(--accent)" },
                }
                text {
                  x: "{cx(i):.1}",
                  y: "{plot_h - 8.0}",
                  text_anchor: "middle",
                  fill: "var(--ink-faint)",
                  font_size: "10",
                  font_family: "var(--font-mono)",
                  "{format_seq(pt.seq_len)}"
                }
              }
            }
          }
        }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "主流 GPU / Apple Silicon 硬件适配账单" } else { "Hardware Fit & Roofline Decode Matrix" } }
            span { class: "panel-caption", if zh { "按 72% 实效带宽估算" } else { "72% HBM efficiency" } }
          }
          div { class: "table-scroll",
            table { class: "data-table",
              thead {
                tr {
                  th { if zh { "硬件卡" } else { "Accelerator" } }
                  th { class: "num", if zh { "单卡显存" } else { "VRAM" } }
                  th { class: "num", if zh { "单卡占用" } else { "1-Card Util" } }
                  th { class: "num", if zh { "最少卡数" } else { "Min Cards" } }
                  th { class: "num", if zh { "带宽上限 Decode" } else { "Roofline Decode" } }
                  th { if zh { "状态" } else { "Status" } }
                }
              }
              tbody {
                for fit in &res.hardware_fits {
                  {
                    let (pill_cls, pill_text) = if fit.utilization_pct <= 82.0 {
                      ("status-pill status-ok", if zh { "余量充足" } else { "Comfortable" })
                    } else if fit.utilization_pct <= 95.0 {
                      ("status-pill status-warn", if zh { "贴边预警" } else { "Tight Fit" })
                    } else {
                      ("status-pill status-oom", if zh { "需多卡切分" } else { "Multi-GPU" })
                    };
                    let decode_str = if p.mode == WorkloadMode::Inference {
                      format!("{:.0} tok/s", fit.est_decode_toks_per_sec)
                    } else {
                      "—".to_string()
                    };
                    rsx! {
                      tr { key: "{fit.spec.name}",
                        td { "{fit.spec.name}" }
                        td { class: "num", "{fit.spec.vram_gb:.0} GiB" }
                        td { class: "num", "{fit.utilization_pct:.1}%" }
                        td { class: "num", "{fit.min_cards}x" }
                        td { class: "num", "{decode_str}" }
                        td { span { class: "{pill_cls}", "{pill_text}" } }
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
  }
}
