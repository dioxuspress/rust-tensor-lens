use dioxus::prelude::*;

use crate::components::controls::{MetricItem, MetricStrip, RangeControl};
use crate::math::bpe::encode_bpe;

const SAMPLE_RUST: &str = "pub async fn rope_scale(ctx: usize) -> Result<Vec<f64>> {\n  // Rust zero-copy tensor kernel fusion\n  let scale = (ctx as f64 / 8192.0).max(1.0);\n  Ok(vec![scale])\n}";
const SAMPLE_ZH: &str = "大模型在长上下文推理中，通过位置编码外推、NF4 量化与注意力算子融合，显著降低显存占用并提升实时计算吞吐。";
const SAMPLE_JSON: &str = "{\"model\": \"Qwen2.5-32B\", \"quant\": \"NF4\", \"context_length\": 65536, \"kv_cache\": \"FP8\", \"safe\": true}";

fn token_color_class(id: u32, is_merged: bool, index: usize) -> &'static str {
  if !is_merged {
    return "tok-raw";
  }
  match (id as usize + index) % 4 {
    0 => "tok-c0",
    1 => "tok-c1",
    2 => "tok-c2",
    _ => "tok-c3",
  }
}

#[component]
pub fn BpeLab(zh: bool) -> Element {
  let mut text = use_signal(|| SAMPLE_RUST.to_string());
  let mut replay_step = use_signal(|| usize::MAX);

  let analysis = use_memo(move || encode_bpe(&text()));
  let res = analysis();

  let max_step = res.merge_steps.len();
  let effective_step = replay_step().min(max_step);
  let active_tokens = res
    .trace_snapshots
    .get(effective_step)
    .cloned()
    .unwrap_or_default();

  let metrics = vec![
    MetricItem {
      label: if zh {
        "当前步 Token 数".into()
      } else {
        "Tokens at Step".into()
      },
      value: format!("{}", active_tokens.len()),
      sub: if zh {
        format!(
          "最终 {} / 初始 {} 字节",
          res.final_tokens.len(),
          res.byte_count
        )
      } else {
        format!(
          "Final {} / {} raw bytes",
          res.final_tokens.len(),
          res.byte_count
        )
      },
      tone: "tone-accent",
    },
    MetricItem {
      label: if zh {
        "压缩比 (Bytes / Tok)".into()
      } else {
        "Bytes / Token".into()
      },
      value: format!("{:.2} B", res.bytes_per_token),
      sub: format!("{:.2} chars/tok", res.chars_per_token),
      tone: "tone-teal",
    },
    MetricItem {
      label: if zh {
        "序列长度压缩率".into()
      } else {
        "Sequence Reduction".into()
      },
      value: format!("{:.1}%", res.compression_pct),
      sub: format!("{} UTF-8 chars", res.char_count),
      tone: "",
    },
    MetricItem {
      label: if zh {
        "BPE 合并总轮次".into()
      } else {
        "Total Merge Passes".into()
      },
      value: format!("{effective_step} / {max_step}"),
      sub: if zh {
        "可拖动右侧步进条回放".into()
      } else {
        "Scrub step slider to replay".into()
      },
      tone: "tone-amber",
    },
  ];

  rsx! {
    div { class: "lab-intro",
      h1 {
        if zh {
          "Byte-Level BPE 分词与合并轨迹步进器"
        } else {
          "Byte-Level BPE Tokenizer & Merge Stepper"
        }
      }
      p {
        if zh {
          "直观观察任意文本如何从 UTF-8 原始字节（0–255）出发，按合并优先级 Rank 贪心迭代压缩为子词 Token，并对比 Rust 代码、中文与 JSON 的词元效率。"
        } else {
          "Inspect how UTF-8 byte streams (0-255) iteratively merge by lowest rank into subword tokens, and compare token efficiency across Rust code, Chinese text, and JSON."
        }
      }
    }

    div { class: "lab-workbench",
      aside { class: "panel",
        div { class: "panel-heading",
          h2 { if zh { "输入语料与步进回放" } else { "Input & Replay" } }
          span { class: "panel-caption", "{res.byte_count} bytes" }
        }

        div { class: "control-group",
          span { class: "control-label", if zh { "语料预设对比" } else { "Corpus Presets" } }
          div { class: "chip-row",
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| {
                text.set(SAMPLE_RUST.to_string());
                replay_step.set(usize::MAX);
              },
              "Rust Code"
            }
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| {
                text.set(SAMPLE_ZH.to_string());
                replay_step.set(usize::MAX);
              },
              if zh { "中文 AI 术语" } else { "Chinese AI" }
            }
            button {
              r#type: "button",
              class: "chip-btn",
              onclick: move |_| {
                text.set(SAMPLE_JSON.to_string());
                replay_step.set(usize::MAX);
              },
              "JSON Payload"
            }
          }
        }

        div { class: "control-group",
          label { class: "control-label", r#for: "bpe-input",
            if zh { "待分词文本" } else { "Text to Tokenize" }
          }
          textarea {
            id: "bpe-input",
            class: "lens-textarea",
            value: "{text}",
            oninput: move |evt| {
              text.set(evt.value());
              replay_step.set(usize::MAX);
            },
          }
        }

        if max_step > 0 {
          RangeControl {
            id: "bpe-step-slider".to_string(),
            label: if zh { "合并回放进度 (0 = 原始字节)".to_string() } else { "Merge Replay Step (0 = Raw Bytes)".to_string() },
            value_text: format!("step {effective_step} / {max_step}"),
            min: 0.0,
            max: max_step as f64,
            step: 1.0,
            value: effective_step as f64,
            on_change: move |v: f64| replay_step.set(v.round() as usize),
          }
        }

        div { class: "formula-note",
          if zh {
            "虚线灰块表示未合并的原始单字节（中文单字在 UTF-8 中占 3 字节，若不在词表则拆为 3 个原始字节 token，解释了为何未针对中文扩词的模型中文 token 消耗更高）。"
          } else {
            "Dashed chips represent unmerged raw bytes (0-255). Non-ASCII UTF-8 characters split into 3 raw bytes unless covered by dedicated merge rules."
          }
        }
      }

      div { class: "panel-stack",
        MetricStrip { items: metrics }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "实时词元切块视图（含 Token ID）" } else { "Tokenized Sequence & IDs" } }
            span { class: "panel-caption", "step {effective_step} · {active_tokens.len()} tokens" }
          }
          div { class: "token-stream",
            if active_tokens.is_empty() {
              span { class: "control-hint",
                if zh { "请在左侧输入文本开始分词。" } else { "Enter text on the left to tokenize." }
              }
            } else {
              for (idx, tok) in active_tokens.iter().enumerate() {
                span {
                  key: "{idx}-{tok.id}",
                  class: "token-chip {token_color_class(tok.id, tok.is_merged, idx)}",
                  title: "ID={tok.id}, bytes={tok.bytes:?}",
                  span { "{tok.display}" }
                  span { class: "token-chip-id", "#{tok.id}" }
                }
              }
            }
          }
        }

        section { class: "panel",
          div { class: "panel-heading",
            h2 { if zh { "BPE 贪心合并执行日志（点击任意一步跳转）" } else { "BPE Merge Trace (Click Step to Inspect)" } }
            span { class: "panel-caption", "{max_step} merges" }
          }
          div { class: "table-scroll trace-scroll",
            table { class: "data-table",
              thead {
                tr {
                  th { if zh { "步骤" } else { "Step" } }
                  th { class: "num", "Rank" }
                  th { if zh { "合并规则 (L + R -> New)" } else { "Merge Rule (L + R -> New)" } }
                  th { class: "num", if zh { "命中次数" } else { "Hits" } }
                  th { class: "num", if zh { "剩余 Tokens" } else { "Tokens After" } }
                }
              }
              tbody {
                for step in &res.merge_steps {
                  {
                    let step_idx = step.step_index;
                    rsx! {
                      tr {
                        key: "{step_idx}",
                        class: if step_idx == effective_step { "is-active" } else { "" },
                        style: "cursor: pointer;",
                        onclick: move |_| replay_step.set(step_idx),
                        td { "#{step_idx}" }
                        td { class: "num", "{step.rank}" }
                        td {
                          code { "{step.left_display} + {step.right_display} -> {step.merged_display} (#{step.merged_id})" }
                        }
                        td { class: "num", "{step.occurrences}" }
                        td { class: "num", "{step.tokens_after}" }
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
