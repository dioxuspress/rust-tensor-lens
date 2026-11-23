mod components;
mod labs;
mod math;

use dioxus::prelude::*;
use labs::{bpe_lab::BpeLab, quant_lab::QuantLab, rope_lab::RopeLab, vram_lab::VramLab};

const MAIN_CSS: Asset = asset!("/assets/main.css");
const FAVICON_SVG: Asset = asset!("/assets/favicon.svg");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabTab {
  Rope,
  Quant,
  Bpe,
  Vram,
}

impl LabTab {
  const ALL: [Self; 4] = [Self::Rope, Self::Quant, Self::Bpe, Self::Vram];

  fn id(self) -> &'static str {
    match self {
      Self::Rope => "tab-rope",
      Self::Quant => "tab-quant",
      Self::Bpe => "tab-bpe",
      Self::Vram => "tab-vram",
    }
  }

  fn label(self, zh: bool) -> &'static str {
    match (self, zh) {
      (Self::Rope, true) => "RoPE / YaRN 外推",
      (Self::Rope, false) => "RoPE / YaRN Scaling",
      (Self::Quant, true) => "INT4 / NF4 / 1.58b 量化",
      (Self::Quant, false) => "INT4 / NF4 / 1.58b Quant",
      (Self::Bpe, true) => "Byte-Level BPE 分词",
      (Self::Bpe, false) => "Byte-Level BPE",
      (Self::Vram, true) => "LLM 显存与 KV-Cache",
      (Self::Vram, false) => "LLM VRAM & KV-Cache",
    }
  }

  fn prev(self) -> Self {
    match self {
      Self::Rope => Self::Vram,
      Self::Quant => Self::Rope,
      Self::Bpe => Self::Quant,
      Self::Vram => Self::Bpe,
    }
  }

  fn next(self) -> Self {
    match self {
      Self::Rope => Self::Quant,
      Self::Quant => Self::Bpe,
      Self::Bpe => Self::Vram,
      Self::Vram => Self::Rope,
    }
  }
}

fn main() {
  dioxus::launch(App);
}

#[component]
fn App() -> Element {
  let mut active_tab = use_signal(|| LabTab::Rope);
  let mut zh = use_signal(|| true);
  let mut dark = use_signal(|| false);

  let is_zh = zh();
  let is_dark = dark();
  let current = active_tab();

  rsx! {
    document::Link { rel: "icon", r#type: "image/svg+xml", href: FAVICON_SVG }
    document::Stylesheet { href: MAIN_CSS }
    div {
      class: "lens-root",
      "data-lens-root": "true",
      "data-theme": if is_dark { "dark" } else { "light" },

      header { class: "lens-header",
        div { class: "lens-header-inner",
          div { class: "lens-brand",
            span { class: "lens-brand-mark", "TL" }
            span { class: "lens-brand-title", "Rust Tensor Lens" }
            span { class: "lens-brand-tag",
              if is_zh { "纯 Rust · WASM 零后端算子实验室" } else { "Pure-Rust WASM Operator Lab" }
            }
          }
          div { class: "lens-header-actions",
            button {
              r#type: "button",
              class: "tool-btn",
              aria_label: if is_zh { "Switch language to English" } else { "切换界面为中文" },
              onclick: move |_| zh.toggle(),
              if is_zh { "EN" } else { "中文" }
            }
            button {
              r#type: "button",
              class: "tool-btn",
              onclick: move |_| dark.toggle(),
              if is_dark {
                if is_zh { "亮色图纸" } else { "Light" }
              } else if is_zh {
                "暗室示波"
              } else {
                "Dark"
              }
            }
            a {
              class: "tool-btn",
              href: "https://github.com/dioxuspress/rust-tensor-lens",
              target: "_blank",
              rel: "noopener noreferrer",
              "GitHub"
            }
            a {
              class: "tool-btn",
              href: "https://rustineverything.app/case",
              target: "_blank",
              rel: "noopener noreferrer",
              if is_zh { "返回案例库" } else { "Cases" }
            }
          }
        }

        div { class: "lab-nav-bar",
          div {
            class: "lab-tablist",
            role: "tablist",
            aria_label: if is_zh { "算子实验模块切换" } else { "Interactive operator labs" },
            for tab in LabTab::ALL {
              button {
                key: "{tab.id()}",
                id: "{tab.id()}",
                r#type: "button",
                class: "lab-tab",
                role: "tab",
                aria_selected: if current == tab { "true" } else { "false" },
                tabindex: if current == tab { 0 } else { -1 },
                onclick: move |_| active_tab.set(tab),
                onkeydown: move |evt| match evt.key() {
                  Key::ArrowRight => {
                    evt.prevent_default();
                    active_tab.set(tab.next());
                  }
                  Key::ArrowLeft => {
                    evt.prevent_default();
                    active_tab.set(tab.prev());
                  }
                  Key::Home => {
                    evt.prevent_default();
                    active_tab.set(LabTab::Rope);
                  }
                  Key::End => {
                    evt.prevent_default();
                    active_tab.set(LabTab::Vram);
                  }
                  _ => {}
                },
                "{tab.label(is_zh)}"
              }
            }
          }
        }
      }

      main {
        class: "lens-main",
        role: "tabpanel",
        aria_labelledby: "{current.id()}",
        match current {
          LabTab::Rope => rsx! { RopeLab { zh: is_zh } },
          LabTab::Quant => rsx! { QuantLab { zh: is_zh } },
          LabTab::Bpe => rsx! { BpeLab { zh: is_zh } },
          LabTab::Vram => rsx! { VramLab { zh: is_zh } },
        }
      }

      footer { class: "lens-footer",
        div { class: "lens-footer-inner",
          span {
            if is_zh {
              "所有计算均在浏览器本地 Rust/WASM 实时执行，零后端 API，零数据上传。"
            } else {
              "All math executes locally in your browser via Rust/WASM — zero backend, zero telemetry."
            }
          }
          nav { class: "footer-links", aria_label: "Rust in Everything",
            a { href: "https://rustineverything.app/ai/math", target: "_blank", rel: "noopener noreferrer",
              if is_zh { "AI 数学基础图鉴" } else { "AI Math Atlas" }
            }
            a { href: "https://rustineverything.app/ai/components", target: "_blank", rel: "noopener noreferrer",
              if is_zh { "AI 核心组件图鉴" } else { "AI Components Atlas" }
            }
            a { href: "https://rustineverything.app/course/model-quantization", target: "_blank", rel: "noopener noreferrer",
              if is_zh { "模型量化实战课" } else { "Quantization Course" }
            }
            a { href: "https://rustineverything.app/course/rust-dl-training", target: "_blank", rel: "noopener noreferrer",
              if is_zh { "纯 Rust 深度学习训练" } else { "Rust DL Training" }
            }
          }
        }
      }
    }
  }
}
