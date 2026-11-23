use dioxus::prelude::*;

#[component]
pub fn RangeControl(
  id: String,
  label: String,
  value_text: String,
  min: f64,
  max: f64,
  step: f64,
  value: f64,
  #[props(default)] hint: Option<String>,
  on_change: EventHandler<f64>,
) -> Element {
  let inline_style = format!("--attr-min:{min};--attr-max:{max};--control-value:{value};");
  rsx! {
    div { class: "control-group",
      div { class: "control-header",
        label { class: "control-label", r#for: "{id}", "{label}" }
        span { class: "control-value", "{value_text}" }
      }
      input {
        id: "{id}",
        class: "lens-slider",
        r#type: "range",
        min: "{min}",
        max: "{max}",
        step: "{step}",
        value: "{value}",
        aria_valuetext: "{value_text}",
        style: "{inline_style}",
        oninput: move |evt| {
          if let Ok(parsed) = evt.value().parse::<f64>() {
            on_change.call(parsed);
          }
        },
      }
      if let Some(note) = hint {
        p { class: "control-hint", "{note}" }
      }
    }
  }
}

#[derive(Clone, PartialEq)]
pub struct MetricItem {
  pub label: String,
  pub value: String,
  pub sub: String,
  pub tone: &'static str,
}

#[component]
pub fn MetricStrip(items: Vec<MetricItem>) -> Element {
  rsx! {
    div { class: "metric-strip-wrap",
      div { class: "metric-strip",
        for item in items {
          div { class: "metric-cell",
            div { class: "metric-label", "{item.label}" }
            div { class: "metric-value {item.tone}", "{item.value}" }
            div { class: "metric-sub", "{item.sub}" }
          }
        }
      }
    }
  }
}
