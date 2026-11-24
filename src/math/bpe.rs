use std::collections::HashMap;

/// Ordered merge rules forming a compact byte-level BPE vocabulary covering
/// English subwords, Rust syntax, JSON tokens, and common Chinese UTF-8 morphemes.
const MERGE_SPECS: &[&str] = &[
  // Common English & code bigrams/subwords
  "in",
  "en",
  "er",
  "re",
  "on",
  "an",
  "at",
  "es",
  "or",
  "te",
  "ti",
  "ar",
  "st",
  "al",
  "le",
  "co",
  "de",
  "se",
  "ro",
  "to",
  "nt",
  "ng",
  "nd",
  "ed",
  "is",
  "it",
  "io",
  "ou",
  "ra",
  "ri",
  "ve",
  "tr",
  "th",
  "he",
  "the",
  "ing",
  "tion",
  "ent",
  "for",
  "and",
  "con",
  "pro",
  "com",
  "pre",
  "sub",
  "vec",
  "the ",
  "in ",
  "to ",
  "of ",
  "and ",
  "for ",
  "with ",
  "from ",
  "that ",
  "this ",
  // Rust syntax & ML domain tokens
  "fn",
  "fn ",
  "pub",
  "pub ",
  "let",
  "let ",
  "mut",
  "mut ",
  "impl",
  "impl ",
  "use",
  "use ",
  "struct",
  "struct ",
  "enum",
  "enum ",
  "match",
  "match ",
  "async",
  "async ",
  "await",
  "return",
  "self",
  "Self",
  "Result",
  "Option",
  "Vec",
  "String",
  "Signal",
  "Tensor",
  "RoPE",
  "YaRN",
  "NF4",
  "INT4",
  "INT8",
  "KV",
  "Cache",
  "VRAM",
  "WASM",
  "Rust",
  "Dioxus",
  "Burn",
  "Candle",
  "Token",
  "token",
  "quant",
  "scale",
  "block",
  "layer",
  "model",
  "head",
  "attn",
  "norm",
  "softmax",
  "linear",
  "matrix",
  "weight",
  "batch",
  "context",
  "length",
  "memory",
  "kernel",
  "fusion",
  "zero",
  "copy",
  "safe",
  "fast",
  // Punctuation & formatting clusters
  "::",
  "->",
  "=>",
  "==",
  "!=",
  "<=",
  ">=",
  "  ",
  "    ",
  "\": ",
  "\",",
  "{\"",
  "\"}",
  "// ",
  "()",
  "{}",
  "[]",
  // Frequent Chinese UTF-8 3-byte characters & 6-byte AI/Rust terms
  "的",
  "是",
  "在",
  "和",
  "与",
  "为",
  "中",
  "大",
  "模",
  "型",
  "量",
  "化",
  "张",
  "注",
  "意",
  "力",
  "推",
  "理",
  "显",
  "存",
  "分",
  "词",
  "压",
  "缩",
  "精",
  "度",
  "位",
  "置",
  "编",
  "码",
  "外",
  "推",
  "上",
  "下",
  "文",
  "窗",
  "口",
  "算",
  "子",
  "融",
  "合",
  "零",
  "拷",
  "贝",
  "内",
  "核",
  "并",
  "发",
  "线",
  "程",
  "安",
  "全",
  "实",
  "时",
  "计",
  "算",
  "模型",
  "量化",
  "张量",
  "注意力",
  "推理",
  "显存",
  "分词",
  "压缩",
  "精度",
  "位置编码",
  "上下文",
  "算子",
  "融合",
  "零拷贝",
  "内核",
  "并发",
  "实时",
  "计算",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeToken {
  pub id: u32,
  pub bytes: Vec<u8>,
  pub display: String,
  pub is_merged: bool,
  pub is_valid_utf8: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeStep {
  pub step_index: usize,
  pub rank: u32,
  pub left_display: String,
  pub right_display: String,
  pub merged_display: String,
  pub merged_id: u32,
  pub occurrences: usize,
  pub tokens_after: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BpeAnalysis {
  pub byte_count: usize,
  pub char_count: usize,
  pub final_tokens: Vec<BpeToken>,
  pub trace_snapshots: Vec<Vec<BpeToken>>,
  pub merge_steps: Vec<MergeStep>,
  pub bytes_per_token: f64,
  pub chars_per_token: f64,
  pub compression_pct: f64,
}

struct BpeVocab {
  merges: HashMap<(u32, u32), (u32, u32)>, // (left, right) -> (rank, merged_id)
  id_to_bytes: Vec<Vec<u8>>,
}

fn build_vocab() -> BpeVocab {
  let mut id_to_bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
  let mut bytes_to_id: HashMap<Vec<u8>, u32> =
    (0..=255u8).map(|b| (vec![b], u32::from(b))).collect();
  let mut merges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
  let mut next_rank = 0u32;

  for &spec in MERGE_SPECS {
    let raw = spec.as_bytes();
    if raw.len() < 2 {
      continue;
    }

    // Greedily express `spec` with already-registered symbols so every multi-byte
    // token is reachable via binary merges from raw bytes.
    let mut seq: Vec<u32> = raw.iter().map(|&b| u32::from(b)).collect();
    loop {
      let mut best: Option<(usize, u32, u32)> = None;
      for i in 0..seq.len().saturating_sub(1) {
        if let Some(&(rank, merged_id)) = merges.get(&(seq[i], seq[i + 1]))
          && best.is_none_or(|(_, best_rank, _)| rank < best_rank)
        {
          best = Some((i, rank, merged_id));
        }
      }
      if let Some((pos, _, merged_id)) = best {
        seq[pos] = merged_id;
        seq.remove(pos + 1);
      } else {
        break;
      }
    }

    // Register left-to-right binary merges until the full `spec` is a single token.
    while seq.len() > 1 {
      let pair = (seq[0], seq[1]);
      let merged_id = if let Some(&(_, existing)) = merges.get(&pair) {
        existing
      } else {
        let mut combined = id_to_bytes[pair.0 as usize].clone();
        combined.extend_from_slice(&id_to_bytes[pair.1 as usize]);
        let id = if let Some(&existing_id) = bytes_to_id.get(&combined) {
          existing_id
        } else {
          let new_id = id_to_bytes.len() as u32;
          id_to_bytes.push(combined.clone());
          bytes_to_id.insert(combined, new_id);
          new_id
        };
        merges.insert(pair, (next_rank, id));
        next_rank += 1;
        id
      };
      seq[0] = merged_id;
      seq.remove(1);
    }
  }

  BpeVocab {
    merges,
    id_to_bytes,
  }
}

pub fn format_token_bytes(bytes: &[u8]) -> (String, bool) {
  match std::str::from_utf8(bytes) {
    Ok(text) => {
      let mut out = String::with_capacity(text.len());
      for ch in text.chars() {
        match ch {
          ' ' => out.push('·'),
          '\n' => out.push('↵'),
          '\t' => out.push('⇥'),
          c if c.is_control() => out.push_str(&format!("\\x{:02X}", c as u32)),
          c => out.push(c),
        }
      }
      (out, true)
    }
    Err(_) => {
      let hex = bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ");
      (format!("<0x{hex}>"), false)
    }
  }
}

fn ids_to_tokens(ids: &[u32], vocab: &BpeVocab) -> Vec<BpeToken> {
  ids
    .iter()
    .map(|&id| {
      let bytes = vocab
        .id_to_bytes
        .get(id as usize)
        .cloned()
        .unwrap_or_default();
      let (display, is_valid_utf8) = format_token_bytes(&bytes);
      BpeToken {
        id,
        bytes,
        display,
        is_merged: id >= 256,
        is_valid_utf8,
      }
    })
    .collect()
}

pub fn encode_bpe(input: &str) -> BpeAnalysis {
  let vocab = build_vocab();
  let raw_bytes = input.as_bytes();
  let byte_count = raw_bytes.len();
  let char_count = input.chars().count();

  let mut ids: Vec<u32> = raw_bytes.iter().map(|&b| u32::from(b)).collect();
  let mut trace_snapshots = vec![ids_to_tokens(&ids, &vocab)];
  let mut merge_steps = Vec::new();

  // Standard BPE loop: always apply the lowest-rank available pair merge across the sequence.
  while ids.len() > 1 {
    let mut best_pair: Option<((u32, u32), u32, u32)> = None;
    for window in ids.windows(2) {
      let pair = (window[0], window[1]);
      if let Some(&(rank, merged_id)) = vocab.merges.get(&pair)
        && best_pair.is_none_or(|(_, best_rank, _)| rank < best_rank)
      {
        best_pair = Some((pair, rank, merged_id));
      }
    }

    let Some((pair, rank, merged_id)) = best_pair else {
      break;
    };

    let mut next_ids = Vec::with_capacity(ids.len());
    let mut i = 0usize;
    let mut occurrences = 0usize;
    while i < ids.len() {
      if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
        next_ids.push(merged_id);
        occurrences += 1;
        i += 2;
      } else {
        next_ids.push(ids[i]);
        i += 1;
      }
    }

    let (left_display, _) = format_token_bytes(&vocab.id_to_bytes[pair.0 as usize]);
    let (right_display, _) = format_token_bytes(&vocab.id_to_bytes[pair.1 as usize]);
    let (merged_display, _) = format_token_bytes(&vocab.id_to_bytes[merged_id as usize]);

    ids = next_ids;
    merge_steps.push(MergeStep {
      step_index: merge_steps.len() + 1,
      rank,
      left_display,
      right_display,
      merged_display,
      merged_id,
      occurrences,
      tokens_after: ids.len(),
    });
    trace_snapshots.push(ids_to_tokens(&ids, &vocab));
  }

  let final_tokens = trace_snapshots.last().cloned().unwrap_or_default();
  let tok_count = final_tokens.len().max(1) as f64;
  let bytes_per_token = if byte_count == 0 {
    0.0
  } else {
    byte_count as f64 / tok_count
  };
  let chars_per_token = if char_count == 0 {
    0.0
  } else {
    char_count as f64 / tok_count
  };
  let compression_pct = if byte_count == 0 {
    0.0
  } else {
    (1.0 - final_tokens.len() as f64 / byte_count as f64) * 100.0
  };

  BpeAnalysis {
    byte_count,
    char_count,
    final_tokens,
    trace_snapshots,
    merge_steps,
    bytes_per_token,
    chars_per_token,
    compression_pct,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lossless_byte_roundtrip_for_mixed_rust_and_chinese() {
    let sample = "pub async fn rope_scale(ctx: usize) -> Result<f64> // 大模型量化与张量注意力";
    let res = encode_bpe(sample);
    let reconstructed: Vec<u8> = res
      .final_tokens
      .iter()
      .flat_map(|t| t.bytes.iter().copied())
      .collect();
    assert_eq!(reconstructed, sample.as_bytes());
    assert!(res.final_tokens.len() < res.byte_count);
    assert_eq!(res.trace_snapshots.len(), res.merge_steps.len() + 1);
  }

  #[test]
  fn empty_input_returns_zero_tokens() {
    let res = encode_bpe("");
    assert_eq!(res.byte_count, 0);
    assert!(res.final_tokens.is_empty());
    assert!(res.merge_steps.is_empty());
  }
}
