//! # apr ptx-map — Kernel Filter
//!
//! `apr ptx-map <FILE> --kernel <NAME>` shows the layer→PTX mapping
//! filtered to a single kernel. Kernel names follow the convention
//! `Q4KGemv`, `Q4KGemvB256`, `FA3HmaB128xS128`, etc. — a non-trivial
//! naming convention worth a parser. This recipe builds the
//! `(quant_format, kernel_class, [block_size_or_seq_len])` decomposition
//! so a CI pipeline can preview which kernel variants would match a
//! given filter.
//!
//! Demonstrates the **PTXMAP.3** recipe for PMAT-096 (apr ptx-map coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-MAP-001 + Mieruka kernel naming spec
//!
//! Run with: cargo run --example cli_ptx_map_kernel_filter
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelName {
    pub quant: Option<String>,          // "Q4K", "Q5K", "FP16", ...
    pub class: String,                  // "Gemv", "Gemm", "FA3Hma", ...
    pub variant_suffix: Option<String>, // "B256", "B128xS128", ...
}

pub fn parse_kernel_name(s: &str) -> Option<KernelName> {
    if s.is_empty() {
        return None;
    }
    // Quant prefix: "Q\d+K" (k-quant family) or "FP\d+" / "BF\d+".
    let quant_len = if s.starts_with('Q') {
        // Match Q + digits + K, e.g. Q4K, Q5K, Q8_0 (special-cased below).
        let after_q: String = s
            .chars()
            .skip(1)
            .take_while(|c| c.is_ascii_digit() || *c == '_')
            .collect();
        let mut len = 1 + after_q.len();
        if s[len..].starts_with('K') {
            len += 1;
        }
        len
    } else if s.starts_with("FP") || s.starts_with("BF") {
        let prefix_len = 2;
        let nums: String = s
            .chars()
            .skip(prefix_len)
            .take_while(char::is_ascii_digit)
            .collect();
        prefix_len + nums.len()
    } else {
        0
    };

    let quant = if quant_len > 0 {
        Some(s[..quant_len].to_string())
    } else {
        None
    };
    let rest = &s[quant_len..];
    if rest.is_empty() {
        return None;
    }

    // Class: chars until the variant suffix marker ('B' followed by a digit) or end.
    let class_end = rest
        .char_indices()
        .find(|(i, c)| {
            *c == 'B'
                && rest[i + c.len_utf8()..]
                    .chars()
                    .next()
                    .is_some_and(|n| n.is_ascii_digit())
        })
        .map_or(rest.len(), |(i, _)| i);
    let class = rest[..class_end].to_string();
    let suffix = if class_end < rest.len() {
        Some(rest[class_end..].to_string())
    } else {
        None
    };

    Some(KernelName {
        quant,
        class,
        variant_suffix: suffix,
    })
}

pub fn matches_filter(kernel: &str, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    // Exact match on full kernel name OR class match.
    if kernel == filter {
        return true;
    }
    parse_kernel_name(kernel).is_some_and(|k| k.class == filter)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ptx_map_kernel_filter")?;

    let kernels = [
        "Q4KGemv",
        "Q4KGemvB256",
        "Q5KGemv",
        "FP16Gemm",
        "FA3HmaB128xS128",
    ];

    for filter in ["", "Gemv", "Q4KGemv", "FA3Hma", "FP16Gemm"] {
        println!("--kernel {filter:>15}");
        for k in kernels {
            if matches_filter(k, filter) {
                println!("    {k}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parse_q4k_gemv() {
        let k = parse_kernel_name("Q4KGemv").unwrap();
        assert_eq!(k.quant.as_deref(), Some("Q4K"));
        assert_eq!(k.class, "Gemv");
        assert_eq!(k.variant_suffix, None);
    }

    #[test]
    fn parse_q4k_gemv_with_block_suffix() {
        let k = parse_kernel_name("Q4KGemvB256").unwrap();
        assert_eq!(k.quant.as_deref(), Some("Q4K"));
        assert_eq!(k.class, "Gemv");
        assert_eq!(k.variant_suffix.as_deref(), Some("B256"));
    }

    #[test]
    fn empty_string_returns_none() {
        assert!(parse_kernel_name("").is_none());
    }

    #[test]
    fn empty_filter_matches_everything() {
        assert!(matches_filter("Q4KGemv", ""));
        assert!(matches_filter("FP16Gemm", ""));
    }

    #[test]
    fn class_filter_matches_all_quant_variants() {
        assert!(matches_filter("Q4KGemv", "Gemv"));
        assert!(matches_filter("Q5KGemv", "Gemv"));
        // FP16Gemm has class "Gemm", not "Gemv".
        assert!(!matches_filter("FP16Gemm", "Gemv"));
    }

    #[test]
    fn exact_match_takes_precedence() {
        // "Q4KGemv" filter matches only "Q4KGemv", not "Q4KGemvB256".
        assert!(matches_filter("Q4KGemv", "Q4KGemv"));
        assert!(!matches_filter("Q4KGemvB256", "Q4KGemv"));
    }
}
