//! # apr serve --max-tokens — Generation Length Cap
//!
//! `apr serve --max-tokens <N>` caps response length per request.
//! Constraints: ≥ 1 (otherwise no response); ≤ context_window − prompt;
//! defaults to 4096 for chat endpoints. This recipe builds the validator.
//!
//! Demonstrates the **SERVE.4** recipe for PMAT-116 (apr serve coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SERVE-001 + OpenAI API conventions
//!
//! Run with: cargo run --example cli_serve_max_tokens_cap
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CapVerdict {
    Ok,
    InvalidZero,
    ExceedsContextRemaining { available: u32, requested: u32 },
    ExceedsAbsoluteCap,
}

const DEFAULT_MAX: u32 = 4096;
const ABSOLUTE_CEILING: u32 = 1_048_576; // 1 Mtok hard cap

pub fn validate(max_tokens: u32, context_window: u32, prompt_tokens: u32) -> CapVerdict {
    if max_tokens == 0 {
        return CapVerdict::InvalidZero;
    }
    if max_tokens > ABSOLUTE_CEILING {
        return CapVerdict::ExceedsAbsoluteCap;
    }
    let available = context_window.saturating_sub(prompt_tokens);
    if max_tokens > available {
        return CapVerdict::ExceedsContextRemaining {
            available,
            requested: max_tokens,
        };
    }
    CapVerdict::Ok
}

pub fn auto_cap(context_window: u32, prompt_tokens: u32) -> u32 {
    let available = context_window.saturating_sub(prompt_tokens);
    DEFAULT_MAX.min(available)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_serve_max_tokens_cap")?;

    let cases = [
        (4096u32, 8192, 1000),
        (10_000, 8192, 1000),
        (0, 8192, 0),
        (2_000_000, 8192, 0),
    ];
    for (mt, ctx, prompt) in cases {
        println!(
            "max={mt} ctx={ctx} prompt={prompt}  →  {:?}",
            validate(mt, ctx, prompt)
        );
    }
    println!("auto(ctx=8192, prompt=1000) = {}", auto_cap(8192, 1000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_cap_passes() {
        assert_eq!(validate(4096, 8192, 1000), CapVerdict::Ok);
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(validate(0, 8192, 0), CapVerdict::InvalidZero);
    }

    #[test]
    fn exceeds_context_rejected() {
        let v = validate(10_000, 8192, 1000);
        assert!(matches!(v, CapVerdict::ExceedsContextRemaining { .. }));
    }

    #[test]
    fn at_context_boundary_passes() {
        // ctx=8192, prompt=1000 → available=7192. Request 7192 → OK.
        assert_eq!(validate(7192, 8192, 1000), CapVerdict::Ok);
    }

    #[test]
    fn exceeds_absolute_cap_rejected() {
        assert_eq!(validate(2_000_000, 8192, 0), CapVerdict::ExceedsAbsoluteCap);
    }

    #[test]
    fn auto_cap_fits_in_context() {
        let cap = auto_cap(8192, 1000);
        assert_eq!(cap, DEFAULT_MAX); // 4096 ≤ 7192 available
    }

    #[test]
    fn auto_cap_clamps_to_available() {
        // Tiny remaining: 8192 - 7000 = 1192 < default 4096.
        let cap = auto_cap(8192, 7000);
        assert_eq!(cap, 1192);
    }

    #[test]
    fn prompt_exceeds_context_yields_zero_available() {
        // Saturating sub: prompt > ctx → 0.
        let v = validate(1, 1000, 5000);
        assert!(matches!(
            v,
            CapVerdict::ExceedsContextRemaining { available: 0, .. }
        ));
    }
}
