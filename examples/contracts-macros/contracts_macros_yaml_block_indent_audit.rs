//! # Contracts-Macros YAML Block Indent Audit
//!
//! Check that block-style YAML uses uniform indent multiples.
//! Returns sorted offending line numbers and the consensus indent.
//!
//! Demonstrates the **CMM.163** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint `indentation: spaces`; PyYAML uniform-indent
//!  recommendation.
//!
//! Run with: cargo run --example contracts_macros_yaml_block_indent_audit
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BlockIndentVerdict {
    Ok {
        offending_lines: Vec<u32>,
        consensus_indent: u32,
    },
    InvalidConfig,
}

pub fn check(buffer: &str) -> BlockIndentVerdict {
    if buffer.is_empty() {
        return BlockIndentVerdict::InvalidConfig;
    }
    let mut indents: Vec<u32> = Vec::new();
    for line in buffer.split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        let leading = line.chars().take_while(|c| *c == ' ').count() as u32;
        if leading > 0 {
            indents.push(leading);
        }
    }
    if indents.is_empty() {
        return BlockIndentVerdict::Ok {
            offending_lines: vec![],
            consensus_indent: 0,
        };
    }
    // Consensus: smallest non-zero indent (assume base block-step).
    let consensus = *indents.iter().min().unwrap_or(&0);
    let mut offenders: Vec<u32> = Vec::new();
    for (i, line) in buffer.split('\n').enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let leading = line.chars().take_while(|c| *c == ' ').count() as u32;
        if leading > 0 && leading % consensus != 0 {
            offenders.push((i as u32) + 1);
        }
    }
    offenders.sort_unstable();
    BlockIndentVerdict::Ok {
        offending_lines: offenders,
        consensus_indent: consensus,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_block_indent_audit")?;

    let buf = "key:\n  child:\n    deep: 1\n   wrong: 2\n";
    println!("check: {:?}", check(buf));
    println!("invalid: {:?}", check(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn uniform_indent_no_offender() {
        let v = check("k:\n  a: 1\n  b: 2\n");
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn deviant_indent_flagged() {
        // Consensus = min indent (2). Line with 3-space indent breaks it.
        let v = check("k:\n  a: 1\n   b: 2\n");
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![3]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(""), BlockIndentVerdict::InvalidConfig);
    }

    #[test]
    fn no_indented_lines_zero_consensus() {
        let v = check("a: 1\nb: 2\n");
        if let BlockIndentVerdict::Ok {
            consensus_indent, ..
        } = v
        {
            assert_eq!(consensus_indent, 0);
        }
    }

    #[test]
    fn consensus_smallest_indent() {
        let v = check("k:\n  a: 1\n    b: 2\n");
        if let BlockIndentVerdict::Ok {
            consensus_indent, ..
        } = v
        {
            assert_eq!(consensus_indent, 2);
        }
    }

    #[test]
    fn multiples_of_consensus_ok() {
        let v = check("k:\n  a:\n    b: 1\n");
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check("k:\n  a: 1\n");
        let r2 = check("k:\n  a: 1\n");
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_lines_skipped() {
        let v = check("k:\n  a: 1\n\n  b: 2\n");
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn lines_sorted() {
        let v = check("k:\n  a: 1\n   b: 2\n   c: 3\n");
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            for w in offending_lines.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn many_lines_handled() {
        let mut buf = String::new();
        for _ in 0..30 {
            buf.push_str("  a: 1\n");
        }
        let v = check(&buf);
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn unicode_value_handled() {
        let v = check("k:\n  a: café\n");
        if let BlockIndentVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }
}
