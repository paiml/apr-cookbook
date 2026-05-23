//! # Contracts-Macros YAML Sequence Padding Normalize
//!
//! Verify that all "- item" sequence entries share the same indent
//! depth. Returns sorted offending lines and the consensus indent.
//!
//! Demonstrates the **CMM.147** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint `indentation: indent-sequences`; Ansible coding
//!  conventions on sequence-indent style.
//!
//! Run with: cargo run --example contracts_macros_yaml_seq_padding_norm
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum SeqPadVerdict {
    Ok {
        offending_lines: Vec<u32>,
        consensus_indent: u32,
    },
    InvalidConfig,
}

pub fn check(buffer: &str) -> SeqPadVerdict {
    if buffer.is_empty() {
        return SeqPadVerdict::InvalidConfig;
    }
    let mut indent_count: BTreeMap<u32, u32> = BTreeMap::new();
    let mut seq_lines: Vec<(u32, u32)> = Vec::new(); // (line_no, indent)
    for (i, line) in buffer.split('\n').enumerate() {
        let leading: u32 = line.chars().take_while(|c| *c == ' ').count() as u32;
        let trimmed = line.trim_start();
        if trimmed.starts_with("- ") || trimmed == "-" {
            *indent_count.entry(leading).or_insert(0) += 1;
            seq_lines.push((i as u32 + 1, leading));
        }
    }
    if seq_lines.is_empty() {
        return SeqPadVerdict::Ok {
            offending_lines: vec![],
            consensus_indent: 0,
        };
    }
    // Consensus = most common indent.
    let consensus = *indent_count
        .iter()
        .max_by_key(|(_, c)| *c)
        .map_or(&0, |(k, _)| k);
    let mut offenders: Vec<u32> = seq_lines
        .iter()
        .filter(|(_, ind)| *ind != consensus)
        .map(|(line, _)| *line)
        .collect();
    offenders.sort_unstable();
    SeqPadVerdict::Ok {
        offending_lines: offenders,
        consensus_indent: consensus,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_seq_padding_norm")?;

    let buf = "items:\n  - a\n  - b\n   - c\n";
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
    fn uniform_seq_no_offender() {
        let v = check("items:\n  - a\n  - b\n");
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn deviant_indent_flagged() {
        let v = check("items:\n  - a\n  - b\n   - c\n");
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            // Line 4 has 3-space indent while consensus is 2.
            assert_eq!(offending_lines, vec![4]);
        }
    }

    #[test]
    fn consensus_indent_majority() {
        let v = check("  - a\n  - b\n  - c\n    - d\n");
        if let SeqPadVerdict::Ok {
            consensus_indent, ..
        } = v
        {
            assert_eq!(consensus_indent, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(""), SeqPadVerdict::InvalidConfig);
    }

    #[test]
    fn no_seq_no_offender() {
        let v = check("not_a_seq:\n  key: val\n");
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn dash_alone_recognized() {
        let v = check("  -\n  - a\n");
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check("  - a\n");
        let r2 = check("  - a\n");
        assert_eq!(r1, r2);
    }

    #[test]
    fn lines_sorted() {
        let v = check("  - a\n   - b\n   - c\n");
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            // Consensus = 3 spaces (2 lines vs 1 line). Line 1 (2-space) flagged.
            assert_eq!(offending_lines, vec![1]);
        }
    }

    #[test]
    fn many_lines_handled() {
        let mut buf = String::new();
        for _ in 0..30 {
            buf.push_str("  - x\n");
        }
        let v = check(&buf);
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn nested_seq_at_same_depth() {
        let v = check("    - a\n    - b\n");
        if let SeqPadVerdict::Ok {
            consensus_indent, ..
        } = v
        {
            assert_eq!(consensus_indent, 4);
        }
    }

    #[test]
    fn unicode_buffer_handled() {
        let v = check("  - café\n  - résumé\n");
        if let SeqPadVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }
}
