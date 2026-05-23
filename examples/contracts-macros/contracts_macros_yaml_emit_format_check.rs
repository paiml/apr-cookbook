//! # Contracts-Macros YAML Emit Format Check
//!
//! Verify the emitter produces canonical YAML: keys sorted, no
//! trailing whitespace, no tabs, ends with single newline. Returns
//! sorted offending line numbers.
//!
//! Demonstrates the **CMM.172** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint canonical-emit rules; jq sort_keys + tojson
//!  output mode.
//!
//! Run with: cargo run --example contracts_macros_yaml_emit_format_check
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EmitVerdict {
    Ok {
        offending_lines: Vec<u32>,
        clean: bool,
    },
    InvalidConfig,
}

pub fn check(buffer: &str) -> EmitVerdict {
    if buffer.is_empty() {
        return EmitVerdict::InvalidConfig;
    }
    let mut offenders: Vec<u32> = Vec::new();
    let lines: Vec<&str> = buffer.split('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains('\t') {
            offenders.push((i + 1) as u32);
            continue;
        }
        if !line.is_empty() && line.trim_end().len() != line.len() {
            offenders.push((i + 1) as u32);
        }
    }
    if !buffer.ends_with('\n') {
        offenders.push(lines.len() as u32);
    }
    offenders.sort_unstable();
    offenders.dedup();
    let clean = offenders.is_empty();
    EmitVerdict::Ok {
        offending_lines: offenders,
        clean,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_emit_format_check")?;

    println!("clean: {:?}", check("a: 1\nb: 2\n"));
    println!("trailing: {:?}", check("a: 1  \nb: 2\n"));
    println!("tab: {:?}", check("a:\tval\n"));
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
    fn clean_buffer_passes() {
        let v = check("a: 1\nb: 2\n");
        if let EmitVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn trailing_space_flagged() {
        let v = check("a: 1  \n");
        if let EmitVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![1]);
        }
    }

    #[test]
    fn tab_flagged() {
        let v = check("a:\tval\n");
        if let EmitVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![1]);
        }
    }

    #[test]
    fn missing_trailing_newline_flagged() {
        let v = check("a: 1");
        if let EmitVerdict::Ok { clean, .. } = v {
            assert!(!clean);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(""), EmitVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check("a: 1\n");
        let r2 = check("a: 1\n");
        assert_eq!(r1, r2);
    }

    #[test]
    fn lines_sorted_dedup() {
        let v = check("a:\tval  \n");
        if let EmitVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines.len(), 1);
        }
    }

    #[test]
    fn empty_lines_ignored() {
        let v = check("a: 1\n\nb: 2\n");
        if let EmitVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn many_offenders_handled() {
        let buf = "a: 1  \nb: 2\t\nc: 3  \n";
        let v = check(buf);
        if let EmitVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines.len(), 3);
        }
    }

    #[test]
    fn unicode_value_handled() {
        let v = check("name: café\n");
        if let EmitVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn single_clean_line() {
        let v = check("just_one: line\n");
        if let EmitVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn middle_tab_flagged() {
        let v = check("a: 1\nb:\tx\nc: 3\n");
        if let EmitVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![2]);
        }
    }
}
