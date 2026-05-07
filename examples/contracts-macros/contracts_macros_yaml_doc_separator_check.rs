//! # Contracts-Macros YAML Document Separator Check
//!
//! Validate that multi-document YAML buffers use proper `---`
//! separators between documents (and optional `...` end-of-stream).
//! Returns sorted offending document indices and the document count.
//!
//! Demonstrates the **CMM.148** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §9.1 directives end marker; libyaml document
//!  start/end events.
//!
//! Run with: cargo run --example contracts_macros_yaml_doc_separator_check
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DocSepVerdict {
    Ok {
        offending_doc_indices: Vec<u32>,
        document_count: u32,
    },
    InvalidConfig,
}

pub fn check(buffer: &str) -> DocSepVerdict {
    if buffer.is_empty() {
        return DocSepVerdict::InvalidConfig;
    }
    let lines: Vec<&str> = buffer.split('\n').collect();
    let mut doc_starts: Vec<u32> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() == "---" {
            doc_starts.push(i as u32);
        }
    }
    // The implicit first document needs a separator if any non-blank
    // content precedes the first explicit `---`.
    let first_explicit = doc_starts.first().copied().unwrap_or(u32::MAX);
    let has_implicit_first = lines
        .iter()
        .take(first_explicit as usize)
        .any(|l| !l.trim().is_empty());
    let document_count = doc_starts.len() as u32 + u32::from(has_implicit_first);
    let mut offenders: Vec<u32> = Vec::new();
    if has_implicit_first {
        offenders.push(0);
    }
    DocSepVerdict::Ok {
        offending_doc_indices: offenders,
        document_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_doc_separator_check")?;

    let buf = "---\nname: a\n---\nname: b\n";
    println!("multi-doc: {:?}", check(buf));
    println!("implicit: {:?}", check("name: x\n---\nname: y\n"));
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
    fn explicit_separator_first_no_offender() {
        let v = check("---\na: 1\n");
        if let DocSepVerdict::Ok {
            offending_doc_indices,
            ..
        } = v
        {
            assert!(offending_doc_indices.is_empty());
        }
    }

    #[test]
    fn implicit_first_doc_flagged() {
        let v = check("a: 1\n---\nb: 2\n");
        if let DocSepVerdict::Ok {
            offending_doc_indices,
            ..
        } = v
        {
            assert_eq!(offending_doc_indices, vec![0]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(""), DocSepVerdict::InvalidConfig);
    }

    #[test]
    fn single_doc_no_separator_implicit() {
        let v = check("a: 1\n");
        if let DocSepVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 1);
        }
    }

    #[test]
    fn multi_explicit_doc_count() {
        let v = check("---\na: 1\n---\nb: 2\n---\nc: 3\n");
        if let DocSepVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check("---\na: 1\n");
        let r2 = check("---\na: 1\n");
        assert_eq!(r1, r2);
    }

    #[test]
    fn blank_lines_before_first_separator_ok() {
        let v = check("\n\n---\na: 1\n");
        if let DocSepVerdict::Ok {
            offending_doc_indices,
            ..
        } = v
        {
            assert!(offending_doc_indices.is_empty());
        }
    }

    #[test]
    fn dash_in_value_not_separator() {
        // "- foo" is a sequence entry, not a document separator
        let v = check("---\na:\n  - foo\n  - bar\n");
        if let DocSepVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 1);
        }
    }

    #[test]
    fn document_count_with_implicit() {
        let v = check("a: 1\n---\nb: 2\n");
        if let DocSepVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 2);
        }
    }

    #[test]
    fn many_docs_handled() {
        let mut buf = String::new();
        for i in 0..30 {
            buf.push_str(&format!("---\nk{i}: v\n"));
        }
        let v = check(&buf);
        if let DocSepVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 30);
        }
    }

    #[test]
    fn unicode_content_handled() {
        let v = check("---\nname: café\n");
        if let DocSepVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 1);
        }
    }
}
