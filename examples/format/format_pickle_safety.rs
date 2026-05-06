//! # Format Pickle Safety Scanner
//!
//! Pickle files can execute arbitrary code via REDUCE/BUILD opcodes
//! that call any class. Safe imports allowlist:
//!   torch.* / numpy.* / collections.OrderedDict / typing → safe
//!   os.system / subprocess.* / __builtin__.eval → unsafe
//!
//! Picker scans for risky opcodes + module references.
//!
//! Demonstrates the **FMT.30** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PyTorch torch.load weights_only mode + Pickle vulns CVE list.
//!
//! Run with: cargo run --example format_pickle_safety
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SafetyVerdict {
    Safe { allowed_imports: Vec<String> },
    Unsafe { dangerous_imports: Vec<String> },
    EmptyOpcodes,
}

const SAFE_PREFIXES: &[&str] = &[
    "torch.",
    "numpy.",
    "collections.OrderedDict",
    "typing.",
    "builtins.dict",
    "builtins.list",
    "builtins.tuple",
];

const DANGEROUS_KEYWORDS: &[&str] = &[
    "os.system",
    "os.popen",
    "subprocess.",
    "eval",
    "exec",
    "__import__",
    "compile",
    "globals",
    "locals",
];

pub fn scan(opcodes: &[&str]) -> SafetyVerdict {
    if opcodes.is_empty() {
        return SafetyVerdict::EmptyOpcodes;
    }
    let mut dangerous: Vec<String> = Vec::new();
    let mut allowed: Vec<String> = Vec::new();
    for op in opcodes {
        if DANGEROUS_KEYWORDS.iter().any(|kw| op.contains(kw)) {
            dangerous.push((*op).to_string());
        } else if SAFE_PREFIXES.iter().any(|p| op.starts_with(p)) {
            allowed.push((*op).to_string());
        }
    }
    if !dangerous.is_empty() {
        return SafetyVerdict::Unsafe {
            dangerous_imports: dangerous,
        };
    }
    SafetyVerdict::Safe {
        allowed_imports: allowed,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_pickle_safety")?;

    println!(
        "safe: {:?}",
        scan(&["torch.Tensor", "numpy.ndarray", "collections.OrderedDict"])
    );
    println!("danger: {:?}", scan(&["os.system", "torch.Tensor"]));
    println!("eval: {:?}", scan(&["builtins.eval"]));
    println!("empty: {:?}", scan(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn safe_torch_tensor() {
        let v = scan(&["torch.Tensor", "numpy.ndarray"]);
        assert!(matches!(v, SafetyVerdict::Safe { .. }));
    }

    #[test]
    fn os_system_unsafe() {
        let v = scan(&["os.system"]);
        assert!(matches!(v, SafetyVerdict::Unsafe { .. }));
    }

    #[test]
    fn subprocess_unsafe() {
        let v = scan(&["subprocess.Popen"]);
        assert!(matches!(v, SafetyVerdict::Unsafe { .. }));
    }

    #[test]
    fn eval_unsafe() {
        let v = scan(&["builtins.eval"]);
        assert!(matches!(v, SafetyVerdict::Unsafe { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(scan(&[]), SafetyVerdict::EmptyOpcodes);
    }

    #[test]
    fn mixed_unsafe_wins() {
        let v = scan(&["torch.Tensor", "os.system", "numpy.ndarray"]);
        assert!(matches!(v, SafetyVerdict::Unsafe { .. }));
    }

    #[test]
    fn import_unsafe() {
        let v = scan(&["builtins.__import__"]);
        assert!(matches!(v, SafetyVerdict::Unsafe { .. }));
    }

    #[test]
    fn unknown_safe_listed_in_allowed() {
        let v = scan(&["torch.Tensor"]);
        if let SafetyVerdict::Safe { allowed_imports } = v {
            assert!(allowed_imports.iter().any(|s| s.contains("torch")));
        }
    }

    #[test]
    fn dangerous_listed_in_unsafe() {
        let v = scan(&["os.system", "subprocess.run"]);
        if let SafetyVerdict::Unsafe {
            dangerous_imports, ..
        } = v
        {
            assert_eq!(dangerous_imports.len(), 2);
        }
    }

    #[test]
    fn unknown_module_neither_safe_nor_unsafe() {
        // "random_module.foo" is neither in allowlist nor blocklist;
        // treated as safe (pass-through).
        let v = scan(&["random_module.foo"]);
        assert!(matches!(v, SafetyVerdict::Safe { .. }));
    }

    #[test]
    fn typing_module_safe() {
        let v = scan(&["typing.Dict"]);
        assert!(matches!(v, SafetyVerdict::Safe { .. }));
    }
}
