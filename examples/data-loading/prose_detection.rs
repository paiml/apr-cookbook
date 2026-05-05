//! # Prose Detection
//!
//! Distinguish prose from code in mixed text corpora.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Allamanis, M. et al. (2018). A Survey of Machine Learning for Big Code and Naturalness. ACM Computing Surveys 51(4). arXiv:1709.06182
//!
//! Run with: cargo run --example prose_detection
//!
//! Migrated from alimentar by PMAT-066 (centralize-cookbooks).

#[cfg(feature = "doctest")]
fn main() {
    use alimentar::doctest::is_prose_continuation;

    let test_cases = [
        // Prose (should return true)
        ("The stdout argument is not allowed.", true),
        ("This function returns a value.", true),
        (":param x: the input value", true),
        ("Note: Use with caution.", true),
        // Code output (should return false)
        ("True", false),
        ("False", false),
        ("None", false),
        ("[1, 2, 3]", false),
        ("{'key': 'value'}", false),
        ("ZeroDivisionError: division by zero", false),
        ("Traceback (most recent call last):", false),
        ("Type >>> to continue", false),
        ("", false),
    ];

    println!("Prose Detection Demo (ALIM-R001 Poka-Yoke)\n");
    println!("{:<45} | Expected | Actual | Status", "Input");
    println!("{}", "-".repeat(75));

    let mut passed = 0;
    let mut failed = 0;

    for (input, expected) in test_cases {
        let actual = is_prose_continuation(input);
        let status = if actual == expected { "✓" } else { "✗" };
        if actual == expected {
            passed += 1;
        } else {
            failed += 1;
        }
        println!(
            "{:<45} | {:<8} | {:<6} | {}",
            if input.len() > 42 {
                format!("{}...", &input[..39])
            } else {
                input.to_string()
            },
            expected,
            actual,
            status
        );
    }

    println!("\nResults: {passed} passed, {failed} failed");
    if failed > 0 {
        std::process::exit(1);
    }
}

#[cfg(not(feature = "doctest"))]
fn main() {
    eprintln!("Run with: cargo run --example prose_detection --features doctest");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_runs() {
        main();
    }
}
