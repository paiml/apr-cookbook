//! # Repl Health Status
//!
//! Health-status reporter for the alimentar REPL session.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly. ISBN: 978-1491929124
//!
//! Run with: cargo run --example repl_health_status
//!
//! Migrated from alimentar by PMAT-066 (centralize-cookbooks).

#[cfg(feature = "repl")]
fn main() {
    use alimentar::repl::HealthStatus;

    println!("=== REPL Health Status (Andon) Demo ===\n");

    // Grade to status mapping
    println!("--- Grade Classification ---");
    let grades = ['A', 'B', 'C', 'D', 'F', 'X'];
    for grade in &grades {
        let status = HealthStatus::from_grade(*grade);
        let indicator = status.indicator();
        let desc = match status {
            HealthStatus::Healthy => "Healthy (green)",
            HealthStatus::Warning => "Warning (yellow)",
            HealthStatus::Critical => "Critical (red)",
            HealthStatus::None => "Unknown",
        };
        println!("  Grade '{grade}' => {status:?} {indicator} - {desc}");
    }

    // Visual indicators
    println!("\n--- Visual Indicators ---");
    println!(
        "  Healthy:  '{}'  (empty - all good)",
        HealthStatus::Healthy.indicator()
    );
    println!(
        "  Warning:  '{}'  (single bang - attention needed)",
        HealthStatus::Warning.indicator()
    );
    println!(
        "  Critical: '{}' (double bang - stop the line!)",
        HealthStatus::Critical.indicator()
    );
    println!(
        "  None:     '{}'  (empty - no data)",
        HealthStatus::None.indicator()
    );

    // Simulated quality scores
    println!("\n--- Quality Score Examples ---");
    let scores = [(95, 'A'), (87, 'B'), (72, 'C'), (58, 'D'), (35, 'F')];
    for (score, grade) in &scores {
        let status = HealthStatus::from_grade(*grade);
        let indicator = status.indicator();
        let suffix = if indicator.is_empty() {
            String::new()
        } else {
            format!(" {indicator}")
        };
        println!("  Score: {score:3} | Grade: {grade} | Status: {status:?}{suffix}");
    }

    // Toyota Way principles
    println!("\n--- Toyota Way Andon Principles ---");
    println!("  - Visual management: immediate problem visibility");
    println!("  - Stop-the-line: Critical issues halt the process");
    println!("  - Continuous improvement: Track and improve grades");
    println!("  - Jidoka: Build quality in, don't inspect it out");

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Run with: cargo run --example repl_health_status --features repl");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_runs() {
        main();
    }
}
