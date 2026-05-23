//! # Monitoring Dropped-Request Classifier
//!
//! Classify a failed inference request into a category to drive
//! remediation:
//!   Timeout → reduce model size or scale up
//!   OutOfMemory → trim KV cache or move to bigger GPU
//!   NetworkReset → check upstream health
//!   ClientCanceled → not actionable, just log
//!   ServerCrash → page oncall
//!   InvalidInput → return 4xx, no remediation
//!
//! Demonstrates the **MON.21** recipe for PMAT-140 (monitoring round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda dropped-invocation taxonomy.
//!
//! Run with: cargo run --example monitor_dropped_request_classifier
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropCause {
    Timeout,
    OutOfMemory,
    NetworkReset,
    ClientCanceled,
    ServerCrash,
    InvalidInput,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Remediation {
    ScaleUp,
    TrimKvCache,
    CheckUpstream,
    LogOnly,
    PageOncall,
    Return4xx,
    Investigate,
}

#[derive(Debug, PartialEq)]
pub enum DropVerdict {
    Ok {
        cause: DropCause,
        remediation: Remediation,
    },
    InvalidStatusCode,
}

pub fn classify(http_status: u16, error_message: &str, latency_ms: u64) -> DropVerdict {
    if http_status == 0 || http_status > 599 {
        return DropVerdict::InvalidStatusCode;
    }
    let lower = error_message.to_ascii_lowercase();
    let cause = if (http_status == 504 || lower.contains("timeout")) && latency_ms > 0 {
        DropCause::Timeout
    } else if lower.contains("out of memory") || lower.contains("oom") {
        DropCause::OutOfMemory
    } else if lower.contains("connection reset") || lower.contains("econnreset") {
        DropCause::NetworkReset
    } else if lower.contains("cancelled") || lower.contains("canceled") || http_status == 499 {
        DropCause::ClientCanceled
    } else if http_status == 500 && (lower.contains("panic") || lower.contains("crash")) {
        DropCause::ServerCrash
    } else if http_status == 400 || http_status == 422 {
        DropCause::InvalidInput
    } else {
        DropCause::Unknown
    };
    let remediation = match cause {
        DropCause::Timeout => Remediation::ScaleUp,
        DropCause::OutOfMemory => Remediation::TrimKvCache,
        DropCause::NetworkReset => Remediation::CheckUpstream,
        DropCause::ClientCanceled => Remediation::LogOnly,
        DropCause::ServerCrash => Remediation::PageOncall,
        DropCause::InvalidInput => Remediation::Return4xx,
        DropCause::Unknown => Remediation::Investigate,
    };
    DropVerdict::Ok { cause, remediation }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_dropped_request_classifier")?;

    println!(
        "504 timeout: {:?}",
        classify(504, "request timeout", 30_000)
    );
    println!("OOM: {:?}", classify(500, "CUDA out of memory", 1_000));
    println!(
        "client cancel: {:?}",
        classify(499, "client closed connection", 50)
    );
    println!("invalid: {:?}", classify(0, "", 0));
    println!("400 bad request: {:?}", classify(400, "missing field", 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn timeout_504_classified() {
        let v = classify(504, "request timeout", 30_000);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::Timeout);
        }
    }

    #[test]
    fn timeout_message_classified() {
        let v = classify(500, "operation timeout", 30_000);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::Timeout);
        }
    }

    #[test]
    fn oom_classified() {
        let v = classify(500, "CUDA out of memory", 1_000);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::OutOfMemory);
        }
    }

    #[test]
    fn network_reset_classified() {
        let v = classify(502, "ECONNRESET", 5);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::NetworkReset);
        }
    }

    #[test]
    fn client_cancel_499_classified() {
        let v = classify(499, "client cancelled", 5);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::ClientCanceled);
        }
    }

    #[test]
    fn server_crash_classified() {
        let v = classify(500, "thread panic", 100);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::ServerCrash);
        }
    }

    #[test]
    fn invalid_input_400_classified() {
        let v = classify(400, "missing field", 5);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::InvalidInput);
        }
    }

    #[test]
    fn unknown_classified() {
        let v = classify(503, "service unavailable", 10);
        if let DropVerdict::Ok { cause, .. } = v {
            assert_eq!(cause, DropCause::Unknown);
        }
    }

    #[test]
    fn invalid_status_zero_rejected() {
        assert_eq!(classify(0, "", 0), DropVerdict::InvalidStatusCode);
    }

    #[test]
    fn invalid_status_above_599_rejected() {
        assert_eq!(classify(600, "", 0), DropVerdict::InvalidStatusCode);
    }

    #[test]
    fn timeout_remediation_scale_up() {
        if let DropVerdict::Ok { remediation, .. } = classify(504, "timeout", 30_000) {
            assert_eq!(remediation, Remediation::ScaleUp);
        }
    }

    #[test]
    fn oom_remediation_trim_cache() {
        if let DropVerdict::Ok { remediation, .. } = classify(500, "OOM", 1_000) {
            assert_eq!(remediation, Remediation::TrimKvCache);
        }
    }
}
