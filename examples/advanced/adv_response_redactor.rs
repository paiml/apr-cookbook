//! # Advanced LLM Response Redactor
//!
//! Before returning to user, redact known-secret patterns from LLM
//! response: API keys, AWS access keys, JWTs, OAuth tokens, etc.
//!
//! Demonstrates the **ADV.27** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TruffleHog secret-detection patterns.
//!
//! Run with: cargo run --example adv_response_redactor
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RedactVerdict {
    Ok { redacted: String, secret_count: u32 },
    EmptyResponse,
}

pub fn redact(response: &str) -> RedactVerdict {
    if response.is_empty() {
        return RedactVerdict::EmptyResponse;
    }
    let mut secret_count = 0u32;
    let mut out = String::with_capacity(response.len());
    let mut first = true;
    for word in response.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        let trimmed = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if is_aws_key(trimmed) {
            out.push_str("[REDACTED-aws-key]");
            secret_count += 1;
        } else if is_jwt(trimmed) {
            out.push_str("[REDACTED-jwt]");
            secret_count += 1;
        } else if is_github_token(trimmed) {
            out.push_str("[REDACTED-github-token]");
            secret_count += 1;
        } else if is_openai_key(trimmed) {
            out.push_str("[REDACTED-openai-key]");
            secret_count += 1;
        } else {
            out.push_str(word);
        }
    }
    RedactVerdict::Ok {
        redacted: out,
        secret_count,
    }
}

fn is_aws_key(s: &str) -> bool {
    s.starts_with("AKIA")
        && s.len() == 20
        && s.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn is_jwt(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        })
}

fn is_github_token(s: &str) -> bool {
    (s.starts_with("ghp_") || s.starts_with("gho_") || s.starts_with("ghs_"))
        && s.len() >= 36
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_openai_key(s: &str) -> bool {
    s.starts_with("sk-")
        && s.len() >= 20
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_response_redactor")?;

    println!(
        "AWS key: {:?}",
        redact("Found key AKIAIOSFODNN7EXAMPLE in code")
    );
    println!(
        "JWT: {:?}",
        redact("token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOjF9.signature_xx_yy")
    );
    println!(
        "GitHub: {:?}",
        redact("export GH=ghp_1234567890abcdef1234567890abcdef1234")
    );
    println!("OpenAI: {:?}", redact("api: sk-proj-abc123xyz456"));
    println!("clean: {:?}", redact("nothing here"));
    println!("empty: {:?}", redact(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redactor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn aws_key_redacted() {
        let v = redact("AKIAIOSFODNN7EXAMPLE in code");
        if let RedactVerdict::Ok {
            redacted,
            secret_count,
        } = v
        {
            assert!(redacted.contains("[REDACTED-aws-key]"));
            assert_eq!(secret_count, 1);
        }
    }

    #[test]
    fn jwt_redacted() {
        let v = redact("eyJhbGc.eyJzdWI.sigxxyy");
        if let RedactVerdict::Ok { redacted, .. } = v {
            assert!(redacted.contains("[REDACTED-jwt]"));
        }
    }

    #[test]
    fn github_token_redacted() {
        let v = redact("ghp_1234567890abcdef1234567890abcdef1234");
        if let RedactVerdict::Ok { redacted, .. } = v {
            assert!(redacted.contains("[REDACTED-github-token]"));
        }
    }

    #[test]
    fn openai_key_redacted() {
        let v = redact("sk-proj-abc123xyz456");
        if let RedactVerdict::Ok { redacted, .. } = v {
            assert!(redacted.contains("[REDACTED-openai-key]"));
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(redact(""), RedactVerdict::EmptyResponse);
    }

    #[test]
    fn no_secrets_unchanged_count() {
        let v = redact("hello world");
        if let RedactVerdict::Ok { secret_count, .. } = v {
            assert_eq!(secret_count, 0);
        }
    }

    #[test]
    fn multiple_secrets_all_redacted() {
        let v = redact("AKIAIOSFODNN7EXAMPLE and ghp_1234567890abcdef1234567890abcdef1234");
        if let RedactVerdict::Ok { secret_count, .. } = v {
            assert_eq!(secret_count, 2);
        }
    }

    #[test]
    fn short_aws_pattern_not_redacted() {
        // Only 19 chars after AKIA → not AWS format.
        let v = redact("AKIAIOSFODNN7EXAMPL");
        if let RedactVerdict::Ok { secret_count, .. } = v {
            assert_eq!(secret_count, 0);
        }
    }

    #[test]
    fn jwt_two_dots_required() {
        // Only one dot → not JWT.
        let v = redact("foo.bar");
        if let RedactVerdict::Ok { secret_count, .. } = v {
            assert_eq!(secret_count, 0);
        }
    }

    #[test]
    fn short_github_token_not_redacted() {
        // Too short → not GH format.
        let v = redact("ghp_short");
        if let RedactVerdict::Ok { secret_count, .. } = v {
            assert_eq!(secret_count, 0);
        }
    }

    #[test]
    fn ghs_prefix_recognized() {
        let v = redact("ghs_1234567890abcdef1234567890abcdef1234");
        if let RedactVerdict::Ok { secret_count, .. } = v {
            assert_eq!(secret_count, 1);
        }
    }
}
