//! # Registry Pull-Secret Credential Type Validator
//!
//! Pull secrets come in formats:
//!   DockerConfigJson: classic ~/.docker/config.json
//!   ServiceAccountToken: Kubernetes JWT
//!   AwsEcrLogin: ECR authorization token (12h validity)
//!   GcpServiceAccount: GCP key.json file
//!   AzureMsi: Azure managed-identity OAuth token
//!
//! Picker classifies + validates expiry.
//!
//! Demonstrates the **REG.23** recipe for PMAT-150 (registry round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kubernetes ImagePullSecret + cloud-provider auth docs.
//!
//! Run with: cargo run --example registry_pull_secret_validator
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretKind {
    DockerConfigJson,
    ServiceAccountToken,
    AwsEcrLogin,
    GcpServiceAccount,
    AzureMsi,
}

#[derive(Debug, PartialEq)]
pub enum SecretVerdict {
    Valid {
        kind: SecretKind,
        rotation_required: bool,
    },
    Expired {
        kind: SecretKind,
        age_hours: u32,
    },
    InvalidFormat {
        reason: &'static str,
    },
}

const ECR_LOGIN_VALIDITY_HOURS: u32 = 12;
const SA_TOKEN_VALIDITY_HOURS: u32 = 24;
const AZURE_MSI_VALIDITY_HOURS: u32 = 24;

pub fn validate(kind: SecretKind, age_hours: u32, has_required_fields: bool) -> SecretVerdict {
    if !has_required_fields {
        return SecretVerdict::InvalidFormat {
            reason: "missing required fields for kind",
        };
    }
    let validity = match kind {
        SecretKind::DockerConfigJson | SecretKind::GcpServiceAccount => u32::MAX,
        SecretKind::ServiceAccountToken => SA_TOKEN_VALIDITY_HOURS,
        SecretKind::AwsEcrLogin => ECR_LOGIN_VALIDITY_HOURS,
        SecretKind::AzureMsi => AZURE_MSI_VALIDITY_HOURS,
    };
    if validity != u32::MAX && age_hours >= validity {
        return SecretVerdict::Expired { kind, age_hours };
    }
    let rotation_required = matches!(
        kind,
        SecretKind::AwsEcrLogin | SecretKind::ServiceAccountToken | SecretKind::AzureMsi
    ) && age_hours * 2 > validity;
    SecretVerdict::Valid {
        kind,
        rotation_required,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_pull_secret_validator")?;

    println!(
        "ECR fresh: {:?}",
        validate(SecretKind::AwsEcrLogin, 1, true)
    );
    println!(
        "ECR halfway: {:?}",
        validate(SecretKind::AwsEcrLogin, 7, true)
    );
    println!(
        "ECR expired: {:?}",
        validate(SecretKind::AwsEcrLogin, 13, true)
    );
    println!(
        "Docker config (forever): {:?}",
        validate(SecretKind::DockerConfigJson, 100_000, true)
    );
    println!(
        "missing fields: {:?}",
        validate(SecretKind::ServiceAccountToken, 1, false)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ecr_fresh_valid() {
        let v = validate(SecretKind::AwsEcrLogin, 1, true);
        assert!(matches!(v, SecretVerdict::Valid { .. }));
    }

    #[test]
    fn ecr_expired_after_12h() {
        let v = validate(SecretKind::AwsEcrLogin, 13, true);
        assert!(matches!(v, SecretVerdict::Expired { .. }));
    }

    #[test]
    fn ecr_rotation_required_at_halfway() {
        // Past halfway = age × 2 > validity → rotate.
        let v = validate(SecretKind::AwsEcrLogin, 7, true);
        if let SecretVerdict::Valid {
            rotation_required, ..
        } = v
        {
            assert!(rotation_required);
        }
    }

    #[test]
    fn docker_config_never_expires() {
        let v = validate(SecretKind::DockerConfigJson, 100_000, true);
        assert!(matches!(v, SecretVerdict::Valid { .. }));
    }

    #[test]
    fn gcp_sa_never_expires() {
        let v = validate(SecretKind::GcpServiceAccount, 100_000, true);
        assert!(matches!(v, SecretVerdict::Valid { .. }));
    }

    #[test]
    fn missing_fields_invalid() {
        let v = validate(SecretKind::AwsEcrLogin, 1, false);
        assert!(matches!(v, SecretVerdict::InvalidFormat { .. }));
    }

    #[test]
    fn sa_token_24h_validity() {
        let v_fresh = validate(SecretKind::ServiceAccountToken, 1, true);
        let v_expired = validate(SecretKind::ServiceAccountToken, 25, true);
        assert!(matches!(v_fresh, SecretVerdict::Valid { .. }));
        assert!(matches!(v_expired, SecretVerdict::Expired { .. }));
    }

    #[test]
    fn azure_msi_24h_validity() {
        let v_fresh = validate(SecretKind::AzureMsi, 1, true);
        let v_expired = validate(SecretKind::AzureMsi, 25, true);
        assert!(matches!(v_fresh, SecretVerdict::Valid { .. }));
        assert!(matches!(v_expired, SecretVerdict::Expired { .. }));
    }

    #[test]
    fn fresh_token_no_rotation_needed() {
        let v = validate(SecretKind::AwsEcrLogin, 1, true);
        if let SecretVerdict::Valid {
            rotation_required, ..
        } = v
        {
            assert!(!rotation_required);
        }
    }

    #[test]
    fn long_lived_secrets_no_rotation() {
        // Docker config + GCP SA never need rotation.
        let v_docker = validate(SecretKind::DockerConfigJson, 1000, true);
        if let SecretVerdict::Valid {
            rotation_required, ..
        } = v_docker
        {
            assert!(!rotation_required);
        }
    }

    #[test]
    fn boundary_at_validity_expired() {
        let v = validate(SecretKind::AwsEcrLogin, ECR_LOGIN_VALIDITY_HOURS, true);
        assert!(matches!(v, SecretVerdict::Expired { .. }));
    }
}
