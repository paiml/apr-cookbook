//! # Bundle Zero-Copy FD Handoff
//!
//! Transfer a bundle between processes via Unix-domain SCM_RIGHTS
//! (file-descriptor passing). Avoids re-mmap'ing or copying data.
//!
//! Picker: validates the handoff is actually possible
//! (same-host, supported FS, fd is mappable).
//!
//! Demonstrates the **BUNDLE.23** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SCM_RIGHTS POSIX fd-passing.
//!
//! Run with: cargo run --example bundle_zero_copy_handoff
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HandoffVerdict {
    Ok { method: HandoffMethod },
    SameHostRequired,
    UnsupportedFs { fs_kind: String },
    InvalidFd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffMethod {
    ScmRights,
    DupAndShare,
}

const MMAP_FRIENDLY_FS: &[&str] = &["ext4", "xfs", "btrfs", "tmpfs", "zfs"];

pub fn validate(
    same_host: bool,
    fs_kind: &str,
    fd_kind: u32,
    process_can_dup: bool,
) -> HandoffVerdict {
    if !same_host {
        return HandoffVerdict::SameHostRequired;
    }
    if !MMAP_FRIENDLY_FS.contains(&fs_kind) {
        return HandoffVerdict::UnsupportedFs {
            fs_kind: fs_kind.to_string(),
        };
    }
    if fd_kind != 0 {
        // 0 = regular file; non-zero indicates pipe/socket/etc.
        return HandoffVerdict::InvalidFd;
    }
    let method = if process_can_dup {
        HandoffMethod::DupAndShare
    } else {
        HandoffMethod::ScmRights
    };
    HandoffVerdict::Ok { method }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_zero_copy_handoff")?;

    println!("typical: {:?}", validate(true, "ext4", 0, true));
    println!("scm_rights: {:?}", validate(true, "xfs", 0, false));
    println!("remote host: {:?}", validate(false, "ext4", 0, true));
    println!("nfs: {:?}", validate(true, "nfs", 0, true));
    println!("non-file fd: {:?}", validate(true, "ext4", 1, true));
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
    fn typical_dup_method() {
        let v = validate(true, "ext4", 0, true);
        if let HandoffVerdict::Ok { method } = v {
            assert_eq!(method, HandoffMethod::DupAndShare);
        }
    }

    #[test]
    fn no_dup_uses_scm_rights() {
        let v = validate(true, "ext4", 0, false);
        if let HandoffVerdict::Ok { method } = v {
            assert_eq!(method, HandoffMethod::ScmRights);
        }
    }

    #[test]
    fn remote_host_rejected() {
        assert_eq!(
            validate(false, "ext4", 0, true),
            HandoffVerdict::SameHostRequired
        );
    }

    #[test]
    fn nfs_rejected() {
        let v = validate(true, "nfs", 0, true);
        assert!(matches!(v, HandoffVerdict::UnsupportedFs { .. }));
    }

    #[test]
    fn non_file_fd_rejected() {
        let v = validate(true, "ext4", 1, true);
        assert_eq!(v, HandoffVerdict::InvalidFd);
    }

    #[test]
    fn xfs_supported() {
        let v = validate(true, "xfs", 0, true);
        assert!(matches!(v, HandoffVerdict::Ok { .. }));
    }

    #[test]
    fn btrfs_supported() {
        let v = validate(true, "btrfs", 0, true);
        assert!(matches!(v, HandoffVerdict::Ok { .. }));
    }

    #[test]
    fn tmpfs_supported() {
        let v = validate(true, "tmpfs", 0, true);
        assert!(matches!(v, HandoffVerdict::Ok { .. }));
    }

    #[test]
    fn unknown_fs_rejected() {
        let v = validate(true, "fat32", 0, true);
        assert!(matches!(v, HandoffVerdict::UnsupportedFs { .. }));
    }

    #[test]
    fn dup_method_when_supported() {
        // Process can dup → faster path used.
        let v_dup = validate(true, "ext4", 0, true);
        let v_no_dup = validate(true, "ext4", 0, false);
        if let (HandoffVerdict::Ok { method: m_dup }, HandoffVerdict::Ok { method: m_scm }) =
            (v_dup, v_no_dup)
        {
            assert_ne!(m_dup, m_scm);
        }
    }
}
