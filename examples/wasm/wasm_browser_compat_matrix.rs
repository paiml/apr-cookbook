//! # WASM Browser Compatibility Matrix
//!
//! Feature support across major browsers (release year approximations):
//!   simd128: Chrome 91+, Firefox 89+, Safari 16.4+, Edge 91+
//!   threads: Chrome 74+ (with COOP+COEP), Firefox 79+, Safari 15+
//!   gc: Chrome 119+, Firefox 120+, Safari 18+
//!   exception_handling: Chrome 95+, Firefox 100+, Safari 15.2+
//!
//! Picker returns a tier (Full/Most/Some/None) for the (browser, version)
//! pair against a feature set.
//!
//! Demonstrates the **WASM.16** recipe for PMAT-139 (wasm round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly feature-detection table (caniuse.com).
//!
//! Run with: cargo run --example wasm_browser_compat_matrix
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Browser {
    Chrome,
    Firefox,
    Safari,
    Edge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    Simd128,
    Threads,
    Gc,
    ExceptionHandling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportTier {
    Full,
    Most,
    Some,
    None,
}

#[derive(Debug, PartialEq)]
pub enum CompatVerdict {
    Ok {
        tier: SupportTier,
        unsupported: Vec<Feature>,
    },
    InvalidVersion,
}

pub fn check(browser: Browser, version: u32, features: &[Feature]) -> CompatVerdict {
    if version == 0 {
        return CompatVerdict::InvalidVersion;
    }
    let unsupported: Vec<Feature> = features
        .iter()
        .filter(|f| !supports(browser, version, **f))
        .copied()
        .collect();
    let tier = match (features.len(), unsupported.len()) {
        (0, _) => SupportTier::Full,
        (n, 0) if n > 0 => SupportTier::Full,
        (n, m) if m * 2 < n => SupportTier::Most,
        (n, m) if m < n => SupportTier::Some,
        _ => SupportTier::None,
    };
    CompatVerdict::Ok { tier, unsupported }
}

fn supports(browser: Browser, version: u32, feature: Feature) -> bool {
    let min_version = min_version_for(browser, feature);
    version >= min_version
}

fn min_version_for(browser: Browser, feature: Feature) -> u32 {
    match (browser, feature) {
        (Browser::Chrome | Browser::Edge, Feature::Simd128) => 91,
        (Browser::Firefox, Feature::Simd128) => 89,
        (Browser::Safari, Feature::Simd128) => 16,

        (Browser::Chrome | Browser::Edge, Feature::Threads) => 74,
        (Browser::Firefox, Feature::Threads) => 79,
        (Browser::Safari, Feature::Threads) => 15,

        (Browser::Chrome | Browser::Edge, Feature::Gc) => 119,
        (Browser::Firefox, Feature::Gc) => 120,
        (Browser::Safari, Feature::Gc) => 18,

        (Browser::Chrome | Browser::Edge, Feature::ExceptionHandling) => 95,
        (Browser::Firefox, Feature::ExceptionHandling) => 100,
        (Browser::Safari, Feature::ExceptionHandling) => 15,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_browser_compat_matrix")?;

    let features = [Feature::Simd128, Feature::Threads, Feature::Gc];
    println!("Chrome 120: {:?}", check(Browser::Chrome, 120, &features));
    println!("Safari 17: {:?}", check(Browser::Safari, 17, &features));
    println!("Firefox 70: {:?}", check(Browser::Firefox, 70, &features));
    println!(
        "Chrome 0 (invalid): {:?}",
        check(Browser::Chrome, 0, &features)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn modern_chrome_full_support() {
        let v = check(
            Browser::Chrome,
            120,
            &[Feature::Simd128, Feature::Threads, Feature::Gc],
        );
        if let CompatVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SupportTier::Full);
        }
    }

    #[test]
    fn old_browser_some_support() {
        // Firefox 80: Threads yes, Simd no, GC no.
        let v = check(
            Browser::Firefox,
            80,
            &[Feature::Simd128, Feature::Threads, Feature::Gc],
        );
        if let CompatVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SupportTier::Some);
        }
    }

    #[test]
    fn very_old_browser_none() {
        let v = check(Browser::Firefox, 50, &[Feature::Simd128, Feature::Threads]);
        if let CompatVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SupportTier::None);
        }
    }

    #[test]
    fn invalid_version_zero_rejected() {
        assert_eq!(
            check(Browser::Chrome, 0, &[Feature::Simd128]),
            CompatVerdict::InvalidVersion
        );
    }

    #[test]
    fn unsupported_listed() {
        let v = check(
            Browser::Chrome,
            80,
            &[Feature::Simd128, Feature::Threads, Feature::Gc],
        );
        if let CompatVerdict::Ok { unsupported, .. } = v {
            assert!(unsupported.contains(&Feature::Simd128));
            assert!(unsupported.contains(&Feature::Gc));
        }
    }

    #[test]
    fn safari_threshold_simd_at_16() {
        let v = check(Browser::Safari, 16, &[Feature::Simd128]);
        if let CompatVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SupportTier::Full);
        }
    }

    #[test]
    fn safari_pre_simd_unsupported() {
        let v = check(Browser::Safari, 15, &[Feature::Simd128]);
        if let CompatVerdict::Ok { unsupported, .. } = v {
            assert_eq!(unsupported, vec![Feature::Simd128]);
        }
    }

    #[test]
    fn empty_features_full_tier() {
        let v = check(Browser::Chrome, 100, &[]);
        if let CompatVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SupportTier::Full);
        }
    }

    #[test]
    fn most_tier_when_majority_supported() {
        // Chrome 120 supports all 4 features.
        let v = check(
            Browser::Chrome,
            95,
            &[
                Feature::Simd128,
                Feature::Threads,
                Feature::ExceptionHandling,
                Feature::Gc, // Chrome 95 doesn't have Gc (119+)
            ],
        );
        if let CompatVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SupportTier::Most);
        }
    }

    #[test]
    fn edge_uses_chrome_thresholds() {
        // Edge tracks Chrome.
        let chrome = check(Browser::Chrome, 120, &[Feature::Simd128]);
        let edge = check(Browser::Edge, 120, &[Feature::Simd128]);
        assert_eq!(chrome, edge);
    }
}
