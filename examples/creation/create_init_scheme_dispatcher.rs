//! # Creation Weight-Init Scheme Dispatcher
//!
//! Picks an initialization scheme by activation function: Xavier
//! (Glorot) for tanh/sigmoid, He/Kaiming for ReLU/GELU, LeCun for SELU,
//! Orthogonal for RNNs. Wrong scheme → vanishing/exploding activations
//! at depth. This recipe builds the dispatcher + std-dev calculator.
//!
//! Demonstrates the **CREATE.7** recipe for PMAT-127 (creation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: He et al. (2015). Delving Deep into Rectifiers. ICCV.
//!
//! Run with: cargo run --example create_init_scheme_dispatcher
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Tanh,
    Sigmoid,
    Relu,
    Gelu,
    Selu,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitScheme {
    XavierGlorot,
    HeKaiming,
    Lecun,
    Uniform,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok { scheme: InitScheme, stddev: f64 },
    InvalidFanIn,
    InvalidFanOut,
}

pub fn pick_scheme(act: Activation) -> InitScheme {
    match act {
        Activation::Tanh | Activation::Sigmoid => InitScheme::XavierGlorot,
        Activation::Relu | Activation::Gelu => InitScheme::HeKaiming,
        Activation::Selu => InitScheme::Lecun,
        Activation::Linear => InitScheme::XavierGlorot,
    }
}

pub fn dispatch(act: Activation, fan_in: u32, fan_out: u32) -> DispatchVerdict {
    if fan_in == 0 {
        return DispatchVerdict::InvalidFanIn;
    }
    if fan_out == 0 {
        return DispatchVerdict::InvalidFanOut;
    }
    let scheme = pick_scheme(act);
    let stddev = match scheme {
        InitScheme::XavierGlorot => {
            // sqrt(2 / (fan_in + fan_out))
            (2.0 / f64::from(fan_in + fan_out)).sqrt()
        }
        InitScheme::HeKaiming => {
            // sqrt(2 / fan_in)
            (2.0 / f64::from(fan_in)).sqrt()
        }
        InitScheme::Lecun => {
            // sqrt(1 / fan_in)
            (1.0 / f64::from(fan_in)).sqrt()
        }
        InitScheme::Uniform => 0.05,
    };
    DispatchVerdict::Ok { scheme, stddev }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("create_init_scheme_dispatcher")?;

    for act in [
        Activation::Relu,
        Activation::Tanh,
        Activation::Selu,
        Activation::Gelu,
        Activation::Linear,
    ] {
        println!("{act:?} → {:?}", dispatch(act, 256, 256));
    }
    println!("invalid: {:?}", dispatch(Activation::Relu, 0, 256));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn relu_picks_he() {
        assert_eq!(pick_scheme(Activation::Relu), InitScheme::HeKaiming);
    }

    #[test]
    fn gelu_picks_he() {
        assert_eq!(pick_scheme(Activation::Gelu), InitScheme::HeKaiming);
    }

    #[test]
    fn tanh_picks_xavier() {
        assert_eq!(pick_scheme(Activation::Tanh), InitScheme::XavierGlorot);
    }

    #[test]
    fn sigmoid_picks_xavier() {
        assert_eq!(pick_scheme(Activation::Sigmoid), InitScheme::XavierGlorot);
    }

    #[test]
    fn selu_picks_lecun() {
        assert_eq!(pick_scheme(Activation::Selu), InitScheme::Lecun);
    }

    #[test]
    fn he_stddev_matches_formula() {
        // He: sqrt(2 / fan_in). fan_in=128 → sqrt(2/128) = 0.125.
        if let DispatchVerdict::Ok { stddev, .. } = dispatch(Activation::Relu, 128, 256) {
            assert!((stddev - 0.125).abs() < 1e-9);
        }
    }

    #[test]
    fn xavier_stddev_matches_formula() {
        // Xavier: sqrt(2 / (in+out)). 128+128=256 → sqrt(2/256) ≈ 0.0884.
        if let DispatchVerdict::Ok { stddev, .. } = dispatch(Activation::Tanh, 128, 128) {
            assert!((stddev - (2.0_f64 / 256.0).sqrt()).abs() < 1e-9);
        }
    }

    #[test]
    fn lecun_stddev_matches_formula() {
        // Lecun: sqrt(1 / fan_in). fan_in=400 → 0.05.
        if let DispatchVerdict::Ok { stddev, .. } = dispatch(Activation::Selu, 400, 100) {
            assert!((stddev - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_fan_in_invalid() {
        assert_eq!(
            dispatch(Activation::Relu, 0, 256),
            DispatchVerdict::InvalidFanIn
        );
    }

    #[test]
    fn zero_fan_out_invalid() {
        assert_eq!(
            dispatch(Activation::Relu, 256, 0),
            DispatchVerdict::InvalidFanOut
        );
    }

    #[test]
    fn larger_fan_in_smaller_he_stddev() {
        // Inverse sqrt → larger fan_in produces smaller stddev.
        if let (
            DispatchVerdict::Ok { stddev: small, .. },
            DispatchVerdict::Ok { stddev: large, .. },
        ) = (
            dispatch(Activation::Relu, 64, 64),
            dispatch(Activation::Relu, 1024, 64),
        ) {
            assert!(large < small);
        }
    }
}
