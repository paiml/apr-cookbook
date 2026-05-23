//! # Visualization Color Palette Picker
//!
//! Pick a perceptually-uniform palette by data type: Viridis for
//! sequential continuous; Plasma for sequential with high-contrast
//! print; RdBu for diverging zero-centered; Tab10 for ≤ 10 categorical;
//! Glasbey for > 10 categorical. Wrong palette → misleading viz. This
//! recipe builds the picker.
//!
//! Demonstrates the **VIZ.2** recipe for PMAT-128 (visualization coverage —
//! closing F-invariant gap from 1 → 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: van der Walt & Smith (2015). A Better Default Colormap for Matplotlib.
//!
//! Run with: cargo run --example viz_color_palette_picker
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    SequentialContinuous,
    SequentialPrint,
    DivergingZeroCentered,
    Categorical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    Viridis,
    Plasma,
    RdBu,
    Tab10,
    Glasbey,
}

#[derive(Debug, PartialEq)]
pub enum PickerVerdict {
    Ok(Palette),
    EmptyData,
    DivergingNeedsZeroCrossing,
}

pub fn pick(kind: DataKind, num_categories: u32, has_zero_crossing: bool) -> PickerVerdict {
    if matches!(kind, DataKind::Categorical) && num_categories == 0 {
        return PickerVerdict::EmptyData;
    }
    let palette = match kind {
        DataKind::SequentialContinuous => Palette::Viridis,
        DataKind::SequentialPrint => Palette::Plasma,
        DataKind::DivergingZeroCentered => {
            if !has_zero_crossing {
                return PickerVerdict::DivergingNeedsZeroCrossing;
            }
            Palette::RdBu
        }
        DataKind::Categorical => {
            if num_categories <= 10 {
                Palette::Tab10
            } else {
                Palette::Glasbey
            }
        }
    };
    PickerVerdict::Ok(palette)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("viz_color_palette_picker")?;

    for (kind, n, zc) in [
        (DataKind::SequentialContinuous, 0, false),
        (DataKind::SequentialPrint, 0, false),
        (DataKind::DivergingZeroCentered, 0, true),
        (DataKind::DivergingZeroCentered, 0, false),
        (DataKind::Categorical, 5, false),
        (DataKind::Categorical, 50, false),
        (DataKind::Categorical, 0, false),
    ] {
        println!("{kind:?} n={n} zc={zc}  →  {:?}", pick(kind, n, zc));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn sequential_picks_viridis() {
        assert_eq!(
            pick(DataKind::SequentialContinuous, 0, false),
            PickerVerdict::Ok(Palette::Viridis)
        );
    }

    #[test]
    fn sequential_print_picks_plasma() {
        assert_eq!(
            pick(DataKind::SequentialPrint, 0, false),
            PickerVerdict::Ok(Palette::Plasma)
        );
    }

    #[test]
    fn diverging_with_zero_crossing_picks_rdbu() {
        assert_eq!(
            pick(DataKind::DivergingZeroCentered, 0, true),
            PickerVerdict::Ok(Palette::RdBu)
        );
    }

    #[test]
    fn diverging_without_zero_crossing_rejected() {
        assert_eq!(
            pick(DataKind::DivergingZeroCentered, 0, false),
            PickerVerdict::DivergingNeedsZeroCrossing
        );
    }

    #[test]
    fn small_categorical_picks_tab10() {
        assert_eq!(
            pick(DataKind::Categorical, 5, false),
            PickerVerdict::Ok(Palette::Tab10)
        );
    }

    #[test]
    fn at_10_categorical_picks_tab10() {
        // Boundary: ≤ 10 inclusive.
        assert_eq!(
            pick(DataKind::Categorical, 10, false),
            PickerVerdict::Ok(Palette::Tab10)
        );
    }

    #[test]
    fn large_categorical_picks_glasbey() {
        assert_eq!(
            pick(DataKind::Categorical, 50, false),
            PickerVerdict::Ok(Palette::Glasbey)
        );
    }

    #[test]
    fn zero_categories_rejected() {
        assert_eq!(
            pick(DataKind::Categorical, 0, false),
            PickerVerdict::EmptyData
        );
    }

    #[test]
    fn over_10_categorical_picks_glasbey() {
        assert_eq!(
            pick(DataKind::Categorical, 11, false),
            PickerVerdict::Ok(Palette::Glasbey)
        );
    }
}
