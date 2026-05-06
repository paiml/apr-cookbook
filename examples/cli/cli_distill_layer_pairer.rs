//! # apr distill — Layer Pairer (teacher → student depth mapping)
//!
//! Layer-wise distillation needs a mapping from teacher layers to
//! student layers. When teacher has T layers and student has S < T,
//! options are: uniform (every T/S-th layer), first-S, last-S, or
//! end-to-end pinning (always pair {0, T-1}). This recipe builds the
//! pairer.
//!
//! Demonstrates the **DISTILL.4** recipe for PMAT-113 (apr distill coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DISTILL-001 + Sun et al. 2019 (Patient KD)
//!
//! Run with: cargo run --example cli_distill_layer_pairer
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingStrategy {
    Uniform,
    FirstN,
    LastN,
    EndToEndPinned,
}

#[derive(Debug, PartialEq)]
pub enum PairingVerdict {
    Ok(Vec<(usize, usize)>),
    StudentLargerThanTeacher,
    EmptyArchitecture,
}

pub fn pair_layers(
    teacher_layers: usize,
    student_layers: usize,
    strategy: PairingStrategy,
) -> PairingVerdict {
    if teacher_layers == 0 || student_layers == 0 {
        return PairingVerdict::EmptyArchitecture;
    }
    if student_layers > teacher_layers {
        return PairingVerdict::StudentLargerThanTeacher;
    }
    let pairs = match strategy {
        PairingStrategy::Uniform => uniform_pairs(teacher_layers, student_layers),
        PairingStrategy::FirstN => (0..student_layers).map(|i| (i, i)).collect(),
        PairingStrategy::LastN => (0..student_layers)
            .map(|i| (teacher_layers - student_layers + i, i))
            .collect(),
        PairingStrategy::EndToEndPinned => {
            let mut p = uniform_pairs(teacher_layers, student_layers);
            if let Some(first) = p.first_mut() {
                *first = (0, 0);
            }
            if let Some(last) = p.last_mut() {
                *last = (teacher_layers - 1, student_layers - 1);
            }
            p
        }
    };
    PairingVerdict::Ok(pairs)
}

fn uniform_pairs(teacher: usize, student: usize) -> Vec<(usize, usize)> {
    (0..student)
        .map(|i| {
            let t_idx = (i * teacher) / student;
            (t_idx, i)
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_distill_layer_pairer")?;

    for strategy in [
        PairingStrategy::Uniform,
        PairingStrategy::FirstN,
        PairingStrategy::LastN,
        PairingStrategy::EndToEndPinned,
    ] {
        println!("{strategy:?}: {:?}", pair_layers(12, 4, strategy));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn uniform_distributes_evenly() {
        let v = pair_layers(12, 4, PairingStrategy::Uniform);
        if let PairingVerdict::Ok(p) = v {
            // Teacher indices spread across [0, 12).
            assert_eq!(p, vec![(0, 0), (3, 1), (6, 2), (9, 3)]);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn first_n_takes_prefix() {
        let v = pair_layers(12, 4, PairingStrategy::FirstN);
        if let PairingVerdict::Ok(p) = v {
            assert_eq!(p, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn last_n_takes_suffix() {
        let v = pair_layers(12, 4, PairingStrategy::LastN);
        if let PairingVerdict::Ok(p) = v {
            assert_eq!(p, vec![(8, 0), (9, 1), (10, 2), (11, 3)]);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn end_to_end_pinned_keeps_endpoints() {
        let v = pair_layers(12, 4, PairingStrategy::EndToEndPinned);
        if let PairingVerdict::Ok(p) = v {
            assert_eq!(p.first(), Some(&(0, 0)));
            assert_eq!(p.last(), Some(&(11, 3)));
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn student_larger_rejected() {
        let v = pair_layers(4, 12, PairingStrategy::Uniform);
        assert_eq!(v, PairingVerdict::StudentLargerThanTeacher);
    }

    #[test]
    fn zero_teacher_or_student_rejected() {
        assert_eq!(
            pair_layers(0, 4, PairingStrategy::Uniform),
            PairingVerdict::EmptyArchitecture
        );
        assert_eq!(
            pair_layers(4, 0, PairingStrategy::Uniform),
            PairingVerdict::EmptyArchitecture
        );
    }

    #[test]
    fn equal_size_yields_identity_for_first_n() {
        let v = pair_layers(4, 4, PairingStrategy::FirstN);
        if let PairingVerdict::Ok(p) = v {
            for (i, (t, s)) in p.iter().enumerate() {
                assert_eq!(*t, i);
                assert_eq!(*s, i);
            }
        }
    }
}
