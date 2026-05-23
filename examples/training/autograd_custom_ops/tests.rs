//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use super::*;

#[test]
fn test_gelu_zero() {
    let input = Tensor::from_vec(vec![0.0], false);
    let output = custom_gelu(&input);
    assert!(
        output.data()[0].abs() < 1e-6,
        "GELU(0) should be ~0, got {}",
        output.data()[0]
    );
}

#[test]
fn test_gelu_positive_large() {
    let input = Tensor::from_vec(vec![10.0], false);
    let output = custom_gelu(&input);
    // For large positive x, GELU(x) ~ x
    assert!(
        (output.data()[0] - 10.0).abs() < 0.01,
        "GELU(10.0) should be ~10.0, got {}",
        output.data()[0]
    );
}

#[test]
fn test_gelu_negative_large() {
    let input = Tensor::from_vec(vec![-10.0], false);
    let output = custom_gelu(&input);
    // For large negative x, GELU(x) ~ 0
    assert!(
        output.data()[0].abs() < 0.01,
        "GELU(-10.0) should be ~0, got {}",
        output.data()[0]
    );
}

#[test]
fn test_gelu_matches_builtin() {
    let data = deterministic_data(100, 16);
    let tensor = Tensor::from_vec(data, false);
    let builtin = entrenar::autograd::gelu(&tensor);
    let custom = custom_gelu(&tensor);

    for i in 0..builtin.data().len() {
        let diff = (builtin.data()[i] - custom.data()[i]).abs();
        assert!(
            diff < 1e-5,
            "GELU mismatch at index {}: builtin={}, custom={}, diff={}",
            i,
            builtin.data()[i],
            custom.data()[i],
            diff
        );
    }
}

#[test]
fn test_gelu_gradient_numerical_verification() {
    let input = Array1::from_vec(deterministic_data(101, 4));
    let ones = Array1::ones(input.len());
    let analytic = compute_gelu_gradient(&input, &ones);
    let numerical = numerical_gradient(gelu_sum, &input, FD_EPSILON);

    for i in 0..input.len() {
        let diff = (analytic[i] - numerical[i]).abs();
        assert!(
            diff < 1e-3,
            "GELU gradient mismatch at index {}: analytic={}, numerical={}, diff={}",
            i,
            analytic[i],
            numerical[i],
            diff
        );
    }
}

#[test]
fn test_huber_loss_quadratic_region() {
    // When |residual| <= delta, Huber = 0.5 * r^2
    let preds = Tensor::from_vec(vec![0.3], false);
    let targets = Array1::from_vec(vec![0.0]);
    let (_, loss) = custom_huber_loss(&preds, &targets, 1.0);
    let expected = 0.5 * 0.3 * 0.3;
    assert!(
        (loss - expected).abs() < 1e-6,
        "Huber quadratic region: expected {}, got {}",
        expected,
        loss
    );
}

#[test]
fn test_huber_loss_linear_region() {
    // When |residual| > delta, Huber = delta * (|r| - 0.5 * delta)
    let preds = Tensor::from_vec(vec![3.0], false);
    let targets = Array1::from_vec(vec![0.0]);
    let delta = 1.0;
    let (_, loss) = custom_huber_loss(&preds, &targets, delta);
    let expected = delta * (3.0 - 0.5 * delta);
    assert!(
        (loss - expected).abs() < 1e-6,
        "Huber linear region: expected {}, got {}",
        expected,
        loss
    );
}

#[test]
fn test_huber_loss_gradient_numerical_verification() {
    let preds = Array1::from_vec(deterministic_data(102, 4));
    let targets = Array1::from_vec(deterministic_targets(103, 4));

    let numerical = numerical_gradient(
        |p| huber_loss_fn(p, &targets, HUBER_DELTA),
        &preds,
        FD_EPSILON,
    );

    let pred_tensor = Tensor::from_vec(preds.to_vec(), true);
    let (mut loss, _) = custom_huber_loss(&pred_tensor, &targets, HUBER_DELTA);
    backward(&mut loss, None);

    let analytic = pred_tensor
        .grad()
        .expect("Gradient should be computed after backward");

    for i in 0..preds.len() {
        let diff = (analytic[i] - numerical[i]).abs();
        assert!(
            diff < 1e-3,
            "Huber gradient mismatch at {}: analytic={}, numerical={}, diff={}",
            i,
            analytic[i],
            numerical[i],
            diff
        );
    }
}

#[test]
fn test_huber_loss_zero_residual() {
    let preds = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let targets = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let (_, loss) = custom_huber_loss(&preds, &targets, 1.0);
    assert!(
        loss.abs() < 1e-6,
        "Huber loss with zero residuals should be 0, got {}",
        loss
    );
}

#[test]
fn test_training_convergence() {
    let dim = 4;
    let mut weights = Tensor::from_vec(deterministic_data(104, dim), true);
    let targets = Array1::from_vec(deterministic_targets(105, dim));
    let mut optimizer = AdamW::default_params(0.05);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..30 {
        optimizer.zero_grad_refs(&mut [&mut weights]);

        let activated = custom_gelu(&weights);
        let (mut loss, loss_val) = custom_huber_loss(&activated, &targets, HUBER_DELTA);

        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        backward(&mut loss, None);
        if let Some(g) = weights.grad() {
            weights.set_grad(g);
        }
        optimizer.step_refs(&mut [&mut weights]);
    }

    assert!(
        last_loss < first_loss,
        "Training should reduce loss: {} -> {}",
        first_loss,
        last_loss
    );
}

#[test]
fn test_deterministic_data_reproducible() {
    let d1 = deterministic_data(42, 10);
    let d2 = deterministic_data(42, 10);
    assert_eq!(d1, d2, "Same seed should produce identical data");
}

#[test]
fn test_deterministic_data_different_seeds() {
    let d1 = deterministic_data(42, 10);
    let d2 = deterministic_data(43, 10);
    assert_ne!(d1, d2, "Different seeds should produce different data");
}

#[test]
fn test_custom_gelu_backward_fires() {
    let input = Tensor::from_vec(vec![1.0, -1.0, 0.5, -0.5], true);
    let output = custom_gelu(&input);

    // Set upstream gradient
    output.set_grad(Array1::ones(4));

    // Trigger backward
    if let Some(op) = output.backward_op() {
        op.backward();
    }

    let grad = input.grad();
    assert!(grad.is_some(), "Input should have gradient after backward");
    let g = grad.expect("just checked");
    assert_eq!(g.len(), 4);
    // GELU gradient at x=0.5 should be positive
    assert!(g[2] > 0.0, "GELU gradient at x=0.5 should be positive");
}
