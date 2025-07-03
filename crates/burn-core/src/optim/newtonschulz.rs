//! Implement Newton-Schulz method for matrix orthogonalization.
use core::marker::PhantomData;

use burn_tensor::backend::Backend;
use burn_tensor::{Float, Tensor};

use crate as burn;
use crate::config::Config;

/// Configuration for the Newton-Schulz method.
#[derive(Config)]
pub struct NewtonSchulzConfig {
    /// The number of iterations to perform.
    #[config(default = 5)]
    pub iterations: usize,
    #[config(default = 3.445)]
    coef_a: f32,
    #[config(default = -4.775)]
    coef_b: f32,
    #[config(default = 2.0315)]
    coef_c: f32,
    #[config(default = 1e-7)]
    /// The epsilon value for numerical stability.
    epsilon: f32,
}

/// Implementation for Newton-Schulz method for matrix orthogonalization.
#[derive(Clone)]
pub struct NewtonSchulz<B: Backend> {
    iterations: usize,
    coef_a: f32,
    coef_b: f32,
    coef_c: f32,
    epsilon: f32,
    _phantom: PhantomData<B>, // no state here.
}

impl<B: Backend> NewtonSchulz<B> {
    /// Creates a new [NewtonSchulz](NewtonSchulz) instance from the provided config.
    pub fn new(config: &NewtonSchulzConfig) -> Self {
        Self {
            iterations: config.iterations,
            coef_a: config.coef_a,
            coef_b: config.coef_b,
            coef_c: config.coef_c,
            epsilon: config.epsilon,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> NewtonSchulz<B> {
    /// Orthogonalize the gradient.
    pub fn transform<const D: usize>(&self, grad: Tensor<B, D, Float>) -> Tensor<B, D, Float> {
        let dims: [usize; D] = grad.shape().dims();
        if D > 2 {
            // no-op for higher dimensions
            // TODO: maybe flatten and apply?
            return grad;
        }

        let mut x = if dims[0] > dims[1] {
            grad.transpose() // prefer fat over tall
        } else {
            grad
        };

        let fnorm = (x.clone().powf_scalar(2.0).sum() + self.epsilon)
            .sqrt()
            .into_scalar();
        x = x / fnorm;

        for _ in 0..self.iterations {
            let a = x.clone().matmul(x.clone().transpose());
            let b = (a.clone() * self.coef_b) + self.coef_c * (a.clone().matmul(a.clone()));
            let dx = b.matmul(x.clone());
            x = self.coef_a * x + dx
        }

        if dims[0] > dims[1] { x.transpose() } else { x }
    }
}
