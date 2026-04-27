use std::f64::consts::E;

use matrix::matrix::Matrix;

pub struct Activation {
    pub function: fn(x: &Matrix) -> Matrix,
    pub derivative: fn(x: &Matrix) -> Matrix,
}

pub const SIGMOID: Activation = Activation {
    function: |x: &Matrix| x.map(|x| 1. / (1. + E.powf(-x))),
    derivative: |x: &Matrix| x.map(|x| E.powf(-x) / (1. + E.powf(-x)).powi(2)),
};

pub const RELU: Activation = Activation {
    function: |x: &Matrix| x.map(|x| if x <= 0. { 0. } else { x }),
    derivative: |x: &Matrix| x.map(|x| if x > 0. { 1. } else { 0. }),
};

pub const LINEAR: Activation = Activation {
    function: |x: &Matrix| x.map(|x| x),
    derivative: |x: &Matrix| x.map(|_| 1.),
};

// This is here because I don't want to deal with options in Layers just becaues of input layers.
pub const NONE: Activation = Activation {
    function: |x: &Matrix| x.map(|_| 0.),
    derivative: |x: &Matrix| x.map(|_| 0.),
};

pub const SOFTMAX: Activation = Activation {
    function: |x: &Matrix| {
        let exp = x.map(|x| E.powf(x));
        // Sum column-wise
        let e_sum = exp.sum(Some(1)).broadcast_rows(x.rows);
        exp.divide_elementwise(&e_sum)
    },
    // Softmax derivative is a Jacobian matrix, which doesn't fit the element-wise
    // backpropagation pattern used here. However, when combined with Categorical
    // Cross-Entropy loss, the gradient simplify to (a - y).
    // For now, we return 1.0 to let the loss function handle the gradient if it's
    // specialized for Softmax, or as a placeholder.
    derivative: |x: &Matrix| x.map(|_| 1.),
};
#[cfg(test)]
mod tests {
    use super::*;
    use matrix::matrix::Matrix;

    #[test]
    fn test_sigmoid() {
        let input = Matrix::from(1, 1, vec![0.0]);
        let output = (SIGMOID.function)(&input);
        assert!((output.data[0] - 0.5).abs() < 1e-10);

        let deriv = (SIGMOID.derivative)(&input);
        assert!((deriv.data[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_relu() {
        let input = Matrix::from(1, 2, vec![-1.0, 2.0]);
        let output = (RELU.function)(&input);
        assert_eq!(output.data, vec![0.0, 2.0]);

        let deriv = (RELU.derivative)(&input);
        assert_eq!(deriv.data, vec![0.0, 1.0]);
    }

    #[test]
    fn test_softmax() {
        let input = Matrix::from(3, 1, vec![1.0, 2.0, 3.0]);
        let output = (SOFTMAX.function)(&input);

        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // e^1 / (e^1 + e^2 + e^3) ≈ 2.718 / (2.718 + 7.389 + 20.085) ≈ 2.718 / 30.192 ≈ 0.09003
        assert!((output.data[0] - 0.09003057317).abs() < 1e-8);
        assert!((output.data[1] - 0.24472847105).abs() < 1e-8);
        assert!((output.data[2] - 0.66524095577).abs() < 1e-8);
    }
}
