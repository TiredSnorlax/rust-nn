use matrix::matrix::Matrix;

pub struct LossFunction {
    pub function: fn(output: &Matrix, y: &Matrix) -> Matrix,
}

pub const MSE: LossFunction = LossFunction {
    function: |output: &Matrix, y: &Matrix| {
        let diff = output.subtract(y);
        let squared = diff.map(|x| x * x);
        squared
    },
};

/// Used for logistic regression
pub const BINARY_CROSSENTROPY: LossFunction = LossFunction {
    function: |output: &Matrix, y: &Matrix| {
        // -ylog(a) - (1 - y)log(1 - a)
        // Clip values to prevent ln(0) or ln(1)
        let eps = 1e-7;
        let clamped = output.map(|a| a.max(eps).min(1.0 - eps));
        let n_log_a = clamped.map(|a| -a.ln());
        let log_1_minus_a = clamped.map(|a| (1.0 - a).ln());
        let term1 = y.multiply_elementwise(&n_log_a);
        let term2 = y.map(|y| 1.0 - y).multiply_elementwise(&log_1_minus_a);
        let res = term1.subtract(&term2);
        res
    },
};

/// The targets for this are not one-hot encoded.
/// They will be in a matrix of (1, m), while the outputs are in a matrix shape of (n, m)
pub const SPARSE_CATEGORICAL_CROSSENTROPY: LossFunction = LossFunction {
    function: |output: &Matrix, y: &Matrix| {
        let eps = 1e-7;
        let mut res = Matrix::new(output.rows, output.cols);
        // For every example
        for j in 0..output.cols {
            let target_class = y.data[j] as usize;
            let index = target_class * output.cols + j;
            let clamped = output.data[index].max(eps).min(1.0 - eps);
            res.data[index] = -clamped.ln();
        }
        res
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::matrix::Matrix;

    fn matrix_from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        Matrix { rows, cols, data }
    }

    #[test]
    fn test_mse_function() {
        let output = matrix_from_vec(1, 2, vec![1.0, 2.0]);
        let y = matrix_from_vec(1, 2, vec![0.5, 3.0]);
        let cost = (MSE.function)(&output, &y);
        // (1.0-0.5)^2 = 0.25
        // (2.0-3.0)^2 = 1.0
        assert_eq!(cost.data, vec![0.25, 1.0]);
    }

    #[test]
    fn test_logistic_function() {
        let output = matrix_from_vec(1, 1, vec![0.5]);
        let y = matrix_from_vec(1, 1, vec![1.0]);
        let cost = (BINARY_CROSSENTROPY.function)(&output, &y);
        // -1*ln(0.5) - (1-1)*ln(1-0.5) = -ln(0.5) ≈ 0.6931
        assert!((cost.data[0] - 0.69314718056).abs() < 1e-10);

        let output = matrix_from_vec(1, 1, vec![0.5]);
        let y = matrix_from_vec(1, 1, vec![0.0]);
        let cost = (BINARY_CROSSENTROPY.function)(&output, &y);
        // -0*ln(0.5) - (1-0)*ln(1-0.5) = -ln(0.5) ≈ 0.6931
        assert!((cost.data[0] - 0.69314718056).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_categorical_cross_entropy_function() {
        let output = matrix_from_vec(
            3,
            2,
            vec![
                0.1, 0.3, // Example 0: Class 0, 1, 2
                0.7, 0.2, 0.2, 0.5,
            ],
        );
        let y = matrix_from_vec(1, 2, vec![1.0, 2.0]);
        let cost = (SPARSE_CATEGORICAL_CROSSENTROPY.function)(&output, &y);

        // Example 0: target 1, output 0.7. Loss: -ln(0.7) ≈ 0.35667
        // Example 1: target 2, output 0.5. Loss: -ln(0.5) ≈ 0.69315

        assert!((cost.data[1 * 2 + 0] - 0.35667494393).abs() < 1e-10);
        assert!((cost.data[2 * 2 + 1] - 0.69314718056).abs() < 1e-10);

        // Other values should be 0
        assert_eq!(cost.data[0 * 2 + 0], 0.0);
        assert_eq!(cost.data[2 * 2 + 0], 0.0);
        assert_eq!(cost.data[0 * 2 + 1], 0.0);
        assert_eq!(cost.data[1 * 2 + 1], 0.0);
    }
}
