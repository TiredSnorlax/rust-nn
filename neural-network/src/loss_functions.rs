use matrix::matrix::Matrix;

pub struct LossFunction {
    pub function: fn(output: &Matrix, y: &Matrix) -> Matrix,
    pub derivative: fn(output: &Matrix, y: &Matrix) -> Matrix,
}

pub const MSE: LossFunction = LossFunction {
    function: |output: &Matrix, y: &Matrix| {
        let diff = output.subtract(y);
        let squared = diff.map(|x| x * x);
        squared
    },
    derivative: |output: &Matrix, y: &Matrix| {
        let diff = output.subtract(y);
        diff.map(|x| 2.0 * x)
    },
};

/// Used for logistic regression
pub const LOGISTIC: LossFunction = LossFunction {
    function: |output: &Matrix, y: &Matrix| {
        // -ylog(a) - (1 - y)log(1 - a)
        // Clip values to prevent ln(0) or ln(1)
        let eps = 1e-15;
        let n_log_a = output.map(|a| -(a.max(eps).min(1.0 - eps)).ln());
        let log_1_minus_a = output.map(|a| (1.0 - (a.max(eps).min(1.0 - eps))).ln());
        let term1 = y.multiply_elementwise(&n_log_a);
        let term2 = y.map(|y| 1.0 - y).multiply_elementwise(&log_1_minus_a);
        let res = term1.subtract(&term2);
        res
    },
    derivative: |a: &Matrix, y: &Matrix| {
        // (a - y) / (a * (1 - a))
        let eps = 1e-15;
        let mut res = Matrix::new(a.rows, a.cols);
        for i in 0..a.data.len() {
            let a_val = a.data[i].max(eps).min(1.0 - eps);
            let y_val = y.data[i];
            res.data[i] = (a_val - y_val) / (a_val * (1.0 - a_val));
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
    fn test_mse_derivative() {
        let output = matrix_from_vec(1, 2, vec![1.0, 2.0]);
        let y = matrix_from_vec(1, 2, vec![0.5, 3.0]);
        let deriv = (MSE.derivative)(&output, &y);
        // 2*(1.0-0.5) = 1.0
        // 2*(2.0-3.0) = -2.0
        assert_eq!(deriv.data, vec![1.0, -2.0]);
    }

    #[test]
    fn test_logistic_function() {
        let output = matrix_from_vec(1, 1, vec![0.5]);
        let y = matrix_from_vec(1, 1, vec![1.0]);
        let cost = (LOGISTIC.function)(&output, &y);
        // -1*ln(0.5) - (1-1)*ln(1-0.5) = -ln(0.5) ≈ 0.6931
        assert!((cost.data[0] - 0.69314718056).abs() < 1e-10);

        let output = matrix_from_vec(1, 1, vec![0.5]);
        let y = matrix_from_vec(1, 1, vec![0.0]);
        let cost = (LOGISTIC.function)(&output, &y);
        // -0*ln(0.5) - (1-0)*ln(1-0.5) = -ln(0.5) ≈ 0.6931
        assert!((cost.data[0] - 0.69314718056).abs() < 1e-10);
    }

    #[test]
    fn test_logistic_derivative() {
        let output = matrix_from_vec(1, 1, vec![0.5]);
        let y = matrix_from_vec(1, 1, vec![1.0]);
        let deriv = (LOGISTIC.derivative)(&output, &y);
        // (0.5 - 1.0) / (0.5 * (1 - 0.5)) = -0.5 / 0.25 = -2.0
        assert_eq!(deriv.data, vec![-2.0]);

        let output = matrix_from_vec(1, 1, vec![0.8]);
        let y = matrix_from_vec(1, 1, vec![1.0]);
        let deriv = (LOGISTIC.derivative)(&output, &y);
        // (0.8 - 1.0) / (0.8 * (1 - 0.8)) = -0.2 / (0.8 * 0.2) = -0.2 / 0.16 = -1.25
        assert_eq!(deriv.data, vec![-1.25]);
    }
}
