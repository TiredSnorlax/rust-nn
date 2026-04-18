use rand::RngExt;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "rows and cols must be greater than 0");
        Self {
            rows,
            cols,
            data: vec![0.; rows * cols],
        }
    }

    pub fn from(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert!(rows > 0 && cols > 0, "rows and cols must be greater than 0");
        Self { rows, cols, data }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "rows and cols must be greater than 0");

        let mut rng = rand::rng();

        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.random_range(0.0..=1.))
            .collect();

        Self { rows, cols, data }
    }

    pub fn random_range(rows: usize, cols: usize, range: std::ops::RangeInclusive<f64>) -> Self {
        assert!(rows > 0 && cols > 0, "rows and cols must be greater than 0");

        let mut rng = rand::rng();

        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.random_range(range.clone()))
            .collect();

        Self { rows, cols, data }
    }

    // Returns in (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Broadcast matrix from shape (n, 1) to (n, x)
    pub fn broadcast_cols(&self, cols: usize) -> Self {
        assert!(cols > 0, "cols must be greater than 0");
        let mut data = Vec::with_capacity(self.rows * cols);
        for row in 0..self.rows {
            for _ in 0..cols {
                data.push(self.data[row]);
            }
        }
        Matrix {
            rows: self.rows,
            cols,
            data,
        }
    }

    /// Broadcast matrix from shape (1, n) to (x, n)
    pub fn broadcast_rows(&self, rows: usize) -> Self {
        assert!(rows > 0, "rows must be greater than 0");
        let mut data = Vec::with_capacity(self.cols * rows);
        for _ in 0..rows {
            for col in 0..self.cols {
                data.push(self.data[col]);
            }
        }
        Matrix {
            rows,
            cols: self.cols,
            data,
        }
    }

    // Add
    // Subtract
    // Multiply (element-wise and matrix mult)
    // Dot
    // Transpose

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrices must have the same shape"
        );

        let mut result = Self::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    pub fn subtract(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrices must have the same shape"
        );

        let mut result = Self::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    pub fn multiply_elementwise(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrices must have the same shape"
        );

        let mut result = Self::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    pub fn divide_elementwise(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrices must have the same shape"
        );

        let mut result = Self::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] / other.data[i];
        }
        result
    }

    pub fn map(&self, func: impl Fn(f64) -> f64) -> Self {
        let mut result = Self::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = func(self.data[i]);
        }
        result
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "There is a mismatch between number of cols and rows"
        );

        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }

        result
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        result
    }

    pub fn sum(&self, axis: Option<usize>) -> Matrix {
        if let Some(dim) = axis {
            match dim {
                1 => {
                    // Sum row-wise
                    let mut result = Matrix::new(self.rows, 1);
                    for i in 0..self.rows {
                        let row_sum: f64 =
                            self.data[i * self.cols..(i + 1) * self.cols].iter().sum();
                        result.data[i] = row_sum;
                    }
                    result
                }

                2 => {
                    // Sum col-wise
                    let mut result = Matrix::new(1, self.cols);
                    let transposed = self.transpose();
                    for i in 0..self.cols {
                        let cols_sum: f64 = transposed.data[i * self.rows..(i + 1) * self.rows]
                            .iter()
                            .sum();
                        result.data[i] = cols_sum;
                    }
                    result
                }

                _ => {
                    panic!(
                        "Invalid axis of {} when using sum on {:?} shape matrix.",
                        dim,
                        self.shape()
                    )
                }
            }
        } else {
            let sum = self.data.iter().sum();
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![sum],
            }
        }
    }

    pub fn max(&self, axis: Option<usize>) -> Matrix {
        if let Some(dim) = axis {
            if dim == 1 {
                let mut result = Vec::with_capacity(self.cols);
                for i in 0..self.rows {
                    let max_val = self.data[i * self.cols..(i + 1) * self.cols]
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
                    result.push(max_val);
                }
                Matrix {
                    rows: self.rows,
                    cols: 1,
                    data: result,
                }
            } else if dim == 2 {
                let transposed = self.transpose();
                let mut result = Vec::with_capacity(self.rows);
                for i in 0..self.cols {
                    let max_val = transposed.data[i * self.rows..(i + 1) * self.rows]
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
                    result.push(max_val);
                }
                Matrix {
                    rows: 1,
                    cols: self.cols,
                    data: result,
                }
            } else {
                panic!(
                    "Invalid axis of {} when using max on {:?} shape matrix.",
                    dim,
                    self.shape()
                )
            }
        } else {
            let max_val = self.data.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![max_val],
            }
        }
    }

    pub fn min(&self, axis: Option<usize>) -> Matrix {
        if let Some(dim) = axis {
            if dim == 1 {
                let mut result = Vec::with_capacity(self.cols);
                for i in 0..self.rows {
                    let min_val = self.data[i * self.cols..(i + 1) * self.cols]
                        .iter()
                        .fold(f64::INFINITY, |a, b| a.min(*b));
                    result.push(min_val);
                }
                Matrix {
                    rows: self.rows,
                    cols: 1,
                    data: result,
                }
            } else if dim == 2 {
                let transposed = self.transpose();
                let mut result = Vec::with_capacity(self.rows);
                for i in 0..self.cols {
                    let min_val = transposed.data[i * self.rows..(i + 1) * self.rows]
                        .iter()
                        .fold(f64::INFINITY, |a, b| a.min(*b));
                    result.push(min_val);
                }
                Matrix {
                    rows: 1,
                    cols: self.cols,
                    data: result,
                }
            } else {
                panic!(
                    "Invalid axis of {} when using min on {:?} shape matrix.",
                    dim,
                    self.shape()
                )
            }
        } else {
            let min_val = self.data.iter().fold(f64::INFINITY, |a, b| a.min(*b));
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![min_val],
            }
        }
    }

    pub fn row(&self, index: usize) -> Vec<f64> {
        self.data[index * self.cols..(index + 1) * self.cols].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a matrix from a 2D array
    fn matrix_from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        let mut matrix = Matrix::new(rows, cols);
        matrix.data = data;
        matrix
    }

    #[test]
    fn test_new_matrix() {
        let matrix = Matrix::new(2, 3);
        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix.data.len(), 6);
        assert!(matrix.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_shape() {
        let matrix = Matrix::new(3, 4);
        assert_eq!(matrix.shape(), (3, 4));
    }

    #[test]
    #[should_panic(expected = "rows and cols must be greater than 0")]
    fn test_new_matrix_invalid_rows() {
        let _matrix = Matrix::new(0, 3);
    }

    #[test]
    #[should_panic(expected = "rows and cols must be greater than 0")]
    fn test_new_matrix_invalid_cols() {
        let _matrix = Matrix::new(3, 0);
    }

    #[test]
    fn test_add_matrices() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let result = matrix1.add(&matrix2);
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_with_negative_numbers() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, -2.0, -3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![-5.0, 6.0, 3.0, -8.0]);

        let result = matrix1.add(&matrix2);
        assert_eq!(result.data, vec![-4.0, 4.0, 0.0, -4.0]);
    }

    #[test]
    #[should_panic(expected = "Matrices must have the same shape")]
    fn test_add_mismatched_shapes() {
        let matrix1 = Matrix::new(2, 2);
        let matrix2 = Matrix::new(2, 3);
        let _result = matrix1.add(&matrix2);
    }

    #[test]
    fn test_subtract_matrices() {
        let matrix1 = matrix_from_vec(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let result = matrix1.subtract(&matrix2);
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.data, vec![9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_subtract_resulting_in_negative() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let result = matrix1.subtract(&matrix2);
        assert_eq!(result.data, vec![-4.0, -4.0, -4.0, -4.0]);
    }

    #[test]
    #[should_panic(expected = "Matrices must have the same shape")]
    fn test_subtract_mismatched_shapes() {
        let matrix1 = Matrix::new(3, 2);
        let matrix2 = Matrix::new(2, 3);
        let _result = matrix1.subtract(&matrix2);
    }

    #[test]
    fn test_multiply_elementwise() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![2.0, 3.0, 4.0, 5.0]);

        let result = matrix1.multiply_elementwise(&matrix2);
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.data, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_multiply_elementwise_with_zero() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, 0.0, 3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![2.0, 3.0, 4.0, 0.0]);

        let result = matrix1.multiply_elementwise(&matrix2);
        assert_eq!(result.data, vec![2.0, 0.0, 12.0, 0.0]);
    }

    #[test]
    fn test_multiply_elementwise_with_negatives() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, -2.0, 3.0, -4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![-2.0, 3.0, -4.0, 5.0]);

        let result = matrix1.multiply_elementwise(&matrix2);
        assert_eq!(result.data, vec![-2.0, -6.0, -12.0, -20.0]);
    }

    #[test]
    #[should_panic(expected = "Matrices must have the same shape")]
    fn test_multiply_elementwise_mismatched_shapes() {
        let matrix1 = Matrix::new(2, 2);
        let matrix2 = Matrix::new(2, 3);
        let _result = matrix1.multiply_elementwise(&matrix2);
    }

    #[test]
    fn test_divide_elementwise() {
        let matrix1 = matrix_from_vec(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![2.0, 4.0, 5.0, 8.0]);

        let result = matrix1.divide_elementwise(&matrix2);
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.data, vec![5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_divide_elementwise_with_fractions() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![2.0, 4.0, 2.0, 8.0]);

        let result = matrix1.divide_elementwise(&matrix2);
        assert_eq!(result.data, vec![0.5, 0.5, 1.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "Matrices must have the same shape")]
    fn test_divide_elementwise_mismatched_shapes() {
        let matrix1 = Matrix::new(2, 2);
        let matrix2 = Matrix::new(3, 2);
        let _result = matrix1.divide_elementwise(&matrix2);
    }

    #[test]
    fn test_matmul() {
        // Matrix1: 2x3, Matrix2: 3x2
        let matrix1 = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix2 = matrix_from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let result = matrix1.matmul(&matrix2);
        assert_eq!(result.shape(), (2, 2));
        // Row 0: [1, 2, 3] · [7, 9, 11] = 1*7 + 2*9 + 3*11 = 58
        //        [1, 2, 3] · [8, 10, 12] = 1*8 + 2*10 + 3*12 = 64
        // Row 1: [4, 5, 6] · [7, 9, 11] = 4*7 + 5*9 + 6*11 = 139
        //        [4, 5, 6] · [8, 10, 12] = 4*8 + 5*10 + 6*12 = 154
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_square_matrices() {
        let matrix1 = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = matrix_from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let result = matrix1.matmul(&matrix2);
        assert_eq!(result.shape(), (2, 2));
        // Row 0: [1, 2] · [5, 7] = 19, [1, 2] · [6, 8] = 22
        // Row 1: [3, 4] · [5, 7] = 43, [3, 4] · [6, 8] = 50
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_1x3_by_3x1() {
        let matrix1 = matrix_from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let matrix2 = matrix_from_vec(3, 1, vec![4.0, 5.0, 6.0]);

        let result = matrix1.matmul(&matrix2);
        assert_eq!(result.shape(), (1, 1));
        // [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(result.data[0], 32.0);
    }

    #[test]
    #[should_panic(expected = "There is a mismatch between number of cols and rows")]
    fn test_matmul_mismatched_dimensions() {
        let matrix1 = Matrix::new(2, 3);
        let matrix2 = Matrix::new(2, 2);
        let _result = matrix1.matmul(&matrix2);
    }

    #[test]
    fn test_transpose_rectangular() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = matrix.transpose();
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_square_matrix() {
        let matrix = matrix_from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let result = matrix.transpose();
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_1x3() {
        let matrix = matrix_from_vec(1, 3, vec![1.0, 2.0, 3.0]);

        let result = matrix.transpose();
        assert_eq!(result.shape(), (3, 1));
        assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose_twice_returns_original() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let transposed_once = matrix.transpose();
        let transposed_twice = transposed_once.transpose();

        assert_eq!(transposed_twice.shape(), matrix.shape());
        assert_eq!(transposed_twice.data, matrix.data);
    }

    #[test]
    fn test_sum_all() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matrix.sum(None);
        assert_eq!(result.data, vec![21.]);
        assert_eq!(result.shape(), (1, 1));
    }

    #[test]
    fn test_sum_row() {
        // [1., 2., 3.] => [6.]
        // [4., 5., 6.] => [15.]
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matrix.sum(Some(1));
        assert_eq!(result.data, vec![6., 15.]);
        assert_eq!(result.shape(), (2, 1));
    }

    #[test]
    fn test_sum_col() {
        // [1., 2., 3.] => [5., 7., 9.]
        // [4., 5., 6.]
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matrix.sum(Some(2));
        assert_eq!(result.data, vec![5., 7., 9.]);
        assert_eq!(result.shape(), (1, 3));
    }

    #[test]
    fn test_max_all() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matrix.max(None);
        assert_eq!(result.data, vec![6.]);
        assert_eq!(result.shape(), (1, 1));
    }

    #[test]
    fn test_max_row() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let result = matrix.max(Some(1));
        assert_eq!(result.data, vec![5., 6.]);
        assert_eq!(result.shape(), (2, 1));
    }

    #[test]
    fn test_max_col() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let result = matrix.max(Some(2));
        assert_eq!(result.data, vec![4., 5., 6.]);
        assert_eq!(result.shape(), (1, 3));
    }

    #[test]
    fn test_min_all() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matrix.min(None);
        assert_eq!(result.data, vec![1.]);
        assert_eq!(result.shape(), (1, 1));
    }

    #[test]
    fn test_min_row() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let result = matrix.min(Some(1));
        assert_eq!(result.data, vec![1., 2.]);
        assert_eq!(result.shape(), (2, 1));
    }

    #[test]
    fn test_min_col() {
        let matrix = matrix_from_vec(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let result = matrix.min(Some(2));
        assert_eq!(result.data, vec![1., 2., 3.]);
        assert_eq!(result.shape(), (1, 3));
    }

    #[test]
    fn test_broadcast_cols() {
        // [1.]  => [1., 1., 1.]
        // [2.]     [2., 2., 2.]
        let matrix = matrix_from_vec(2, 1, vec![1.0, 2.0]);
        let result = matrix.broadcast_cols(3);
        assert_eq!(result.shape(), (2, 3));
        assert_eq!(result.data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_broadcast_rows() {
        // [1., 2.] => [1., 2., 1., 2.]
        let matrix = matrix_from_vec(1, 2, vec![1.0, 2.0]);
        let result = matrix.broadcast_rows(3);
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }
}
