use crate::{activations::Activation, loss_functions::LossFunction};
use matrix::matrix::Matrix;
use rand::RngExt;

pub struct NeuralNetwork {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    activations: Vec<Activation>,
    cost_function: LossFunction,
}

impl NeuralNetwork {
    pub fn new(
        layers: Vec<usize>,
        activations: Vec<Activation>,
        cost_function: LossFunction,
    ) -> Self {
        assert!(layers.len() > 0, "Do not create empty neural network");

        assert!(
            activations.len() == layers.len() - 1,
            "Wrong number of activations. Input layer does not have an activation."
        );

        let mut weights: Vec<Matrix> = Vec::with_capacity(layers.len());
        let mut bias: Vec<Matrix> = Vec::with_capacity(layers.len());

        // Here we skip the input layer
        for i in 1..layers.len() {
            let num_neurons = layers[i];
            let prev_num_neurons = layers[i - 1];
            // He initialization: bounds = sqrt(2/fan_in) for ReLU
            let bound = (2.0 / prev_num_neurons as f64).sqrt();
            weights.push(Matrix::random_range(
                num_neurons,
                prev_num_neurons,
                -bound..=bound,
            ));
            // Bias initialized to 0, not random
            bias.push(Matrix::new(num_neurons, 1));
        }

        Self {
            layers,
            weights,
            bias,
            activations,
            cost_function,
        }
    }

    // This will return (output, activation values, z values)
    /// Inputs here should be shaped (m, n_features), where each row is an example
    /// Outputs will be shaped (n_i, m)
    pub fn feed_forward(&self, inputs: &Matrix) -> (Matrix, Vec<Matrix>, Vec<Matrix>) {
        assert_eq!(
            inputs.shape().1,
            self.layers[0],
            "Input shape does not match the network's input layer"
        );

        let mut values = inputs.transpose(); // shape: (n_features, m)
        // Activation values
        let mut a_values = Vec::with_capacity(self.layers.len());
        a_values.push(values.clone());
        // Values before going through activation function
        let mut z_values = Vec::with_capacity(self.weights.len());

        for i in 0..self.weights.len() {
            let w = &self.weights[i]; // shape: (n_i, n_{i-1})
            let b = &self.bias[i]; // shape: (n_i, 1)

            // Compute weighted sum: z = w * a + b
            let z_weighted = w.matmul(&values); // shape: (n_i, m)

            // Broadcast bias from shape (n_i, 1) to (n_i, m)
            let batch_size = z_weighted.shape().1;
            let b_broadcast = b.broadcast_cols(batch_size);
            // let n_i = z_weighted.shape().0;
            // let mut b_broadcast_data = Vec::with_capacity(n_i * batch_size);
            // for row in 0..n_i {
            //     for _ in 0..batch_size {
            //         b_broadcast_data.push(b.data[row]);
            //     }
            // }
            // let b_broadcast = Matrix {
            //     rows: n_i,
            //     cols: batch_size,
            //     data: b_broadcast_data,
            // };

            let z = z_weighted.add(&b_broadcast); // shape: (n_i, m)
            z_values.push(z.clone());
            let activation = &self.activations[i]; // shape: (n_i, m)
            let a = z.map(activation.function);
            a_values.push(a.clone());

            values = a;
        }

        // These a_values is a vector of matrices. Each matrice represents the activations of all the examples for a given layer
        // Same for z_values
        (values, a_values, z_values)
    }

    pub fn cost(&self, outputs: &Matrix, targets: &Matrix, batch_size: usize) -> f64 {
        // let diff = outputs.subtract(&targets);
        // let squared = diff.map(|x| x * x);
        let cost = (self.cost_function.function)(outputs, targets);
        let sum = cost.sum(None).data[0];

        sum / batch_size as f64
    }

    // For now we will run backprop immediately after feed_forward.
    // We still average out the values and accumulte the gradients for all training examples
    #[allow(non_snake_case)]
    pub fn backpropagate(
        &mut self,
        // Matrix of (n_output, m) shape.
        outputs: Matrix,
        // Matrix of (n_i, m) shape.
        targets: &Matrix,
        // Vector of (n_i, m) shaped matrices. One matrix for each layer.
        a_values: Vec<Matrix>,
        // same as a_values
        z_values: Vec<Matrix>,
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        // This will contain the gradients for every layer. Each layer is represented by a matrix.
        let mut dc_dw: Vec<Matrix> = Vec::with_capacity(self.weights.len());
        let mut dc_db: Vec<Matrix> = Vec::with_capacity(self.bias.len());

        let m = outputs.shape().1 as f64;

        // dC_da
        let mut dC_da = (self.cost_function.derivative)(&outputs, targets); // (n_output, m)

        // Weights: 1, 0
        // Activations: 2, 1, 0
        for i in (1..self.layers.len()).rev() {
            // w_i is used to index weights and biases as there is one less weight/bias than layers
            // activations and weights have the same number as layers so they use i
            let w_i = i - 1;

            // da_dz = g'(z)
            // dC_dz = dC_da * da_dz
            let dC_dz =
                dC_da.multiply_elementwise(&z_values[w_i].map(self.activations[w_i].derivative)); // shape: (n_i, m)

            // dz_dw = a^(L - 1)
            let dz_dw = a_values[i - 1].clone(); // shape: (n_{i-1}, m)

            let dz_da = &self.weights[w_i]; // shape: (n_i, n_{i-1})

            // dC_dw = dC_dz * dz_dw
            let dC_dw_i = dC_dz.matmul(&dz_dw.transpose()); // shape: (n_i, n_{i-1})

            // dz_db = 0
            // dC_db = dC_dz * dz_db
            let dC_db_i = dC_dz.clone();

            // Average the gradients across the batch
            // Weight gradients: (n_i, n_{i-1}) - sum already done via matmul across batch
            dc_dw.push(dC_dw_i.map(|x| x / m)); // shape: (n_i, n_{i-1})
            // Bias gradients: sum across batch examples, then average
            dc_db.push(dC_db_i.sum(Some(1)).map(|x| x / m)); // shape: (n_i, 1)

            // For the next iteration
            // dC_da_L-1 = dC_dz * dz_da
            //
            // The partial derivative of the cost with respect to one of the activations in layer L-1 is a sum of
            // the partial derivatives of the cost with respect to each activation in layer L
            //
            // If im not wrong, the matmul here sums across each activation in layer L since the n_i in both shapes is gone,
            // meaning that the n_i dimension is 'reduced' and the result is (n_{i-1}, m)
            dC_da = dz_da.transpose().matmul(&dC_dz);
        }

        // Reverse the gradients vector as we went through the layers backwards
        dc_dw.reverse();
        dc_db.reverse();
        return (dc_dw, dc_db);
    }

    pub fn gradient_descent(&mut self, dc_dw: Vec<Matrix>, dc_db: Vec<Matrix>, learning_rate: f64) {
        for i in 0..self.weights.len() {
            // w = w - dc_dw * learning_rate
            self.weights[i] = self.weights[i].subtract(&dc_dw[i].map(|x| x * learning_rate));
            // b = b - dc_db * learning_rate
            self.bias[i] = self.bias[i].subtract(&dc_db[i].map(|x| x * learning_rate));
        }
    }

    // Inputs and targest here should be batched
    // inputs is a vector of matrices, where each matrice is a batch of inputs, shaped (batch_size, input_features)
    // targets is a vector of matrices, where each matrice is a batch of targets, shaped (batch_size, target_features)
    /// Returns a vector of costs for each epoch.
    pub fn train(
        &mut self,
        inputs: Vec<Matrix>,
        targets: Vec<Matrix>,
        num_epochs: usize,
        learning_rate: f64,
    ) -> Vec<f64> {
        let num_batches = inputs.len();
        let update_interval = num_epochs / 10;
        let mut history: Vec<f64> = Vec::with_capacity(num_epochs);
        for i in 0..num_epochs {
            let mut total_cost = 0.0;

            for batch in 0..num_batches {
                let batch_size = inputs[batch].rows;
                let (outputs, a_values, z_values) = self.feed_forward(&inputs[batch]);
                let cost = self.cost(&outputs, &targets[batch].transpose(), batch_size);
                total_cost += cost;
                let (dc_dw, dc_db) =
                    self.backpropagate(outputs, &targets[batch].transpose(), a_values, z_values);

                self.gradient_descent(dc_dw, dc_db, learning_rate);
            }
            let total_cost = total_cost / num_batches as f64;

            history.push(total_cost);

            if (i) % update_interval == 0 {
                println!("Epoch: {} / {}, Cost: {}", i, num_epochs, total_cost);
            }
        }
        history
    }

    pub fn evaluate(&self, test_inputs: Matrix, test_targets: Matrix) {
        let (outputs, _, _) = self.feed_forward(&test_inputs);
        let cost = self.cost(&outputs, &test_targets.transpose(), test_inputs.rows);
        println!("Test Cost: {}", cost);
        let mut rng = rand::rng();
        for _ in 0..10 {
            // randomly sample
            let i = rng.random_range(0..test_inputs.rows);
            println!(
                "Predicted: {}, Label: {}",
                outputs.data[i],
                test_targets.transpose().data[i]
            )
        }
    }

    pub fn predict(&self, inputs: Matrix) -> Matrix {
        let (outputs, _, _) = self.feed_forward(&inputs);
        outputs
    }

    pub fn load_weights_and_biases(
        layers: Vec<usize>,
        weights: Vec<Matrix>,
        biases: Vec<Matrix>,
        activations: Vec<Activation>,
        cost_function: LossFunction,
    ) -> Self {
        Self {
            layers,
            weights,
            bias: biases,
            activations,
            cost_function,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        activations::{RELU, SIGMOID},
        loss_functions::MSE,
    };

    #[test]
    fn test_feed_forward_1_1_1() {
        // Test a simple [1, 1, 1] network with fixed weights and biases
        // Input layer: 1 neuron
        // Hidden layer: 1 neuron
        // Output layer: 1 neuron
        //
        // Weights: [layer0: [[2.0]], layer1: [[3.0]]]
        // Biases: [layer0: [[0.5]], layer1: [[0.1]]]
        // Activation: Sigmoid
        //
        // Input: [1.0]
        // Layer 0: z = 2.0 * 1.0 + 0.5 = 2.5, a = relu(2.5) = 2.5
        // Layer 1: z = 3.0 * 2.5 + 0.1 = 7.6, a = relu(7.6) = 7.6

        let weights = vec![
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![2.0],
            },
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![3.0],
            },
        ];

        let biases = vec![
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![0.5],
            },
            Matrix {
                rows: 1,
                cols: 1,
                data: vec![0.1],
            },
        ];

        let nn = NeuralNetwork::load_weights_and_biases(
            vec![1, 1, 1],
            weights,
            biases,
            vec![RELU, RELU],
            MSE,
        );

        let input = Matrix {
            rows: 1,
            cols: 1,
            data: vec![1.],
        };
        let (output, _, _) = nn.feed_forward(&input);

        assert_eq!(output.shape(), (1, 1));
        // Output should be close to 0.947
        assert!(output.data[0] == 7.6);
        println!("{:?}", output.data);
    }

    #[test]
    fn test_feed_forward_2_3_2() {
        // Test a more complex [2, 3, 2] network with fixed weights and biases
        // Input layer: 2 neurons
        // Hidden layer: 3 neurons
        // Output layer: 2 neurons
        //
        // Weights:
        //   layer0 (input->hidden): 3x2 matrix
        //   layer1 (hidden->output): 2x3 matrix
        // Biases:
        //   layer0: 3x1 vector
        //   layer1: 2x1 vector
        // Activation: Sigmoid
        //
        // Input: [1.0, 0.5]^T
        // The forward pass will compute activations through both layers

        let weights = vec![
            // Input to hidden: 3x2
            // Each row represents weights for one hidden neuron from both inputs
            Matrix {
                rows: 3,
                cols: 2,
                data: vec![
                    1.0, 0.5, // Hidden neuron 0: weights from input 0 and 1
                    0.5, 1.0, // Hidden neuron 1: weights from input 0 and 1
                    0.0, 0.5, // Hidden neuron 2: weights from input 0 and 1
                ],
            },
            // Hidden to output: 2x3
            // Each row represents weights for one output neuron from all hidden neurons
            Matrix {
                rows: 2,
                cols: 3,
                data: vec![
                    1.0, 0.5, 0.2, // Output neuron 0: weights from hidden 0, 1, 2
                    0.3, 1.0, 0.1, // Output neuron 1: weights from hidden 0, 1, 2
                ],
            },
        ];

        let biases = vec![
            Matrix {
                rows: 3,
                cols: 1,
                data: vec![0.1, 0.2, 0.3],
            },
            Matrix {
                rows: 2,
                cols: 1,
                data: vec![0.5, 0.3],
            },
        ];

        let nn = NeuralNetwork::load_weights_and_biases(
            vec![2, 3, 2],
            weights,
            biases,
            vec![RELU, SIGMOID],
            MSE,
        );

        let input = Matrix {
            rows: 1,
            cols: 2,
            data: vec![1.0, 0.5],
        };
        let (output, _, _) = nn.feed_forward(&input);

        assert_eq!(output.shape(), (2, 1));
        // Expected outputs with RELU hidden layer and SIGMOID output layer:

        // output[0] ≈ 0.928 (sigmoid(2.56))
        // output[1] ≈ 0.877 (sigmoid(1.96))
        assert!(output.data[0] > 0.92 && output.data[0] < 0.93);
        assert!(output.data[1] > 0.87 && output.data[1] < 0.88);
        println!("{:?}", output.data)
    }

    /// Helper function to create a matrix with direct field initialization
    fn create_matrix(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        Matrix { rows, cols, data }
    }

    /// TEST 1: Train a neural network to learn the AND gate with batch training
    ///
    /// Network Architecture: [2, 3, 1]
    /// - Input layer: 2 neurons (binary inputs)
    /// - Hidden layer: 3 neurons with RELU activation
    /// - Output layer: 1 neuron with SIGMOID activation
    ///
    /// Training Data (AND gate truth table):
    /// Batch (size 4): All 4 examples processed together
    /// (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1
    ///
    /// Assertions:
    /// - All predictions should be close to their targets
    /// - The network should learn the AND gate function
    #[test]
    fn test_train_and_gate() {
        // All 4 AND gate examples in a single batch of size 4
        // Shape: (4 examples, 2 features)
        let batch_inputs = vec![create_matrix(
            4,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )];

        // Shape: (4 examples, 1 target)
        let batch_targets = vec![create_matrix(
            4,
            1,
            vec![
                0.0, // 0 AND 0 = 0
                0.0, // 0 AND 1 = 0
                0.0, // 1 AND 0 = 0
                1.0, // 1 AND 1 = 1
            ],
        )];

        let mut nn = NeuralNetwork::new(vec![2, 3, 1], vec![RELU, SIGMOID], MSE);

        println!("\n=== AND Gate Test (batch_size=4) ===");
        nn.train(batch_inputs.clone(), batch_targets.clone(), 800, 0.5);

        // Test all 4 combinations individually
        let test_cases = vec![
            (create_matrix(1, 2, vec![0.0, 0.0]), 0.0, "(0,0)"),
            (create_matrix(1, 2, vec![0.0, 1.0]), 0.0, "(0,1)"),
            (create_matrix(1, 2, vec![1.0, 0.0]), 0.0, "(1,0)"),
            (create_matrix(1, 2, vec![1.0, 1.0]), 1.0, "(1,1)"),
        ];

        for (input, expected, label) in test_cases {
            let (output, _, _) = nn.feed_forward(&input);
            let predicted = output.data[0];
            let error = (predicted - expected).abs();

            println!(
                "AND{}: predicted={:.4}, target={:.1}, error={:.4}",
                label, predicted, expected, error
            );

            assert!(
                error < 0.3,
                "AND gate test failed for {}: predicted={:.4}, expected={:.1}",
                label,
                predicted,
                expected
            );
        }
    }

    /// TEST 2: Train a neural network to learn the OR gate with batch training
    ///
    /// Network Architecture: [2, 3, 1]
    /// - Input layer: 2 neurons (binary inputs)
    /// - Hidden layer: 3 neurons with RELU activation
    /// - Output layer: 1 neuron with SIGMOID activation
    ///
    /// Training Data (OR gate truth table):
    /// Batch (size 4): All 4 examples processed together
    /// (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→1
    ///
    /// OR gate is typically easier to learn than AND gate.
    ///
    /// Assertions:
    /// - All predictions should be close to their targets
    /// - The network should learn the OR gate function
    #[test]
    fn test_train_or_gate() {
        // All 4 OR gate examples in a single batch of size 4
        // Shape: (4 examples, 2 features)
        let batch_inputs = vec![create_matrix(
            4,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )];

        // Shape: (4 examples, 1 target)
        let batch_targets = vec![create_matrix(
            4,
            1,
            vec![
                0.0, // 0 OR 0 = 0
                1.0, // 0 OR 1 = 1
                1.0, // 1 OR 0 = 1
                1.0, // 1 OR 1 = 1
            ],
        )];

        let mut nn = NeuralNetwork::new(vec![2, 3, 1], vec![RELU, SIGMOID], MSE);

        println!("\n=== OR Gate Test (batch_size=4) ===");
        nn.train(batch_inputs.clone(), batch_targets.clone(), 500, 0.5);

        // Test all 4 combinations individually
        let test_cases = vec![
            (create_matrix(1, 2, vec![0.0, 0.0]), 0.0, "(0,0)"),
            (create_matrix(1, 2, vec![0.0, 1.0]), 1.0, "(0,1)"),
            (create_matrix(1, 2, vec![1.0, 0.0]), 1.0, "(1,0)"),
            (create_matrix(1, 2, vec![1.0, 1.0]), 1.0, "(1,1)"),
        ];

        for (input, expected, label) in test_cases {
            let (output, _, _) = nn.feed_forward(&input);
            let predicted = output.data[0];
            let error = (predicted - expected).abs();

            println!(
                "OR{}: predicted={:.4}, target={:.1}, error={:.4}",
                label, predicted, expected, error
            );

            assert!(
                error < 0.25,
                "OR gate test failed for {}: predicted={:.4}, expected={:.1}",
                label,
                predicted,
                expected
            );
        }
    }

    /// TEST 3: Verify batch training on variable batch sizes
    ///
    /// Network Architecture: [1, 2, 1]
    /// - Input layer: 1 neuron
    /// - Hidden layer: 2 neurons with RELU activation
    /// - Output layer: 1 neuron with SIGMOID activation
    ///
    /// This test verifies that batch training works correctly with
    /// different batch sizes (batch_size=1 and batch_size=2).
    ///
    /// Assertions:
    /// - Cost should decrease during batch training
    /// - Both batch sizes should achieve reasonable performance
    #[test]
    fn test_batch_training_various_sizes() {
        println!("\n=== Batch Training with Various Sizes ===");

        // Training data: mixed batch sizes (1 and 2)
        // Shape: (batch_size, n_features)
        let inputs = vec![
            create_matrix(1, 1, vec![0.3]),
            create_matrix(2, 1, vec![0.5, 0.7]),
        ];
        let targets = vec![
            create_matrix(1, 1, vec![0.2]),
            create_matrix(2, 1, vec![0.4, 0.8]),
        ];

        let mut nn = NeuralNetwork::new(vec![1, 2, 1], vec![RELU, SIGMOID], MSE);

        // Calculate initial cost
        let (initial_out_0, _, _) = nn.feed_forward(&inputs[0]);
        let (initial_out_1, _, _) = nn.feed_forward(&inputs[1]);
        let initial_cost_0 = nn.cost(&initial_out_0, &targets[0].transpose(), 1);
        let initial_cost_1 = nn.cost(&initial_out_1, &targets[1].transpose(), 2);
        let initial_total_cost = initial_cost_0 + initial_cost_1;

        println!("Initial cost: {:.6}", initial_total_cost);

        // Train with mixed batch sizes (1 and 2)
        nn.train(inputs.clone(), targets.clone(), 300, 0.1);

        // Calculate final cost
        let (final_out_0, _, _) = nn.feed_forward(&inputs[0]);
        let (final_out_1, _, _) = nn.feed_forward(&inputs[1]);
        let final_cost_0 = nn.cost(&final_out_0, &targets[0].transpose(), 1);
        let final_cost_1 = nn.cost(&final_out_1, &targets[1].transpose(), 2);
        let final_total_cost = final_cost_0 + final_cost_1;

        println!("Final cost: {:.6}", final_total_cost);
        println!(
            "Cost reduction: {:.6}",
            initial_total_cost - final_total_cost
        );

        // Cost should decrease
        assert!(
            final_total_cost < initial_total_cost,
            "Batch training should decrease cost"
        );
    }

    #[test]
    fn test_train_or_gate_logistic() {
        use crate::loss_functions::LOGISTIC;

        let batch_inputs = vec![create_matrix(
            4,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )];

        let batch_targets = vec![create_matrix(
            4,
            1,
            vec![
                0.0, // 0 OR 0 = 0
                1.0, // 0 OR 1 = 1
                1.0, // 1 OR 0 = 1
                1.0, // 1 OR 1 = 1
            ],
        )];

        let mut nn = NeuralNetwork::new(vec![2, 3, 1], vec![RELU, SIGMOID], LOGISTIC);

        println!("\n=== OR Gate Test Logistic (batch_size=4) ===");
        nn.train(batch_inputs.clone(), batch_targets.clone(), 500, 0.5);

        let test_cases = vec![
            (create_matrix(1, 2, vec![0.0, 0.0]), 0.0, "(0,0)"),
            (create_matrix(1, 2, vec![0.0, 1.0]), 1.0, "(0,1)"),
            (create_matrix(1, 2, vec![1.0, 0.0]), 1.0, "(1,0)"),
            (create_matrix(1, 2, vec![1.0, 1.0]), 1.0, "(1,1)"),
        ];

        for (input, expected, label) in test_cases {
            let (output, _, _) = nn.feed_forward(&input);
            let predicted = output.data[0];
            let error = (predicted - expected).abs();

            println!(
                "OR{}: predicted={:.4}, target={:.1}, error={:.4}",
                label, predicted, expected, error
            );

            assert!(
                error < 0.25,
                "OR gate test (logistic) failed for {}: predicted={:.4}, expected={:.1}",
                label,
                predicted,
                expected
            );
        }
    }
}
