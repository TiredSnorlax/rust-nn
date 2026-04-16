# Rust Neural Network (rust-nn)

A from-scratch implementation of a neural network library in Rust, including a custom matrix math library and a data processing utility.

This is just a project for learning. There is no real/significant optimization in place. The "vectorization" is still just a for loop.

## Project Structure

This workspace consists of three main crates:

- **`matrix`**: A low-level linear algebra library providing the `Matrix` struct and essential operations.
- **`neural-network`**: A high-level library for building, training, and evaluating neural networks.
- **`playground`**: An example application demonstrating the use of both libraries, specifically applied to the Abalone dataset.

The only dependency for **matrix** and **neural-network** is the **rand** crate.

## Features

### Matrix Library (`matrix`)
- Matrix operations: `add`, `subtract`, `multiply_elementwise`, `divide_elementwise`.
- Matrix multiplication (`matmul`) and `transpose`.
- Reductions: `sum`, `max`, `min` (supports global and axis-wise operations).
- Broadcasting: `broadcast_cols` and `broadcast_rows` for flexible arithmetic.
- Randomized initialization with range support.

### Neural Network Library (`neural-network`)
- **Flexible Architecture**: Create networks with any number of layers and neurons.
- **Activation Functions**:
  - `SIGMOID`
  - `RELU`
  - `LINEAR`
- **Loss Functions**:
  - `MSE` (Mean Squared Error)
  - `LOGISTIC` (Binary Cross-Entropy) with numerical stability clipping.
- **Training**:
  - Backpropagation with automated gradient calculation.
  - Gradient Descent optimizer.
  - Batch training support.
- **Data Utilities**:
  - CSV parsing and processing.
  - One-hot encoding for categorical features.
  - Data splitting (train/test) and batching.

### Usage Example

```rust
use matrix::matrix::Matrix;
use neural_network::{
    activations::{RELU, SIGMOID},
    loss_functions::MSE,
    nn::NeuralNetwork,
};

fn main() {
    // Define a 2-3-1 network
    let mut nn = NeuralNetwork::new(
        vec![2, 3, 1], 
        vec![RELU, SIGMOID], 
        MSE
    );

    // Training data (XOR example)
    let inputs = vec![Matrix::from(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])];
    let targets = vec![Matrix::from(4, 1, vec![0.0, 1.0, 1.0, 0.0])];

    // Train for 1000 epochs
    nn.train(inputs, targets, 1000, 0.1);

    // Predict
    let prediction = nn.predict(Matrix::from(1, 2, vec![1.0, 0.0]));
    println!("Prediction: {:?}", prediction.data);
}
```
