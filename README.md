# Rust Neural Network (rust-nn)

A from-scratch implementation of a neural network library in Rust, including a custom matrix math library and a data processing utility.

This is just a project for learning. There is no real/significant optimization in place. The "vectorization" is still just a for loop.

## Project Structure

This workspace consists of three main crates:

- **`matrix`**: A low-level linear algebra library providing the `Matrix` struct and essential operations.
- **`neural-network`**: A high-level library for building, training, and evaluating neural networks. This uses the matrix library.
- **`playground`**: A collection of examples showcasing the neural network in different problems.
The only dependency for **matrix** and **neural-network** is the **rand** crate.

## Features

### Matrix Library (`matrix`)
- Matrix operations: `add`, `subtract`, `multiply_elementwise`, `divide_elementwise`.
- Matrix multiplication (`matmul`) and `transpose`.
- Reductions: `sum`, `max`, `min` (supports global and axis-wise operations).
- Broadcasting: `broadcast_cols` and `broadcast_rows` for flexible arithmetic.
- Randomized initialization with range support.
  - Normal distribution as well

### Neural Network Library (`neural-network`)
- **Flexible Architecture**: Create networks with any number of layers and neurons.
- **Activation Functions**:
  - `SIGMOID`
  - `RELU`
  - `LINEAR`
- **Loss Functions**:
  - `MSE` (Mean Squared Error)
  - `Binary Cross-Entropy` 
  - `Sparse Categorical Cross-Entropy` 
- **Training**:
  - Backpropagation with automated gradient calculation.
  - Different optimizers.
    1. Gradient descent with momentum
    2. RMSprop
    3. Adam
  - Regularization
  - Weight Decay
  - Custom Learning Rate decay
- **Data Utilities**:
  - CSV parsing and processing.
  - One-hot encoding for categorical features.
  - Data splitting (train/test) and batching.

## Examples

The `playground` crate contains several examples demonstrating the library's capabilities on real-world datasets.
