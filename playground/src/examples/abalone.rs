use matrix::matrix::Matrix;
use neural_network::{
    activations::{LINEAR, RELU},
    dataframe::{Dataframe, FeatureTypes},
    loss_functions::MSE,
    nn::NeuralNetwork,
};
use rand::RngExt;

use crate::examples::helpers::plot_cost;

pub fn run_abalone() {
    let file_name = "D:/projects/rust-nn/playground/data/abalone/abalone.data";
    let names = vec![
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole Weight",
        "Shucked Weight",
        "Viscera Weight",
        "Shell Weight",
        "Rings",
    ];

    let feature_types = vec![
        FeatureTypes::OneHot(3, vec!["M", "F", "I"]),
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
    ];

    let df = Dataframe::from_file(file_name, names, 8, feature_types, ",", true).unwrap();
    df.show_example(0);
    df.show_example(4);

    // Save for regularization later
    let (temp_x, _) = df.convert_to_matrix().unwrap();
    // Max and Min along the columns
    let max = temp_x.max(Some(1));
    let min = temp_x.min(Some(1));

    let (train, test) = df.split(0.8);

    test.show_example(1);

    let train = train.batch(100);

    let mut train_x = Vec::new();
    let mut train_y = Vec::new();

    for batch in train {
        let (x, y) = batch.convert_to_matrix().unwrap();

        // Regularization using min-max scaling (x - min) / (max - min)
        let min = min.broadcast_rows(x.rows);
        let max = max.broadcast_rows(x.rows);
        let x = x.subtract(&min).divide_elementwise(&max.subtract(&min));
        train_x.push(x);
        train_y.push(y);
    }

    let mut nn = NeuralNetwork::new(
        vec![10, 64, 64, 1],
        vec![RELU, RELU, LINEAR],
        MSE,
        Box::new(neural_network::optimizers::SGD::new(0.001, 0.0)),
    );

    let history = nn.train(train_x, train_y, 100, 0.2);
    plot_cost(&history, "abalone-cost.png").unwrap();

    // Evaluate
    let (test_x, test_y) = test.convert_to_matrix().unwrap();
    let outputs = nn.evaluate(test_x, test_y.clone());

    // Visualize
    let mut rng = rand::rng();
    for _ in 0..10 {
        // randomly sample
        let i = rng.random_range(0..test_y.rows);
        println!(
            "Predicted: {:?}, Label: {}",
            &outputs.data[i],
            test_y.transpose().data[i]
        )
    }

    // Test Predictions

    // should be 9
    let prediction_input = Matrix::from(
        2,
        10,
        vec![
            1., 0., 0., 0.475, 0.37, 0.125, 0.5095, 0.2165, 0.1125, 0.165, 0., 1., 0., 0.55, 0.415,
            0.135, 0.7635, 0.318, 0.21, 0.2,
        ],
    );

    let prediction = nn.predict(prediction_input);
    println!("Prediction: {:?}", prediction);
}
