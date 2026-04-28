use neural_network::{
    activations::{LINEAR, NONE, RELU},
    dataframe::{Dataframe, FeatureTypes},
    loss_functions::MSE,
    nn::{Layer, NeuralNetwork},
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

    // Save for regularization later
    let (temp_x, temp_y) = df.convert_to_matrix().unwrap();
    // Max and Min along the columns
    let max = temp_x.max(Some(1));
    let min = temp_x.min(Some(1));
    let mean = temp_x.mean(Some(1));

    let max_y = temp_y.max(Some(1));
    let min_y = temp_y.min(Some(1));
    let mean_y = temp_y.mean(Some(1));

    println!("Means: {:?}, {:?}", mean, mean_y);

    let (train, test) = df.split(0.8);

    test.show_example(1);

    let train = train.batch(32);

    let mut train_x = Vec::new();
    let mut train_y = Vec::new();

    for batch in train {
        let (x, y) = batch.convert_to_matrix().unwrap();

        // Regularization using mean normalization (x - mean) / (max - min)
        let min = min.broadcast_rows(x.rows);
        let max = max.broadcast_rows(x.rows);
        let mean = mean.broadcast_rows(x.rows);
        let x = x.subtract(&mean).divide_elementwise(&max.subtract(&min));

        let y = y
            .subtract(&mean_y.broadcast_rows(y.rows))
            .divide_elementwise(
                &max_y
                    .broadcast_rows(y.rows)
                    .subtract(&min_y.broadcast_rows(y.rows)),
            );
        train_x.push(x);
        train_y.push(y);
    }

    println!("{:?}", train_x[0].row(0));

    let mut nn = NeuralNetwork::new(
        vec![
            Layer {
                units: 10,
                activation: NONE,
            },
            Layer {
                units: 64,
                activation: RELU,
            },
            Layer {
                units: 64,
                activation: RELU,
            },
            Layer {
                units: 1,
                activation: LINEAR,
            },
        ],
        MSE,
        Box::new(neural_network::optimizers::Adam::new(
            0.001, None, 0.9, 0.999, 0.0,
        )),
    );

    let history = nn.train(train_x, train_y, 20, 0.2);
    plot_cost(&history, "abalone-cost.png").unwrap();

    // Evaluate
    let (test_x, test_y) = test.convert_to_matrix().unwrap();
    let test_x = test_x
        .subtract(&mean.broadcast_rows(test_x.rows))
        .divide_elementwise(
            &max.broadcast_rows(test_x.rows)
                .subtract(&min.broadcast_rows(test_x.rows)),
        );

    let test_y = test_y
        .subtract(&mean_y.broadcast_rows(test_y.rows))
        .divide_elementwise(
            &max_y
                .broadcast_rows(test_y.rows)
                .subtract(&min_y.broadcast_rows(test_y.rows)),
        );

    let outputs = nn.evaluate(test_x, test_y.clone());

    // Visualize
    let test_y = test_y.transpose();
    let mut rng = rand::rng();
    for _ in 0..10 {
        // randomly sample
        let i = rng.random_range(0..test_y.cols);
        // Convert back from normalization
        let prediction = outputs.data[i] * (max_y.data[0] - min_y.data[0]) + mean_y.data[0];
        let actual = test_y.data[i] * (max_y.data[0] - min_y.data[0]) + mean_y.data[0];
        println!("Predicted: {:?}, Label: {}", prediction, actual)
    }
}
