use neural_network::{
    activations::{LINEAR, RELU},
    dataframe::{Dataframe, FeatureTypes},
    loss_functions::MSE,
    nn::NeuralNetwork,
};
use rand::RngExt;

use crate::examples::helpers::plot_cost;

pub fn run_fuel_efficiency() {
    let file_name = "./playground/data/auto-mpg/auto-mpg.data";
    let names = vec![
        "displacement",
        "mpg",
        "cylinders",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        // will be dropped as this feature is not relevant
        "car_name",
    ];

    let feature_types = vec![
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        FeatureTypes::Continuous,
        // Using string here is just for the parsing of the data
        // Will be dropped later
        FeatureTypes::String,
    ];

    let mut df = Dataframe::from_file(file_name, names, 1, feature_types, " ", true).unwrap();
    df.show_example(0);

    // The index changed because the swap_remove is used to remove the target.
    df.drop_col(1);
    df.show_example(0);

    // Preprocessing of data
    // Mean normalization
    let (temp_x, _) = df.convert_to_matrix().unwrap();
    let rows = temp_x.rows as f64;
    // Max and Min along the columns
    let max = temp_x.max(Some(2));
    let min = temp_x.min(Some(2));
    let mean = temp_x.sum(Some(2)).map(|x| x / rows);

    println!("Max: {:?}", max);
    println!("Min: {:?}", min);
    println!("Mean: {:?}", mean);

    let (train, test) = df.split(0.8);

    let train = train.batch(100);

    let mut train_x = Vec::new();
    let mut train_y = Vec::new();

    for batch in train {
        let (x, y) = batch.convert_to_matrix().unwrap();

        // Regularization using min-max scaling (x - min) / (max - min)
        let min = min.broadcast_rows(x.rows);
        let max = max.broadcast_rows(x.rows);
        let mean = mean.broadcast_rows(x.rows);
        let x = x.subtract(&mean).divide_elementwise(&max.subtract(&min));
        train_x.push(x);
        train_y.push(y);
    }

    // Define neural network structure
    let mut nn = NeuralNetwork::new(
        vec![train_x[0].cols, 64, 64, 1],
        vec![RELU, RELU, LINEAR],
        MSE,
    );

    let history = nn.train(train_x, train_y, 100, 0.01, 0.2);
    plot_cost(history, "fuel-cost.png").unwrap();

    // Evaluate model
    // Regularize test data using mean from whole set
    let (test_x, test_y) = test.convert_to_matrix().unwrap();
    let test_x = test_x
        .subtract(&mean.broadcast_rows(test_x.rows))
        .divide_elementwise(
            &max.broadcast_rows(test_x.rows)
                .subtract(&min.broadcast_rows(test_x.rows)),
        );

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
}
