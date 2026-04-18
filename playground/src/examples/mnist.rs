use neural_network::{
    activations::{RELU, SOFTMAX},
    dataframe::{Dataframe, FeatureTypes},
    loss_functions::SPARSE_CATEGORICAL_CROSSENTROPY,
    nn::NeuralNetwork,
};
use rand::RngExt;

use crate::examples::helpers::{plot_cost, plot_image};

pub fn run_mnist() {
    let train_file_name = "./playground/data/mnist/mnist_train.csv";
    let test_file_name = "./playground/data/mnist/mnist_test.csv";

    let names = vec![""; 785];

    let feature_types = vec![FeatureTypes::Continuous; 785];

    // Note that the last pixel here is swapped to the first pixel.
    // This is due to the swap_remove used that runs in O(1) time instead of remove
    // This should not affect the learning process, just how we humans view the image
    let train_df = Dataframe::from_file(
        train_file_name,
        names.clone(),
        0,
        feature_types.clone(),
        ",",
        false,
    )
    .unwrap();
    let test_df =
        Dataframe::from_file(test_file_name, names, 0, feature_types, ",", false).unwrap();

    let train = train_df.batch(100);

    // No need to batch test
    let mut train_x = Vec::new();
    let mut train_y = Vec::new();
    // convert pixel values from 0-255 -> 0-1
    for batch in train {
        let (x, y) = batch.convert_to_matrix().unwrap();

        train_x.push(x.map(|x| x / 255.0));
        train_y.push(y);
    }

    // convert pixel values from 0-255 -> 0-1
    let (test_x, test_y) = test_df.convert_to_matrix().unwrap();
    let test_x = test_x.map(|x| x / 255.0);

    // Construct neural network
    let mut nn = NeuralNetwork::new(
        vec![train_x[0].cols, 128, 10],
        vec![RELU, SOFTMAX],
        SPARSE_CATEGORICAL_CROSSENTROPY,
    );

    let history = nn.train(train_x, train_y, 10, 0.001, 0.2);
    plot_cost(history, "mnist-cost.png").unwrap();

    // Evaluate model
    // Regularize test data using mean from whole set

    let outputs = nn.evaluate(test_x.clone(), test_y.clone());

    // Visualize
    // let mut rng = rand::rng();
    let mut wrong_indexes = Vec::new();

    for i in 0..outputs.rows {
        let pred = outputs
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        let label = test_y.transpose().data[i] as usize;
        if pred != label {
            wrong_indexes.push((i, pred, label));
        }
    }

    println!(
        "Accuracy: {}",
        1.0 - wrong_indexes.len() as f64 / outputs.rows as f64
    );

    println!("Wrong indexes: {:?}", wrong_indexes);

    // Visualize the errors
    // Sample randomly
    let mut rng = rand::rng();
    for _ in 0..15 {
        let index = rng.random_range(0..wrong_indexes.len());
        let (i, pred, label) = wrong_indexes[index];
        let image = test_x.row(i);
        plot_image(
            &format!("wrong-mnist/{}.png", i),
            &image,
            4,
            format!("P: {:?}, L: {}", pred, label),
        )
        .unwrap();
    }
}
