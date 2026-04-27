// Optimizers will handle all the gradient updates after backpropagation.
// SGD, RMSprop, Adam

use matrix::matrix::Matrix;

pub trait Optimizer {
    fn step(
        &mut self,
        dc_dw: &Vec<Matrix>,
        dc_db: &Vec<Matrix>,
        weights: &mut Vec<Matrix>,
        bias: &mut Vec<Matrix>,
    );

    fn initialize(&mut self, weights: &Vec<Matrix>, bias: &Vec<Matrix>);
}

pub struct SGD {
    pub learning_rate: f64,
    pub weight_decay: f64,
    // For momentum
    momentum: Option<f64>,
    v_dc_dw: Vec<Matrix>,
    v_dc_db: Vec<Matrix>,
}

impl SGD {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            weight_decay,
            momentum: None,
            v_dc_dw: Vec::new(),
            v_dc_db: Vec::new(),
        }
    }

    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = Some(momentum);
        self
    }
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        dc_dw: &Vec<Matrix>,
        dc_db: &Vec<Matrix>,
        weights: &mut Vec<Matrix>,
        bias: &mut Vec<Matrix>,
    ) {
        let decayed = 1. - self.weight_decay;
        for i in 0..weights.len() {
            // Momentum
            if let Some(m) = self.momentum {
                let one_minus_m = 1.0 - m;
                self.v_dc_dw[i] = self.v_dc_dw[i].map(|x| x * m).add(&dc_dw[i]);

                self.v_dc_db[i] = self.v_dc_db[i]
                    .map(|x| x * m)
                    .add(&dc_db[i].map(|x| x * one_minus_m));

                // w = w - v_dc_dw * learning_rate
                weights[i] = weights[i]
                    .map(|x| x * decayed)
                    .subtract(&self.v_dc_dw[i].map(|x| x * self.learning_rate));
                // b = b - v_dc_db * learning_rate
                bias[i] = bias[i].subtract(&self.v_dc_db[i].map(|x| x * self.learning_rate));
            } else {
                // w = w - dc_dw * learning_rate
                weights[i] = weights[i]
                    .map(|x| x * decayed)
                    .subtract(&dc_dw[i].map(|x| x * self.learning_rate));
                // b = b - dc_db * learning_rate
                bias[i] = bias[i].subtract(&dc_db[i].map(|x| x * self.learning_rate));
            }
        }
    }

    fn initialize(&mut self, weights: &Vec<Matrix>, bias: &Vec<Matrix>) {
        // Create a copy of the weights but initialize v_dc_dw and v_dc_db to zero
        self.v_dc_dw = weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();
        self.v_dc_db = bias.iter().map(|b| Matrix::new(b.rows, b.cols)).collect();
    }
}
