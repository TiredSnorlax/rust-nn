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
        timestep: i32,
    );

    fn initialize(&mut self, weights: &Vec<Matrix>, bias: &Vec<Matrix>);
}

pub struct SGD {
    pub learning_rate: f64,
    /// This function takes in timestep and returns the learning rate for that timestep.
    pub learning_rate_decay_fn: Option<Box<dyn Fn(i32, f64) -> f64>>,
    pub weight_decay: f64,
    // For momentum
    momentum: Option<f64>,
    v_dc_dw: Vec<Matrix>,
    v_dc_db: Vec<Matrix>,
}

impl SGD {
    pub fn new(
        learning_rate: f64,
        learning_rate_decay_fn: Option<Box<dyn Fn(i32, f64) -> f64>>,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            learning_rate_decay_fn,
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
        timestep: i32,
    ) {
        let decayed = 1. - self.weight_decay;
        for i in 1..weights.len() {
            // Momentum
            if let Some(m) = self.momentum {
                let one_minus_m = 1.0 - m;
                self.v_dc_dw[i] = self.v_dc_dw[i]
                    .map(|x| x * m)
                    .add(&dc_dw[i].map(|x| x * one_minus_m));

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
        if let Some(f) = &self.learning_rate_decay_fn {
            self.learning_rate = (f)(timestep, self.learning_rate);
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

pub struct RMSprop {
    pub learning_rate: f64,
    pub learning_rate_decay_fn: Option<Box<dyn Fn(i32, f64) -> f64>>,
    pub weight_decay: f64,
    pub beta: f64,
    s_dc_dw: Vec<Matrix>,
    s_dc_db: Vec<Matrix>,
}

impl RMSprop {
    pub fn new(
        learning_rate: f64,
        learning_rate_decay_fn: Option<Box<dyn Fn(i32, f64) -> f64>>,
        beta: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            learning_rate_decay_fn,
            weight_decay,
            beta,
            s_dc_dw: Vec::new(),
            s_dc_db: Vec::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(
        &mut self,
        dc_dw: &Vec<Matrix>,
        dc_db: &Vec<Matrix>,
        weights: &mut Vec<Matrix>,
        bias: &mut Vec<Matrix>,
        timestep: i32,
    ) {
        let decayed = 1. - self.weight_decay;
        let epsilon = 1e-8;
        for i in 1..weights.len() {
            let one_minus_beta = 1.0 - self.beta;
            self.s_dc_dw[i] = self.s_dc_dw[i]
                .map(|x| x * self.beta)
                .add(&dc_dw[i].map(|x| (x.powi(2)) * one_minus_beta));

            self.s_dc_db[i] = self.s_dc_db[i]
                .map(|x| x * self.beta)
                .add(&dc_db[i].map(|x| (x.powi(2)) * one_minus_beta));

            // w = w - learning_rate * (dW / sqrt(s_dc_dw) + epsilon)
            let right_denominator_w = self.s_dc_dw[i].map(|x| x.sqrt() + epsilon);
            let right_term_w = dc_dw[i].divide_elementwise(&right_denominator_w);

            let right_denominator_b = self.s_dc_db[i].map(|x| x.sqrt() + epsilon);
            let right_term_b = dc_db[i].divide_elementwise(&right_denominator_b);

            weights[i] = weights[i]
                .map(|x| x * decayed)
                .subtract(&right_term_w.map(|x| x * self.learning_rate));
            // b = b - v_dc_db * learning_rate
            bias[i] = bias[i].subtract(&right_term_b.map(|x| x * self.learning_rate));
        }
        if let Some(f) = &self.learning_rate_decay_fn {
            self.learning_rate = (f)(timestep, self.learning_rate);
        }
    }

    fn initialize(&mut self, weights: &Vec<Matrix>, bias: &Vec<Matrix>) {
        self.s_dc_dw = weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();
        self.s_dc_db = bias.iter().map(|b| Matrix::new(b.rows, b.cols)).collect();
    }
}

pub struct Adam {
    pub learning_rate: f64,
    pub learning_rate_decay_fn: Option<Box<dyn Fn(i32, f64) -> f64>>,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,

    v_dc_dw: Vec<Matrix>,
    v_dc_db: Vec<Matrix>,
    s_dc_dw: Vec<Matrix>,
    s_dc_db: Vec<Matrix>,
}

impl Adam {
    pub fn new(
        learning_rate: f64,
        learning_rate_decay_fn: Option<Box<dyn Fn(i32, f64) -> f64>>,
        beta1: f64,
        beta2: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            learning_rate_decay_fn,
            beta1,
            beta2,
            weight_decay,
            v_dc_dw: Vec::new(),
            v_dc_db: Vec::new(),
            s_dc_dw: Vec::new(),
            s_dc_db: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn initialize(&mut self, weights: &Vec<Matrix>, bias: &Vec<Matrix>) {
        // Create a copy of the weights but initialize v_dc_dw and v_dc_db to zero
        self.v_dc_dw = weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();
        self.v_dc_db = bias.iter().map(|b| Matrix::new(b.rows, b.cols)).collect();

        self.s_dc_dw = weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();
        self.s_dc_db = bias.iter().map(|b| Matrix::new(b.rows, b.cols)).collect();
    }

    fn step(
        &mut self,
        dc_dw: &Vec<Matrix>,
        dc_db: &Vec<Matrix>,
        weights: &mut Vec<Matrix>,
        bias: &mut Vec<Matrix>,
        timestep: i32,
    ) {
        let decayed = 1. - self.weight_decay;
        let epsilon = 1e-8;
        for i in 1..weights.len() {
            let one_minus_beta1 = 1.0 - self.beta1;
            let one_minus_beta2 = 1.0 - self.beta2;

            // Update first moments (EMA of gradients)
            self.v_dc_dw[i] = self.v_dc_dw[i]
                .map(|x| x * self.beta1)
                .add(&dc_dw[i].map(|x| x * one_minus_beta1));

            self.v_dc_db[i] = self.v_dc_db[i]
                .map(|x| x * self.beta1)
                .add(&dc_db[i].map(|x| x * one_minus_beta1));

            // Update second moments (EMA of squared gradients)
            self.s_dc_dw[i] = self.s_dc_dw[i]
                .map(|x| x * self.beta2)
                .add(&dc_dw[i].map(|x| (x.powi(2)) * one_minus_beta2));

            self.s_dc_db[i] = self.s_dc_db[i]
                .map(|x| x * self.beta2)
                .add(&dc_db[i].map(|x| (x.powi(2)) * one_minus_beta2));

            // Bias correction
            let v_corrected_w = self.v_dc_dw[i].map(|x| x / (1. - self.beta1.powi(timestep)));
            let v_corrected_b = self.v_dc_db[i].map(|x| x / (1. - self.beta1.powi(timestep)));
            let s_corrected_w = self.s_dc_dw[i].map(|x| x / (1. - self.beta2.powi(timestep)));
            let s_corrected_b = self.s_dc_db[i].map(|x| x / (1. - self.beta2.powi(timestep)));

            // Compute update terms
            let right_denominator_w = s_corrected_w.map(|x| x.sqrt() + epsilon);
            let right_term_w = v_corrected_w.divide_elementwise(&right_denominator_w);

            let right_denominator_b = s_corrected_b.map(|x| x.sqrt() + epsilon);
            let right_term_b = v_corrected_b.divide_elementwise(&right_denominator_b);

            // Apply updates
            weights[i] = weights[i]
                .map(|x| x * decayed)
                .subtract(&right_term_w.map(|x| x * self.learning_rate));

            bias[i] = bias[i].subtract(&right_term_b.map(|x| x * self.learning_rate));
        }
        if let Some(f) = &self.learning_rate_decay_fn {
            self.learning_rate = (f)(timestep, self.learning_rate);
        }
    }
}

pub fn default_learning_rate_decay(step_size: i32, gamma: f64) -> Box<dyn Fn(i32, f64) -> f64> {
    Box::new(move |timestep: i32, prev: f64| -> f64 {
        if timestep % step_size == 0 {
            prev * gamma
        } else {
            prev
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_rate_decay() {
        let initial_lr = 0.1;
        let step_size = 10;
        let gamma = 0.5;

        let decay_fn = default_learning_rate_decay(step_size, gamma);
        let mut optimizer = SGD::new(initial_lr, Some(decay_fn), 0.0);

        // Setup dummy weights and biases
        let weights = vec![Matrix::new(0, 0), Matrix::new(2, 2)];
        let bias = vec![Matrix::new(0, 0), Matrix::new(2, 1)];
        optimizer.initialize(&weights, &bias);

        let mut mut_weights = weights.clone();
        let mut mut_bias = bias.clone();
        let dc_dw = vec![Matrix::new(0, 0), Matrix::new(2, 2)];
        let dc_db = vec![Matrix::new(0, 0), Matrix::new(2, 1)];

        // Simulate 25 timesteps
        for t in 1..=25 {
            optimizer.step(&dc_dw, &dc_db, &mut mut_weights, &mut mut_bias, t);

            if t < 10 {
                assert_eq!(
                    optimizer.learning_rate, 0.1,
                    "LR should not decay before t=10"
                );
            } else if t < 20 {
                assert_eq!(
                    optimizer.learning_rate, 0.05,
                    "LR should decay once at t=10"
                );
            } else {
                assert_eq!(
                    optimizer.learning_rate, 0.025,
                    "LR should decay twice at t=20"
                );
            }
        }

        assert_eq!(optimizer.learning_rate, 0.1 * 0.5 * 0.5);
    }
}
