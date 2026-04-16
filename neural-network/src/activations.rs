use std::f64::consts::E;

pub struct Activation {
    pub function: fn(x: f64) -> f64,
    pub derivative: fn(x: f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: |x: f64| 1. / (1. + E.powf(-x)),
    derivative: |x: f64| E.powf(-x) / (1. + E.powf(-x)).powi(2),
};

pub const RELU: Activation = Activation {
    function: |x: f64| if x <= 0. { 0. } else { x },
    derivative: |x: f64| if x > 0. { 1. } else { 0. },
};

pub const LINEAR: Activation = Activation {
    function: |x: f64| x,
    derivative: |_| 1.,
};
