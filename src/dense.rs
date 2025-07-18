use std::array;

use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

pub struct Neuron<const SIZE: usize> {
    weights: [f64; SIZE],
    bias: f64,
}

impl<const SIZE: usize> Neuron<SIZE> {
    pub const fn new(weights: [f64; SIZE], bias: f64) -> Self {
        Self { weights, bias }
    }

    pub const fn forward(&self, input: [f64; SIZE]) -> f64 {
        let mut result = self.bias;
        let mut i = 0;
        while i < SIZE {
            result += input[i] * self.weights[i];
            i += 1;
        }
        result
    }
}

impl<const SIZE: usize> Distribution<Neuron<SIZE>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Neuron<SIZE> {
        Neuron::new(array::from_fn(|_| rng.random_range(-0.01..0.01)), 0.0)
    }
}

pub struct Layer<const INPUT: usize, const OUTPUT: usize> {
    neurons: [Neuron<INPUT>; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> Layer<INPUT, OUTPUT> {
    pub const fn new(neurons: [Neuron<INPUT>; OUTPUT]) -> Self {
        Self { neurons }
    }

    pub const fn forward_sample(&self, input: [f64; INPUT]) -> [f64; OUTPUT] {
        let mut outputs = [0.0; OUTPUT];
        let mut i = 0;
        while i < OUTPUT {
            outputs[i] = self.neurons[i].forward(input);
            i += 1;
        }
        outputs
    }

    pub fn forward_batch(
        &self,
        input: impl IntoIterator<Item = [f64; INPUT]>,
    ) -> impl Iterator<Item = [f64; OUTPUT]> {
        input.into_iter().map(|sample| self.forward_sample(sample))
    }
}

impl<const INPUT: usize, const OUTPUT: usize> Distribution<Layer<INPUT, OUTPUT>>
    for StandardUniform
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Layer<INPUT, OUTPUT> {
        Layer::new(array::from_fn(|_| rng.random()))
    }
}

#[cfg(test)]
mod tests {
    use super::{Layer, Neuron};
    use crate::{batch_equal, f64_equal, sample_equal};

    #[test]
    const fn a_single_neuron() {
        let neuron = Neuron::new([0.2, 0.8, -0.5], 2.0);
        let output = neuron.forward([1.0, 2.0, 3.0]);
        assert!(f64_equal(output, 2.3));
    }

    #[test]
    const fn a_larger_neuron() {
        let neuron = Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0);
        let output = neuron.forward([1.0, 2.0, 3.0, 2.5]);
        assert!(f64_equal(output, 4.8));
    }

    #[test]
    const fn a_layer_of_neurons() {
        let layer = Layer::new([
            Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
            Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
            Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
        ]);
        let outputs = layer.forward_sample([1.0, 2.0, 3.0, 2.5]);
        let expected = [4.8, 1.21, 2.385];
        assert!(outputs.len() == expected.len());
        assert!(sample_equal(outputs, expected));
    }

    #[test]
    fn a_layer_of_neurons_and_batch_of_data() {
        let layer = Layer::new([
            Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
            Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
            Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
        ]);
        let output = layer.forward_batch([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ]);
        let expected = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]];
        assert!(batch_equal(output, expected));
    }

    #[test]
    fn second_layer() {
        let layer = Layer::new([
            Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
            Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
            Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
        ]);
        let layer2 = Layer::new([
            Neuron::new([0.1, -0.14, 0.5], -1.0),
            Neuron::new([-0.5, 0.12, -0.33], 2.0),
            Neuron::new([-0.44, 0.73, -0.13], -0.5),
        ]);
        let output = layer2.forward_batch(layer.forward_batch([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ]));
        let expected = [
            [0.5031, -1.04185, -2.03875],
            [0.2434, -2.7332, -5.7633],
            [-0.99314, 1.41254, -0.35655],
        ];
        assert!(batch_equal(output, expected));
    }
}
