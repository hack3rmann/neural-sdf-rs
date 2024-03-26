use crate::{math::{Vector, Matrix}, util::*};
use rayon::iter::IndexedParallelIterator;
use static_assertions::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;



/// Sin layer constructor struct.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[derive(Serialize, Deserialize)]
pub struct Sin;

impl From<Sin> for LayerType {
    fn from(_: Sin) -> Self {
        Self::Sin
    }
}



/// Sigmoid layer constructor struct.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[derive(Serialize, Deserialize)]
pub struct Sigmoid;

impl From<Sigmoid> for LayerType {
    fn from(_: Sigmoid) -> Self {
        Self::Sigmoid
    }
}



/// Relu layer constructor struct
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[derive(Serialize, Deserialize)]
pub struct Relu;

impl From<Relu> for LayerType {
    fn from(_: Relu) -> Self {
        Self::Relu
    }
}



/// Dense layer constructor struct
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[derive(Serialize, Deserialize)]
pub struct Dense {
    pub inputs: usize,
    pub outputs: usize,
    pub initializer: Initializer,
}

impl Dense {
    /// Constructs new dense layer constructor with `initializer = Siren`
    pub const fn new(inputs: usize, outputs: usize) -> Self {
        Self { inputs, outputs, initializer: Initializer::Siren }
    }

    /// Configures initializer of the dense layer
    pub const fn initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }
}

impl From<Dense> for LayerType {
    fn from(Dense { inputs, outputs, initializer }: Dense) -> Self {
        Self::Dense { inputs, outputs, initializer }
    }
}



/// The type of network layer
#[derive(Clone, PartialEq, Copy, Hash, Debug, Default)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "layer_type")]
pub enum LayerType {
    #[default]
    Id,
    Sin,
    Sigmoid,
    Relu,
    Dense { inputs: usize, outputs: usize, initializer: Initializer },
}
assert_impl_all!(LayerType: Send, Sync);

impl LayerType {
    /// Returns `Some(..)` with activation function if `self` is such
    pub const fn to_activation_fn(self) -> Option<ActivationFunction> {
        use ActivationFunction::*;

        Some(match self {
            Self::Id => Id,
            Self::Sin => Sin,
            Self::Sigmoid => Sigmoid,
            Self::Relu => Relu,
            Self::Dense { .. } => return None,
        })
    }
}



/// Network initializer
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Initializer {
    #[default]
    Siren,
}
assert_impl_all!(Initializer: Send, Sync);

impl Initializer {
    /// Executes an initializer on `transition`
    pub fn execute(self, transition: &mut DenseLayerTransition) {
        match self {
            Self::Siren => Self::siren_init(transition),
        }
    }

    fn siren_init(transition: &mut DenseLayerTransition) {
        use rand::prelude::*;

        transition.biases.fill(0.0);

        let mut rng = rand::thread_rng();
        let range = f32::sqrt(6.0 / transition.input_len() as f32)
            / ActivationFunction::SIN_SCALE;

        transition.weights.values.fill_with(|| {
            2.0 * range * rng.gen::<f32>() - range
        });
    }
}



#[derive(Clone, Debug, Default, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct NetworkBuilder {
    pub layer_transitions: Vec<LayerType>,
}
assert_impl_all!(NetworkBuilder: Send, Sync);

impl NetworkBuilder {
    /// Constructs new network builder
    pub const fn new() -> Self {
        Self { layer_transitions: vec![] }
    }

    /// Adds a layer to network
    pub fn layer(&mut self, layer: impl Into<LayerType>) -> &mut Self {
        self.layer_transitions.push(layer.into());
        self
    }

    /// Adds a bunch of layers to network
    pub fn layers(&mut self, layers: impl IntoIterator<Item = LayerType>)
        -> &mut Self
    {
        self.layer_transitions.extend(layers);
        self
    }

    /// Builds new network
    /// 
    /// # Panic
    /// 
    /// Panics if layer transition sizes do not match or
    /// activation functions are not placed between transitions
    pub fn build(&self) -> Network {
        let is_layer_structure_preserved = self.layer_transitions
            .windows(2)
            .all(|window| {
                use LayerType::*;

                let &[prev, cur] = window else { panic!() };

                let different = matches!(prev, Relu | Sigmoid | Sin)
                    ^ matches!(cur, Relu | Sigmoid | Sin);
                
                let non_identity = prev != Id && cur != Id;

                different && non_identity
            });
        
        assert_ne!(self.layer_transitions.len(), 0);

        assert!(matches!(
            self.layer_transitions.first().unwrap(),
            LayerType::Dense { .. },
        ));

        assert!(matches!(
            self.layer_transitions.last().unwrap(),
            LayerType::Dense { .. },
        ));

        assert!(
            is_layer_structure_preserved,
            "layer structure does not preserved",
        );

        let transitions = self.layer_transitions.chunks(2).map(|chunk| {
            let LayerType::Dense {
                inputs, outputs, initializer,
            } = chunk[0] else {
                unreachable!()
            };

            let dense = DenseLayerTransition::from_initializer(
                inputs, outputs, initializer,
            );

            let activation_fn = chunk.get(1)
                .copied()
                .unwrap_or_default()
                .to_activation_fn()
                .unwrap();

            LayerTransition { dense, activation_fn }
        }).collect::<Vec<_>>();

        let are_transitions_correct = transitions.windows(2).all(|window| {
            let [prev, cur] = window else { unreachable!() };
            prev.output_len() == cur.input_len()
        });

        assert!(
            are_transitions_correct,
            "layer transitions should have matching sizes",
        );

        Network { transitions }
    }
}



/// Activation function type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationFunction {
    Sin,
    Sigmoid,
    Relu,
    #[default]
    Id,
}
assert_impl_all!(ActivationFunction: Send, Sync);

impl ActivationFunction {
    /// Sine activation function parameter
    pub const SIN_SCALE: f32 = 30.0;

    /// Gives activation function evaluator
    pub const fn evaluator(self) -> fn(f32) -> f32 {
        match self {
            Self::Sin => |value| f32::sin(Self::SIN_SCALE * value),
            Self::Sigmoid => |value| 1.0 / (1.0 + f32::exp(-value)),
            Self::Relu => |value| value.max(0.0),
            Self::Id => |value| value,
        }
    }

    /// Gives activation function derivative evaluator
    /// 
    /// # Note
    /// 
    /// Due to possible use of activation function value
    /// it has two parameters: input and maybe activation function value
    pub fn differential(self) -> fn(f32, Option<f32>) -> f32 {
        match self {
            Self::Sin => |value, _|
                Self::SIN_SCALE * f32::cos(Self::SIN_SCALE * value),
            Self::Sigmoid => |value, prev| match prev {
                Some(prev) => prev * (1.0 - prev),
                None => {
                    let exp = f32::exp(-value);
                    exp / (exp + 1.0).powi(2)
                },
            },
            Self::Relu => |value, _| if value < 0.0 { 0.0 } else { 1.0 },
            Self::Id => |_, _| 1.0,
        }
    }
}



/// Dense transition between network layers
#[derive(Clone, PartialEq, Default, Debug)]
#[derive(Serialize, Deserialize)]
pub struct DenseLayerTransition {
    pub weights: Matrix,
    pub biases: Vector,
}
assert_impl_all!(DenseLayerTransition: Send, Sync);

impl DenseLayerTransition {
    /// Constructs new [dense layer transition](DenseLayerTransition)
    pub const fn new(weights: Matrix, biases: Vector) -> Self {
        Self { weights, biases }
    }

    /// Partially applies transition to `vector` by currying
    pub const fn applicator<'s>(&'s self, vector: &'s Vector)
        -> impl Fn(usize) -> f32 + Send + Sync + 's
    {
        |i| self.weights.vector_multiplicator(vector)(i) + self.biases[i]
    }

    /// Constructs [dense layer transition](DenseLayerTransition)
    /// and initializes it with `init` method
    pub fn from_initializer(
        n_inputs: usize, n_outputs: usize, init: Initializer,
    ) -> Self {
        let weights = Matrix::new_zeroed(n_outputs as u32, n_inputs as u32);
        let biases = Vector::new_zeroed(n_outputs);

        let mut result = Self { weights, biases };

        init.execute(&mut result);

        result
    }

    /// Number of output neurons
    pub const fn output_len(&self) -> usize {
        self.weights.n_rows as usize
    }

    /// Number of input neurons
    pub const fn input_len(&self) -> usize {
        self.weights.n_columns as usize
    }

    pub fn par_values_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut f32> + '_ {
        use rayon::prelude::*;

        self.weights.values.par_iter_mut().chain(self.biases.par_iter_mut())
    }

    pub fn par_values(&self) -> impl IndexedParallelIterator<Item = f32> + '_ {
        use rayon::prelude::*;

        self.weights.values.par_iter().chain(self.biases.par_iter()).copied()
    }
}



/// Transition between network layers
#[derive(Clone, PartialEq, Debug)]
#[derive(Serialize, Deserialize)]
pub struct LayerTransition {
    pub activation_fn: ActivationFunction,
    pub dense: DenseLayerTransition,
}
assert_impl_all!(LayerTransition: Send, Sync);

impl LayerTransition {
    /// Applies transition to `from` writing it to `to`.
    pub fn apply_from(&self, from: &Layer, to: &mut Layer) {
        use rayon::prelude::*;

        to.resize(self.output_len(), 0.0);

        to.input.par_iter_mut().enumerate().for_each(|(i, value)| {
            *value = self.dense.applicator(&from.activation)(i)
        });

        to.activation.par_iter_mut().enumerate().for_each(|(i, value)| {
            *value = self.activation_fn.evaluator()(to.input[i])
        });
    }

    pub fn mul_add(&mut self, other: &Self, mul: f32) {
        use rayon::prelude::*;

        self.dense.par_values_mut()
            .zip(other.par_values())
            .for_each(move |(this, other)| {
                *this += mul * other;
            });
    }
}

impl std::ops::Deref for LayerTransition {
    type Target = DenseLayerTransition;

    fn deref(&self) -> &Self::Target {
        &self.dense
    }
}

impl std::ops::DerefMut for LayerTransition {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dense
    }
}



/// Input and output neuron activations
#[derive(Clone, PartialEq, Debug, Default)]
#[derive(Serialize, Deserialize)]
pub struct Layer {
    /// Values before activation function
    pub input: Vector,
    /// Values after activation function
    pub activation: Vector,
}
assert_impl_all!(Layer: Send, Sync);

impl Layer {
    pub fn resize(&mut self, len: usize, value: f32) {
        self.input.resize(len, value);
        self.activation.resize(len, value);
    }
}



#[derive(Clone, PartialEq, Debug, Default)]
#[derive(Serialize, Deserialize)]
pub struct Neurons {
    /// All network activations layerwise
    pub layers: Vec<Layer>,
}
assert_impl_all!(Neurons: Send, Sync);

impl Neurons {
    /// *Mean Squared Error* between last layer activation and the `expectation`
    pub fn mse(&self, expectation: &Vector) -> f32 {
        self.layers.last().unwrap().activation.mse(expectation)
    }
}



#[derive(Clone, Debug, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct NetworkLayout {
    pub layer_sizes: Arc<[usize]>,
}

impl NetworkLayout {
    pub fn from_network(network: &Network) -> Self {
        let mut layer_sizes = Vec::with_capacity(network.n_layers());

        layer_sizes.push(network.transitions.first().unwrap().input_len());

        for transition in &network.transitions {
            layer_sizes.push(transition.output_len());
        }

        Self { layer_sizes: layer_sizes.into() }
    }

    pub fn transitions(&self)
        -> impl ExactSizeIterator<Item = (usize, usize)> + '_
    {
        self.layer_sizes.windows(2)
            .map(|window| {
                let &[input, output] = window else { unreachable!() };
                (input, output)
            })
    }

    pub fn allocate_gradient(&self) -> Network {
        let transitions = self.transitions().map(|(input_len, output_len)| {
            let dense = DenseLayerTransition {
                weights: Matrix::new_zeroed(output_len as u32, input_len as u32),
                biases: Vector::new_zeroed(output_len),
            };

            let activation_fn = ActivationFunction::Id;

            LayerTransition { dense, activation_fn }
        }).collect::<Vec<_>>();

        Network { transitions }
    }

    pub fn allocate_output_buffer(&self) -> Neurons {
        let layers = self.layer_sizes.iter().map(|&size| {
            let input = Vector::new_zeroed(size);
            let activation = input.clone();

            Layer { input, activation }
        }).collect();

        Neurons { layers }
    }

    pub fn allocate_propagation_buffer(&self) -> Vec<Vector> {
        self.layer_sizes.iter().copied().map(Vector::new_zeroed).collect()
    }
}

impl From<&Network> for NetworkLayout {
    fn from(value: &Network) -> Self {
        Self::from_network(value)
    }
}



#[derive(Clone, Debug, PartialEq, Default)]
#[derive(Serialize, Deserialize)]
pub struct Network {
    pub transitions: Vec<LayerTransition>,
}
assert_impl_all!(Network: Send, Sync);

impl Network {
    pub const fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    /// Collects layout information of the network to do allocations
    pub fn layout(&self) -> NetworkLayout {
        NetworkLayout::from_network(self)
    }

    /// Number of neural layers in the network
    pub fn n_layers(&self) -> usize {
        self.transitions.len() + 1
    }

    /// Fills network parameters with `value`
    pub fn fill(&mut self, value: f32) {
        use rayon::prelude::*;

        self.transitions.par_iter_mut()
            .flat_map(|transition| transition.par_values_mut())
            .for_each(|param| *param = value);
    }

    /// Executes network writing the results of all activations and inputs
    pub fn execute(&self, input: &Vector, results: &mut Neurons) {
        results.layers.resize_with(self.n_layers(), default);

        results.layers[0].input.clone_from(input);
        results.layers[0].activation.clone_from(input);

        for (i, transition) in self.transitions.iter().enumerate() {
            let (cur, next) = results.layers.get_two_mut(i, i + 1).unwrap();
            transition.apply_from(cur, next);
        }
    }

    /// Backward propagation algorithm
    pub fn propagate_back(
        &self, gradient: &mut Self, expectation: &Vector, results: &Neurons,
        buffer: &mut Vec<Vector>,
    ) {
        use rayon::prelude::*;

        assert_eq!(
            self.transitions.last().unwrap().output_len(),
            expectation.dimension(),
        );

        assert_eq!(
            expectation.dimension(),
            results.layers.last().unwrap().activation.dimension(),
        );

        // `buf` will contain deltas of each layer
        buffer.resize_with(self.n_layers(), Default::default);

        let last = buffer.last_mut().unwrap();
        last.resize(expectation.dimension(), 0.0);

        last.par_iter_mut()
            .enumerate()
            .for_each(|(i, value)| {
                let layer = results.layers.last().unwrap();
                let activation_derivative
                    = self.transitions.last().unwrap()
                        .activation_fn
                        .differential();

                *value = (layer.activation[i] - expectation[i])
                    / expectation.len() as f32
                    * activation_derivative(layer.input[i], None)
            });

        // Manual .windows_mut implementation
        for layer in (1..=self.n_layers() - 1).rev() {
            let (prev, cur) = buffer.get_two_mut(layer - 1, layer).unwrap();

            prev.resize(results.layers[layer - 1].activation.dimension(), 0.0);
            cur.resize(results.layers[layer].activation.dimension(), 0.0);

            prev.transform_transposed_from(
                &self.transitions[layer - 1].weights,
                cur,
            );

            prev.par_iter_mut()
                .enumerate()
                .for_each(|(i, value)| {
                    let activation_derivative = self.transitions[layer - 1]
                        .activation_fn
                        .differential();

                    *value *= activation_derivative(
                        results.layers[layer - 1].input[i], None,
                    );
                });
        }

        gradient.transitions.par_iter_mut()
            .enumerate()
            .for_each(|(layer, value)| {
                let n_columns = value.weights.n_columns;

                value.weights.values.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, value)| {
                        let (j, k) = Matrix::get_indices(n_columns, i);
                        *value = results.layers[layer].activation[k]
                            * buffer[layer + 1][j];
                    });

                value.biases.clone_from(&buffer[layer + 1]);
            });
    }

    pub fn sub_gradient(&mut self, gradient: &Self, learning_rate: f32) {
        self.mul_add(gradient, -learning_rate);
    }

    pub fn mul_add(&mut self, other: &Self, mul: f32) {
        use rayon::prelude::*;

        self.transitions.par_iter_mut()
            .zip(other.transitions.par_iter())
            .for_each(move |(this, other)| this.mul_add(other, mul));
    }
}



pub trait Optimizer: Send + Sync + 'static {
    fn apply_gradient(&mut self, network: &mut Network, gradient: &Network);
    fn reset(&mut self);
}
assert_obj_safe!(Optimizer);



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constant {
    pub learning_rate: f32,
}

impl Constant {
    pub const fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for Constant {
    fn apply_gradient(&mut self, network: &mut Network, gradient: &Network) {
        network.sub_gradient(gradient, self.learning_rate);
    }

    fn reset(&mut self) { }
}



#[derive(Clone, Debug, PartialEq, Default)]
#[derive(Serialize, Deserialize)]
pub struct Adam {
    pub first_decay: f32,
    pub second_decay: f32,
    time: i32,
    step_size: f32,
    first_momentum: Network,
    second_momentum: Network,
    eps: f32,
}
assert_impl_all!(Adam: Send, Sync);

impl Adam {
    const FIRST_DECAY_DEFAULT: f32 = 0.9;
    const SECOND_DECAY_DEFAULT: f32 = 0.999;

    pub fn new(learning_rate: f32, layout: &NetworkLayout) -> Self {
        let first_momentum = layout.allocate_gradient();
        let second_momentum = first_momentum.clone();

        Self {
            first_decay: Self::FIRST_DECAY_DEFAULT,
            second_decay: Self::SECOND_DECAY_DEFAULT,
            time: 0,
            eps: 1e-8,
            step_size: learning_rate,
            first_momentum,
            second_momentum,
        }
    }

    pub fn first_decay(mut self, decay: f32) -> Self {
        self.first_decay = decay;
        self
    }

    pub fn second_decay(mut self, decay: f32) -> Self {
        self.second_decay = decay;
        self
    }

    pub fn decays(self, decay1: f32, decay2: f32) -> Self {
        self.first_decay(decay1).second_decay(decay2)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_value(
        time: i32, eps: f32, decay1: f32, decay2: f32, step_size: f32,
        momentum1: &mut f32, momentum2: &mut f32,
        gradient: f32, network: &mut f32,
    ) {
        *momentum1 = decay1 * *momentum1 + (1.0 - decay1) * gradient;
        *momentum2 = decay2 * *momentum2 + (1.0 - decay2) * gradient.powi(2);

        let first_estimate = *momentum1 / (1.0 - decay1.powi(time));
        let second_estimate = *momentum2 / (1.0 - decay2.powi(time));

        *network -= step_size * first_estimate / (second_estimate.sqrt() + eps);
    }
}

impl Optimizer for Adam {
    fn apply_gradient(&mut self, network: &mut Network, gradient: &Network) {
        use rayon::prelude::*;

        self.time += 1;

        self.first_momentum.transitions.par_iter_mut()
            .zip(self.second_momentum.transitions.par_iter_mut())
            .zip(gradient.transitions.par_iter())
            .zip(network.transitions.par_iter_mut())
            .flat_map(|(((first, second), gradient), network)| {
                first.par_values_mut()
                    .zip(second.par_values_mut())
                    .zip(gradient.par_values())
                    .zip(network.par_values_mut())
            })
            .for_each(|(((momentum1, momentum2), gradient), network)| {
                Self::update_value(
                    self.time,
                    self.eps,
                    self.first_decay,
                    self.second_decay,
                    self.step_size,
                    momentum1,
                    momentum2,
                    gradient,
                    network,
                );
            });
    }

    fn reset(&mut self) {
        self.first_momentum.fill(0.0);
        self.second_momentum.fill(0.0);
        self.time = 0;
    }
}



#[derive(Clone, Debug, PartialEq, Default)]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct Dataset {
    pub data: Vec<(Vector, Vector)>,
}
assert_impl_all!(Dataset: Send, Sync);

impl Dataset {
    pub const fn new() -> Self {
        Self { data: vec![] }
    }

    pub fn from_generator(size: usize, gen: impl Fn() -> (Vector, Vector)) -> Self {
        Self { data: (0..size).map(move |_| gen()).collect() }
    }
}

impl From<Vec<(Vector, Vector)>> for Dataset {
    fn from(value: Vec<(Vector, Vector)>) -> Self {
        Self { data: value }
    }
}

impl std::ops::Deref for Dataset {
    type Target = [(Vector, Vector)];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl std::ops::DerefMut for Dataset {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}



#[derive(Clone, Default, PartialEq, Debug, Eq, Copy, Hash, PartialOrd, Ord)]
pub enum TrainLog {
    DrawLoading,
    #[default]
    Silent,
}
assert_impl_all!(TrainLog: Send, Sync);



pub struct Trainer<'d> {
    optimizer: Box<dyn Optimizer>,
    pub batch_size: usize,
    pub n_iterations: usize,
    pub dataset: &'d Dataset,
    gradient: Network,
    grad_buf: Network,
    prop_buf: Vec<Vector>,
    neurons: Neurons,
}
assert_impl_all!(Trainer: Send, Sync);

impl<'d> Trainer<'d> {
    const DEFAULT_BATCH_SIZE: usize = 1000;
    const DEFAULT_N_ITERATIONS: usize = 1000;

    pub fn new(
        optimizer: impl Optimizer, dataset: &'d Dataset, layout: &NetworkLayout,
    ) -> Self {
        Self {
            optimizer: Box::new(optimizer),
            batch_size: Self::DEFAULT_BATCH_SIZE,
            n_iterations: Self::DEFAULT_N_ITERATIONS,
            dataset,
            gradient: layout.allocate_gradient(),
            grad_buf: layout.allocate_gradient(),
            prop_buf: layout.allocate_propagation_buffer(),
            neurons: layout.allocate_output_buffer(),
        }
    }

    pub fn execute(&mut self, network: &mut Network, log: TrainLog) {
        let is_logging_enabled = matches!(log, TrainLog::DrawLoading);

        for _ in kdam::tqdm!(0..self.n_iterations, desc = "Iterating", position = 0) {
            self.optimizer.reset();

            let batches = kdam::tqdm!(
                0..self.dataset.data.len() / self.batch_size,
                desc = "Training",
                position = 1
            );

            for batch_index in batches {
                self.gradient.fill(0.0);

                let start = (self.batch_size * batch_index) % self.dataset.data.len();

                let data = kdam::tqdm!(
                    self.dataset.data[start..].iter().take(self.batch_size),
                    desc = "Calculating gradient",
                    position = 2
                );

                for (input, expectation) in data {
                    network.execute(input, &mut self.neurons);
                    network.propagate_back(
                        &mut self.grad_buf,
                        expectation,
                        &self.neurons,
                        &mut self.prop_buf,
                    );

                    self.gradient.mul_add(
                        &self.grad_buf,
                        1.0 / self.batch_size as f32,
                    );
                }

                self.optimizer.apply_gradient(network, &self.gradient);
            }
        }

        if is_logging_enabled {
            eprint!("\n\n");
        }
    }

    pub fn iterations(mut self, n_iterations: usize) -> Self {
        self.n_iterations = n_iterations;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_builds() {
        use Initializer::*;

        let network = Network::builder()
            .layer(Dense::new(2, 8).initializer(Siren))
            .layer(Sin)
            .layer(Dense::new(8, 8))
            .layer(Sin)
            .layer(Dense::new(8, 1))
            .build();

        eprintln!("{network:#?}");
    }

    #[test]
    fn network_runs() {
        let network = Network::builder()
            .layer(Dense::new(2, 10))
            .layer(Sin)
            .layer(Dense::new(10, 1))
            .build();

        let layout = network.layout();

        let input = Vector::from_iter([0.5, 0.2]);
        let mut result = layout.allocate_output_buffer();

        network.execute(&input, &mut result);

        eprintln!("{result:#?}");
    }

    #[test]
    fn network_propagates_back() {
        let mut network = Network::builder()
            .layer(Dense::new(2, 8))
            .layer(Sin)
            .layer(Dense::new(8, 8))
            .layer(Sin)
            .layer(Dense::new(8, 1))
            .build();

        let layout = network.layout();

        let input = Vector::from_iter([0.4, 0.5]);
        let expectation = Vector::from_iter([0.1]);
        let mut result = layout.allocate_output_buffer();
        let mut prop_buf = layout.allocate_propagation_buffer();

        let mut grad = network.clone();

        network.execute(&input, &mut result);

        eprintln!("MSE before = {}", result.mse(&expectation));

        network.propagate_back(&mut grad, &expectation, &result, &mut prop_buf);
        network.sub_gradient(&grad, 0.1);
        network.execute(&input, &mut result);

        eprintln!("MSE after = {}", result.mse(&expectation));
    }

    #[test]
    fn network_train() {
        let mut network = Network::builder()
            .layer(Dense::new(3, 64))
            .layer(Sin)
            .layer(Dense::new(64, 64))
            .layer(Sin)
            .layer(Dense::new(64, 1))
            .build();

        let layout = network.layout();

        let input = Vector::from_iter([0.4, 0.5, 0.1]);
        let expecation = Vector::from_iter([0.1]);
        let mut result = layout.allocate_output_buffer();
        let mut prop_buf = layout.allocate_propagation_buffer();
        let mut grad = layout.allocate_gradient();

        network.execute(&input, &mut result);

        eprintln!("MSE before = {}", result.mse(&expecation));

        let mut adam = Adam::new(0.01, &layout);

        for _ in 0..1000 {
            network.propagate_back(&mut grad, &expecation, &result, &mut prop_buf);
            adam.apply_gradient(&mut network, &grad);
            network.execute(&input, &mut result);
        }

        eprintln!("MSE after = {}", result.mse(&expecation));
    }

    #[test]
    fn network_ser() {
        let network = Network::builder()
            .layer(Dense::new(2, 8))
            .layer(Sin)
            .layer(Dense::new(8, 10))
            .layer(Relu)
            .layer(Dense::new(10, 1))
            .build();

        let out = serde_json::to_string_pretty(&network).unwrap();

        eprintln!("{out}");
    }
}