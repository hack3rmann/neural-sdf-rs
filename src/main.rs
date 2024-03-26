pub mod util;
pub mod neural;
pub mod math;

use serde::{Deserialize, Serialize};
use crate::{math::*, neural::*};
use glam::*;



#[allow(dead_code)]
fn distance_to_color(distance: f32) -> Vec3 {
    const POSITIVE_COLOR: Vec3 = vec3(0.32, 0.37, 0.88);
    const NEGATIVE_COLOR: Vec3 = vec3(0.97, 0.44, 0.1);
    const ZERO_COLOR: Vec3 = vec3(1.0, 0.96, 0.96);
    const SCALE: f32 = 10.0;

    let activation = |x: f32| -> f32 {
        let exp = f32::exp(-x);
        (1.0 - exp) / (1.0 + exp)
    };

    let param = activation(SCALE * distance);

    if param < 0.0 {
        ZERO_COLOR.lerp(NEGATIVE_COLOR, param.abs())
    } else {
        ZERO_COLOR.lerp(POSITIVE_COLOR, param)
    }
}

#[allow(dead_code)]
fn distance_to_outline(distance: f32) -> Vec3 {
    const WIDTH: f32 = 0.05;

    if distance.abs() < WIDTH {
        Vec3::ONE.lerp(Vec3::ZERO, distance / WIDTH)
    } else {
        Vec3::ZERO
    }
}

fn compact_color(mut color: Vec3) -> u32 {
    color = 255.0 * color.clamp(Vec3::ZERO, Vec3::ONE);

    u32::from_le_bytes([
        color.x as u8,
        color.y as u8,
        color.z as u8,
        255,
    ])
}


fn calculate_mse(network: &Network, layout: &NetworkLayout, dataset: &Dataset) -> f32 {
    use rayon::prelude::*;

    const MSE_DATA_COUNT: usize = u16::MAX as usize;

    let result = kdam::par_tqdm!(dataset.data.par_iter().take(MSE_DATA_COUNT), desc = "Calculating MSE")
        .map(|(input, expectation)| {
            let mut results = layout.allocate_output_buffer();
            network.execute(input, &mut results);
            results.mse(expectation) / MSE_DATA_COUNT as f32
        })
        .sum::<f32>();

    eprintln!();

    result
}


async fn train_example(dataset: &Dataset) -> anyhow::Result<Network> {
    let mut network = Network::builder()
        .layer(Dense::new(3, 64))
        .layer(Sin)
        .layer(Dense::new(64, 64))
        .layer(Sin)
        .layer(Dense::new(64, 64))
        .layer(Sin)
        .layer(Dense::new(64, 1))
        .build();

    let layout = network.layout();

    eprintln!("MSE before = {}", calculate_mse(&network, &layout, dataset));

    let mut trainer = Trainer::new(
        Adam::new(1e-5, &layout), dataset, &layout,
    ).batch_size(512).iterations(2);

    trainer.execute(&mut network, TrainLog::DrawLoading);

    eprintln!("MSE after = {}", calculate_mse(&network, &layout, dataset));

    Ok(network)
}

async fn generate_dataset() -> anyhow::Result<Dataset> {
    use wavefront_obj::obj;
    use rayon::prelude::*;
    use rand::seq::SliceRandom;

    let obj_src = tokio::fs::read_to_string("assets/cup_1n.obj").await?;

    let object = obj::parse(obj_src)?;

    let vertices = object.objects[0].vertices.as_slice();

    const IMAGE_SIZE: usize = 64; 

    let mut dataset = Vec::with_capacity(IMAGE_SIZE.pow(3));

    kdam::par_tqdm!((0..IMAGE_SIZE.pow(3)).into_par_iter(), desc = "Generating dataset")
        .map(|i| ((i / IMAGE_SIZE) / IMAGE_SIZE, (i / IMAGE_SIZE) % IMAGE_SIZE, i % IMAGE_SIZE))
        .map(|(x, y, z)| (
            x as f32 / (IMAGE_SIZE - 1) as f32,
            y as f32 / (IMAGE_SIZE - 1) as f32,
            z as f32 / (IMAGE_SIZE - 1) as f32,
        ))
        .map(|(x, y, z)| vec3(2.0 * x - 1.0, 1.0 - 2.0 * y, 2.0 * z - 1.0))
        .map(|pos| (pos, vertices.par_iter()
            .map(|&vertex| {
                let vertex = vec3(vertex.x as f32, vertex.y as f32, vertex.z as f32);
                Vec3::length(pos - vertex)
            })
            .reduce(|| f32::INFINITY, f32::min))
        )
        // .map(|pos| (pos, Vec3::length(pos - Vec3::ZERO) - 0.8))
        .map(|(pos, dist)| (
            Vector::from_iter(pos.to_array()),
            Vector::from_iter([dist]),
        ))
        .collect_into_vec(&mut dataset);

    eprintln!();

    eprintln!("Shuffling dataset...");

    dataset.shuffle(&mut rand::thread_rng());

    Ok(Dataset { data: dataset })
}

async fn draw(network: &Network, layout: &NetworkLayout) -> anyhow::Result<()> {
    const IMAGE_SIZE: usize = 256;

    let mut result = layout.allocate_output_buffer();

    let image = kdam::tqdm!(0..IMAGE_SIZE.pow(2), desc = "Drawing image")
        .map(|i| (i % IMAGE_SIZE, i / IMAGE_SIZE))
        .map(|(x, y)| (
            x as f32 / (IMAGE_SIZE - 1) as f32,
            y as f32 / (IMAGE_SIZE - 1) as f32,
        ))
        .map(|(x, y)| (2.0 * x - 1.0, 1.0 - 2.0 * y))
        // .map(|(x, y)| {
        //     let pos = vec2(x, y);

        //     Vec2::length(pos - vec2(0.0, 0.0)) - 0.8
        // })
        .map(|(x, y)| Vector::from_iter([0.0, x, y]))
        .map(|input| {
            network.execute(&input, &mut result);
            let &[distance] = result.layers.last().unwrap()
                .activation.values.as_slice()
            else { unreachable!() };

            distance
        })
        .map(distance_to_color)
        .map(compact_color)
        .collect::<Vec<_>>();

    eprintln!();

    let file = std::fs::File::create("target/result.png")?;

    let buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        buf_writer, IMAGE_SIZE as u32, IMAGE_SIZE as u32,
    );

    encoder.set_color(png::ColorType::Rgba);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&image))?;

    Ok(())
}



#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct TestNetwork {
    #[serde(rename = "fc1.weight")]
    fc1_weight: Vec<Vec<f64>>,
    #[serde(rename = "fc1.bias")]
    fc1_bias: Vec<f64>,
    #[serde(rename = "fc2.weight")]
    fc2_weight: Vec<Vec<f64>>,
    #[serde(rename = "fc2.bias")]
    fc2_bias: Vec<f64>,
    #[serde(rename = "fc3.weight")]
    fc3_weight: Vec<Vec<f64>>,
    #[serde(rename = "fc3.bias")]
    fc3_bias: Vec<f64>,
    #[serde(rename = "out.weight")]
    out_weight: Vec<Vec<f64>>,
    #[serde(rename = "out.bias")]
    out_bias: Vec<f64>,
}

impl From<TestNetwork> for Network {
    fn from(value: TestNetwork) -> Self {
        let make_dense = |weight: Vec<Vec<f64>>, bias: Vec<f64>| -> DenseLayerTransition {
            DenseLayerTransition {
                weights: Matrix {
                    n_rows: weight.len() as u32,
                    n_columns: weight[0].len() as u32,
                    values: weight.into_iter()
                        .flat_map(IntoIterator::into_iter)
                        .map(|value| value as f32)
                        .collect(),
                },
                biases: bias.into_iter().map(|value| value as f32).collect(),
            }
        };

        Self {
            transitions: vec![
                LayerTransition {
                    activation_fn: ActivationFunction::Sin,
                    dense: make_dense(value.fc1_weight, value.fc1_bias),
                },
                LayerTransition {
                    activation_fn: ActivationFunction::Sin,
                    dense: make_dense(value.fc2_weight, value.fc2_bias),
                },
                LayerTransition {
                    activation_fn: ActivationFunction::Sin,
                    dense: make_dense(value.fc3_weight, value.fc3_bias),
                },
                LayerTransition {
                    activation_fn: ActivationFunction::Id,
                    dense: make_dense(value.out_weight, value.out_bias),
                },
            ],
        }
    }
}



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use std::path::Path;
    use clap::Parser;
    
    const DATASET: &str = "assets/ball_dataset.bin";
    
    env_logger::init();

    let dataset = if !Path::new(DATASET).exists() {
        let dataset = generate_dataset().await?;

        tokio::fs::write(
            DATASET,
            bincode::serialize(&dataset)?,
        ).await?;

        dataset
    } else {
        eprintln!("Reading dataset from '{DATASET}'...");

        bincode::deserialize(
            &tokio::fs::read(DATASET).await?
        )?
    };

    let network = if CliArgs::parse().train {
        train_example(&dataset).await?
    } else {
        serde_json::from_str(
            &tokio::fs::read_to_string("target/network_parameters.json").await?
        )?
    };

    // let network = serde_json::from_str::<TestNetwork>(
        // &tokio::fs::read_to_string("assets/trained_network.json").await?,
    // )?;

    // let network = Network::from(network);
    let layout = network.layout();

    draw(&network, &layout).await?;

    let mse = calculate_mse(&network, &layout, &dataset);

    eprintln!("MSE = {mse}");

    tokio::fs::write(
        "target/network_parameters.json",
        serde_json::to_string(&network)?,
    ).await?;

    Ok(())
}



#[derive(clap::Parser, Debug)]
pub struct CliArgs {
    #[arg(long, short)]
    train: bool,
    #[arg(long, short)]
    no_render: bool,
}