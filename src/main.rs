pub mod util;
pub mod neural;
pub mod math;

use clap::Parser;
use crate::{neural::*, math::*};
use glam::*;



fn distance_to_color(distance: f32) -> Vec3 {
    const POSITIVE_COLOR: Vec3 = vec3(0.32, 0.37, 0.88);
    const NEGATIVE_COLOR: Vec3 = vec3(0.97, 0.44, 0.1);
    const ZERO_COLOR: Vec3 = vec3(1.0, 0.96, 0.96);
    const SCALE: f32 = 3.5;

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

fn compact_color(mut color: Vec3) -> u32 {
    color = 255.0 * color.clamp(Vec3::ZERO, Vec3::ONE);

    u32::from_le_bytes([
        color.x as u8,
        color.y as u8,
        color.z as u8,
        255,
    ])
}



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _cli_args = CliArgs::parse();

    let mut network = Network::builder()
        .layer(Dense::new(2, 64))
        .layer(Sin)
        .layer(Dense::new(64, 64))
        .layer(Sin)
        .layer(Dense::new(64, 1))
        .build();

    let layout = network.layout();

    let dataset = Dataset::from((0..100).map(|_| {
        let x = 2.0 * rand::random::<f32>() - 1.0;
        let y = 2.0 * rand::random::<f32>() - 1.0;
        let pos = vec2(x, y);

        let expect = Vec2::length(pos - vec2(0.0, 0.0)) - 0.8;

        (Vector::from_iter(pos.to_array()), Vector::from_iter([expect]))
    }).collect::<Vec<_>>());

    let mut trainer = Trainer::new(
        Adam::new(1e-5, &layout), dataset, &layout,
    ).batch_size(100).iterations(2000);

    trainer.execute(&mut network, TrainLog::DrawLoading);

    const IMAGE_SIZE: usize = 512;

    let mut result = layout.allocate_output_buffer();

    _ = kdam::term::hide_cursor();

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
        .map(|(x, y)| Vector::from_iter([x, y]))
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

    _ = kdam::term::show_cursor();

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



#[derive(clap::Parser, Debug)]
pub struct CliArgs {
    #[arg(long, short)]
    train: bool,
    #[arg(long, short)]
    no_render: bool,
}