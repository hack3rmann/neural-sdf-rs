pub mod util;
pub mod neural;
pub mod math;

use clap::Parser;
use crate::{neural::*, math::*};
use glam::*;



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _cli_args = CliArgs::parse();

    let mut network = Network::builder()
        .layer(Dense::new(2, 64))
        .layer(Sin)
        .layer(Dense::new(64, 64))
        .layer(Sin)
        .layer(Dense::new(64, 64))
        .layer(Sin)
        .layer(Dense::new(64, 1))
        .build();

    let layout = network.layout();

    let dataset = Dataset::from((0..10_000).map(|_| {
        let x = 2.0 * rand::random::<f32>() - 1.0;
        let y = 2.0 * rand::random::<f32>() - 1.0;
        let pos = vec2(x, y);

        let expect = Vec2::length(pos - vec2(0.0, 0.0)) - 0.8;

        (Vector::from_iter(pos.to_array()), Vector::from_iter([expect]))
    }).collect::<Vec<_>>());

    let mut trainer = Trainer::new(
        Adam::new(0.00001, &layout), dataset, &layout
    ).batch_size(100).iterations(1000);

    trainer.execute(&mut network, TrainLog::DrawLoading);

    Ok(())
}



#[derive(clap::Parser, Debug)]
pub struct CliArgs {
    #[arg(long, short)]
    train: bool,
    #[arg(long, short)]
    no_render: bool,
}