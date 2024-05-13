mod math;
mod neural;
mod util;
pub mod arch;

use anyhow::Result as AnyResult;
use arch::Arch;
use math::Vector;
use neural::NetworkBuilder;
use std::{fs::File, io::Read, time::Instant};
use glam::Vec3;
use clap::Parser;



#[derive(Clone, Debug, PartialEq, Default)]
pub struct NetworkTest {
    pub points: Vec<Vec3>,
    pub data: Vec<f32>,
}

impl NetworkTest {
    pub fn from_bytes(mut bytes: impl Read) -> Self {
        let mut count = 0_u32;
        bytes.read_exact(bytemuck::bytes_of_mut(&mut count)).unwrap();

        let mut points = vec![Vec3::ZERO; count as usize];
        bytes.read_exact(bytemuck::cast_slice_mut(&mut points)).unwrap();

        let mut data = vec![0.0; count as usize];
        bytes.read_exact(bytemuck::cast_slice_mut(&mut data)).unwrap();

        Self { points, data }
    }

    pub fn get(&self) -> impl ExactSizeIterator<Item = (Vec3, f32)> + '_ {
        Iterator::zip(self.points.iter().copied(), self.data.iter().copied())
    }
}



#[tokio::main]
async fn main() -> AnyResult<()> {
    let args = Args::parse();

    let arch = tokio::fs::read_to_string(&args.arch)
        .await?
        .parse::<Arch>()?;

    let mut network = NetworkBuilder::from(arch).build();
    let layout = network.layout();
    let mut results = layout.allocate_output_buffer();

    network.fill_from_bytes(&mut File::open(&args.weights)?);

    let tests = NetworkTest::from_bytes(&mut File::open(&args.test)?);

    let mut n_fails: usize = 0;

    let time = Instant::now();

    for (input, expected) in kdam::tqdm!(tests.get(), desc = "Passing tests") {
        network.execute(&Vector::from_iter(input.to_array()), &mut results);
        
        if results.mse(&Vector::from_iter([expected])) >= 1e-5 {
            eprintln!("failed to execute network on input {input}: \
                       expected [{expected}], got: {:?}",
                       results.layers.last().unwrap().activation.values);

            n_fails += 1;
        }

        if n_fails == args.number_of_fails {
            panic!("Too many errors, aborting.");
        }
    }

    println!();
    
    eprintln!("All tests passed!");
    
    let time = time.elapsed();

    if args.bench {
        println!("Execution time: {time:?}");
    }

    Ok(())
}




#[derive(Parser, Debug)]
#[command(version, about = "Neural network executor", long_about = None)]
struct Args {
    /// Path to network architecture file.
    #[arg(short, long)]
    pub arch: String,

    /// Path to tests file.
    #[arg(short, long)]
    pub test: String,

    /// Path to network parameters file.
    #[arg(short, long)]
    pub weights: String,

    /// Number of first fails to skip.
    #[arg(short, long, default_value_t = 5)]
    pub number_of_fails: usize,

    /// Do benchmarking.
    #[arg(short, long)]
    pub bench: bool,
}