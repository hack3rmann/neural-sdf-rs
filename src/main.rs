mod math;
mod neural;
mod util;
pub mod arch;

use anyhow::Result as AnyResult;
use arch::Arch;
use math::Vector;
use neural::NetworkBuilder;
use std::{fs::File, io::Read};
use glam::Vec3;



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
    let arch = tokio::fs::read_to_string("assets/sdf2_arch.txt")
        .await?
        .parse::<Arch>()?;

    let mut network = NetworkBuilder::from(arch).build();
    let layout = network.layout();
    let mut results = layout.allocate_output_buffer();

    network.fill_from_bytes(
        &mut File::open("assets/sdf2_weights.bin")?,
    );

    let tests = NetworkTest::from_bytes(
        &mut File::open("assets/sdf2_test.bin")?,
    );

    for (input, expected) in kdam::tqdm!(tests.get()) {
        network.execute(&Vector::from_iter(input.to_array()), &mut results);
        assert!(results.mse(&Vector::from_iter([expected])) < 1e-5);
    }

    Ok(())
}