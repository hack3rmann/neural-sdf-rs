use rayon::prelude::*;
use serde::{Deserialize, Serialize};



#[derive(Clone, Debug, PartialEq, Default)]
#[derive(Serialize, Deserialize)]
pub struct Matrix {
    pub n_rows: u32,
    pub n_columns: u32,
    pub values: Vec<f32>,
}

impl Matrix {
    pub const fn get_index(n_columns: u32, i: usize, j: usize) -> usize {
        j + n_columns as usize * i
    }

    pub const fn get_indices(n_columns: u32, i: usize) -> (usize, usize) {
        (i / n_columns as usize, i % n_columns as usize)
    }

    pub const fn dimensions(&self) -> (usize, usize) {
        (self.n_rows as usize, self.n_columns as usize)
    }

    pub fn new_zeroed(n_rows: u32, n_columns: u32) -> Self {
        Self { n_rows, n_columns, values: vec![0.0; n_rows as usize * n_columns as usize] }
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.values[Self::get_index(self.n_columns, i, j)]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        self.values[Self::get_index(self.n_columns, i, j)] = value;
    }

    pub fn vector_multiplicator<'s>(&'s self, src: &'s Vector)
        -> impl Fn(usize) -> f32 + Send + Sync + 's
    {
        |i| (0..src.dimension())
            .map(|p| self[(i, p)] * src[p])
            .reduce(std::ops::Add::add)
            .unwrap()
    }

    pub fn vector_transposed_multiplicator<'s>(&'s self, src: &'s Vector)
        -> impl Fn(usize) -> f32 + Send + Sync + 's
    {
        |i| (0..src.dimension())
            .map(|p| self[(p, i)] * src[p])
            .reduce(std::ops::Add::add)
            .unwrap()
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.values[Self::get_index(self.n_columns, i, j)]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.values[Self::get_index(self.n_columns, i, j)]
    }
}



#[repr(transparent)]
#[derive(Clone, Debug, Default, PartialEq)]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vector {
    pub values: Vec<f32>,
}

impl Vector {
    pub fn new_zeroed(size: usize) -> Self {
        Self { values: vec![0.0; size] }
    }

    pub fn map(&mut self, f: impl Fn(usize, f32) -> f32 + Send + Sync + 'static) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, value)| *value = f(i, *value));
    }

    pub fn resize(&mut self, size: usize, value: f32) {
        self.values.resize(size, value);
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    pub fn transform_from(&mut self, matrix: &Matrix, vector: &Self) {
        self.values.resize(matrix.n_rows as usize, 0.0);

        self.values.par_iter_mut()
            .enumerate()
            .for_each(|(i, value)| *value = (0..vector.dimension())
                .map(|p| matrix[(i, p)] * vector[p])
                .reduce(std::ops::Add::add)
                .expect("matrices and vectors can't be empty")
            );
    }

    pub fn transform_transposed_from(&mut self, matrix: &Matrix, vector: &Self) {
        self.values.resize(matrix.n_columns as usize, 0.0);

        self.values.par_iter_mut()
            .enumerate()
            .for_each(|(i, value)| *value = (0..vector.dimension())
                .map(|p| matrix[(p, i)] * vector[p])
                .reduce(std::ops::Add::add)
                .expect("matrices and vectors can't be empty")
            );
    }

    pub fn mse(&self, from: &Self) -> f32 {
        self.par_iter()
            .zip(from.par_iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>() / self.len() as f32
    }
}

impl FromIterator<f32> for Vector {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self { values: Vec::from_iter(iter) }
    }
}

impl From<Vec<f32>> for Vector {
    fn from(value: Vec<f32>) -> Self {
        Self { values: value }
    }
}

impl std::ops::Deref for Vector {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl std::ops::DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl std::ops::AddAssign<&Self> for Vector {
    fn add_assign(&mut self, rhs: &Self) {
        self.values.par_iter_mut()
            .zip(rhs.values.par_iter())
            .for_each(|(this, other)| *this += *other);
    }
}