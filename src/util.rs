


pub fn default<T: Default>() -> T {
    T::default()
}



pub trait SliceExt<T> {
    fn get_two_mut(&mut self, first: usize, second: usize)
        -> Option<(&mut T, &mut T)>;
}

impl<T> SliceExt<T> for [T] {
    fn get_two_mut(&mut self, first: usize, second: usize)
        -> Option<(&mut T, &mut T)>
    {
        if first >= self.len() || second >= self.len() || first == second {
            return None;
        }

        let ptr = self.as_mut_ptr();

        unsafe {
            Some((
                ptr.add(first).as_mut().unwrap_unchecked(),
                ptr.add(second).as_mut().unwrap_unchecked(),
            ))
        }
    }
}