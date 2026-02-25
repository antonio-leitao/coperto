/// Flat triangular matrix (f32) - stores n*(n-1)/2 entries
pub struct DistanceMatrix {
    pub data: Vec<f32>,
    pub n: usize,
}

impl DistanceMatrix {
    pub fn new(n: usize) -> Self {
        let total_size = n * (n - 1) / 2;
        Self {
            data: vec![f32::INFINITY; total_size],
            n,
        }
    }

    #[inline(always)]
    fn index(i: usize, j: usize) -> usize {
        // Assumes i < j
        j * (j - 1) / 2 + i
    }

    #[inline(always)]
    pub fn get(&self, u: usize, v: usize) -> f32 {
        if u == v {
            return 0.0;
        }
        let (i, j) = if u < v { (u, v) } else { (v, u) };
        unsafe { *self.data.get_unchecked(Self::index(i, j)) }
    }

    #[inline(always)]
    pub fn set(&mut self, u: usize, v: usize, val: f32) {
        if u == v {
            return;
        }
        let (i, j) = if u < v { (u, v) } else { (v, u) };
        unsafe {
            *self.data.get_unchecked_mut(Self::index(i, j)) = val;
        }
    }
}
