#[inline]
pub fn intersection_into(xs: &[u64], ys: &[u64], dest: &mut Vec<u64>) {
    dest.clear(); // Ensure the destination is empty before use.

    let (short, long) = if xs.len() < ys.len() {
        (xs, ys)
    } else {
        (ys, xs)
    };

    if short.is_empty() {
        return;
    }

    let mut last_nonzero_word_idx = 0;
    let mut has_nonzero = false;

    // Use `dest` as scratch space. It will be truncated later.
    // This is faster than repeated `push`.
    dest.resize(short.len(), 0);

    for (i, &x_word) in short.iter().enumerate() {
        if x_word == 0 {
            continue;
        }
        let intersection_word = x_word & long[i];
        dest[i] = intersection_word;

        if intersection_word != 0 {
            last_nonzero_word_idx = i;
            has_nonzero = true;
        }
    }

    // Truncate the vector to the minimal required size.
    if has_nonzero {
        dest.truncate(last_nonzero_word_idx + 1);
    } else {
        // If the intersection is empty, clear the vector.
        dest.clear();
    }
}

/// Computes the set difference (xs - ys) into `dest`.
/// Result contains elements in `xs` that are NOT in `ys`.
#[inline]
pub fn subtract_into(xs: &[u64], ys: &[u64], dest: &mut Vec<u64>) {
    dest.clear();

    if xs.is_empty() {
        return;
    }

    dest.resize(xs.len(), 0);

    let mut last_nonzero_idx = 0;
    let mut has_nonzero = false;

    for (i, &x_word) in xs.iter().enumerate() {
        // If ys is shorter, those bits are implicitly 0, so nothing to subtract
        let y_word = ys.get(i).copied().unwrap_or(0);
        let diff_word = x_word & !y_word;
        dest[i] = diff_word;

        if diff_word != 0 {
            last_nonzero_idx = i;
            has_nonzero = true;
        }
    }

    if has_nonzero {
        dest.truncate(last_nonzero_idx + 1);
    } else {
        dest.clear();
    }
}

const WORD_SIZE_BITS: usize = 64;
/// A memory-efficient, dynamically-sized bit set for representing sets of vertices.
/// The underlying `Vec` only stores words up to the highest set bit.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BitSet {
    pub bits: Vec<u64>,
}

impl BitSet {
    pub fn new() -> Self {
        Self { bits: Vec::new() }
    }
    /// Check if a bit at position `v` is set
    pub fn contains(&self, v: usize) -> bool {
        let word_idx = v / WORD_SIZE_BITS;

        if word_idx >= self.bits.len() {
            return false;
        }

        let bit_idx = v % WORD_SIZE_BITS;
        (self.bits[word_idx] & (1u64 << bit_idx)) != 0
    }
    /// Inserts a vertex `v` into the set, resizing the underlying vector if necessary.
    #[inline]
    pub fn insert(&mut self, v: usize) {
        let word_idx = v / WORD_SIZE_BITS;
        let bit_idx = v % WORD_SIZE_BITS;

        // Dynamically resize if the vertex is outside the current capacity.
        if word_idx >= self.bits.len() {
            // `resize` is efficient and fills with the provided value (0).
            self.bits.resize(word_idx + 1, 0);
        }
        self.bits[word_idx] |= 1u64 << bit_idx;
    }

    /// Removes a vertex `v` from the set.
    ///
    /// Returns `true` if the vertex was present in the set, `false` otherwise.
    pub fn remove(&mut self, v: usize) -> bool {
        let word_idx = v / WORD_SIZE_BITS;

        // If the word index is out of bounds, the vertex cannot be in the set.
        if word_idx >= self.bits.len() {
            return false;
        }

        let bit_idx = v % WORD_SIZE_BITS;

        // Create a mask with a '1' at the position of the bit to check.
        let mask = 1u64 << bit_idx;
        let word = &mut self.bits[word_idx];

        // Check if the bit is currently set.
        if (*word & mask) != 0 {
            // The bit was present. Unset it by ANDing with the inverted mask.
            *word &= !mask;
            true
        } else {
            // The bit was not present.
            false
        }
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.bits
    }

    pub fn as_mut_slice(&mut self) -> &mut Vec<u64> {
        &mut self.bits
    }

    pub fn intersect(&self, other: &BitSet) -> BitSet {
        let min_len = self.bits.len().min(other.bits.len());

        if min_len == 0 {
            return BitSet::new();
        }

        let mut bits = Vec::with_capacity(min_len);
        let mut last_nonzero_idx = 0;
        let mut has_nonzero = false;

        for i in 0..min_len {
            let word = self.bits[i] & other.bits[i];
            bits.push(word);
            if word != 0 {
                last_nonzero_idx = i;
                has_nonzero = true;
            }
        }

        if has_nonzero {
            bits.truncate(last_nonzero_idx + 1);
        } else {
            bits.clear();
        }

        BitSet { bits }
    }

    /// Intersects this BitSet with another slice in-place (self = self ∩ other).
    #[inline]
    pub fn intersect_inplace(&mut self, other: &[u64]) {
        if other.is_empty() {
            self.bits.clear();
            return;
        }

        // Truncate to the shorter length first - bits beyond other's length become 0
        if self.bits.len() > other.len() {
            self.bits.truncate(other.len());
        }

        let mut last_nonzero_idx = 0;
        let mut has_nonzero = false;

        for (i, word) in self.bits.iter_mut().enumerate() {
            *word &= other[i];
            if *word != 0 {
                last_nonzero_idx = i;
                has_nonzero = true;
            }
        }

        if has_nonzero {
            self.bits.truncate(last_nonzero_idx + 1);
        } else {
            self.bits.clear();
        }
    }
    /// Subtracts another slice from this BitSet in-place (self = self - other).
    /// Slightly more optimized single-pass version.
    pub fn subtract_inplace(&mut self, other: &[u64]) {
        if self.bits.is_empty() || other.is_empty() {
            return;
        }

        let overlap_len = self.bits.len().min(other.len());
        let mut last_nonzero_idx = 0;
        let mut has_nonzero = false;

        // Process overlapping region
        for i in 0..overlap_len {
            self.bits[i] &= !other[i];
            if self.bits[i] != 0 {
                last_nonzero_idx = i;
                has_nonzero = true;
            }
        }

        // Check remaining words (unchanged, but may be nonzero)
        for i in overlap_len..self.bits.len() {
            if self.bits[i] != 0 {
                last_nonzero_idx = i;
                has_nonzero = true;
            }
        }

        if has_nonzero {
            self.bits.truncate(last_nonzero_idx + 1);
        } else {
            self.bits.clear();
        }
    }
    /// Create a BitSet with the first `n` bits set to 1
    pub fn new_ones(n: usize) -> Self {
        if n == 0 {
            return Self::new();
        }

        let full_words = n / WORD_SIZE_BITS;
        let remaining_bits = n % WORD_SIZE_BITS;

        let mut bits = vec![u64::MAX; full_words];

        if remaining_bits > 0 {
            // Create a word with only the first `remaining_bits` set
            bits.push((1u64 << remaining_bits) - 1);
        }

        Self { bits }
    }
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
    pub fn count(&self) -> u32 {
        self.bits.iter().map(|&word| word.count_ones()).sum()
    }
    pub fn clear(&mut self) {
        self.bits.clear();
    }
    pub fn iter(&self) -> BitSetIter<'_> {
        BitSetIter {
            // We pass a slice of the BitSet's internal vector.
            // This is a zero-cost operation.
            bits_slice: &self.bits,
            word_index: 0,
            // Safely get the first word, or 0 if the set is empty.
            current_word: self.bits.get(0).copied().unwrap_or(0),
        }
    }
}

/// A non-owning reference to a BitSet's data.
#[derive(Clone, Copy)] // It's just a slice, so it's cheap to copy
pub struct BitSetRef<'a> {
    pub bits: &'a [u64],
}

impl<'a> BitSetRef<'a> {
    pub fn is_empty(&self) -> bool {
        // A BitSet is empty if all its words are zero. The workspace might contain
        // zero-words, so we can't just check self.bits.is_empty().
        self.bits.iter().all(|&word| word == 0)
    }

    pub fn iter(self) -> BitSetIter<'a> {
        BitSetIter {
            // Note: We need to adapt BitSetIter to work with a slice
            // instead of an owned BitSet.
            bits_slice: self.bits,
            word_index: 0,
            current_word: self.bits.get(0).copied().unwrap_or(0),
        }
    }
}

pub struct BitSetIter<'a> {
    bits_slice: &'a [u64],
    word_index: usize,
    current_word: u64,
}

impl<'a> Iterator for BitSetIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let bit_pos = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1;
                return Some((self.word_index * WORD_SIZE_BITS) + bit_pos);
            }

            self.word_index += 1;
            if self.word_index >= self.bits_slice.len() {
                // Changed from self.set.bits.len()
                return None;
            }
            self.current_word = self.bits_slice[self.word_index]; // Changed
        }
    }
}
