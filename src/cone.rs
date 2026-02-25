use crate::bitvec::{intersection_into, BitSet};
use crate::distance_matrix::DistanceMatrix;
use anyhow::{anyhow, Result};

/// A simplicial tower for 1-dimensional complexes (graphs).
/// Implements "Small Coning" logic with Union-Find.
pub struct Tower {
    parent: Vec<usize>, // DSU parent pointers
    active: BitSet,     // Tracks active vertices

    // Adjacency matrix stored as bitsets
    pub neighbors: Vec<BitSet>,
    // Map of canonical edge (u, v) -> birth time
    // pub distances: HashMap<(usize, usize), f32>,
    pub distances: DistanceMatrix,

    // Scratch buffers to avoid allocation during contract
    work: BitSet,
    work_aux: BitSet,
}

impl Tower {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            active: BitSet::new_ones(n),
            neighbors: vec![BitSet::new(); n],
            distances: DistanceMatrix::new(n),
            work: BitSet::new(),
            work_aux: BitSet::new(),
        }
    }

    /// Path compression DSU find
    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    // #[inline]
    // fn canonical(u: usize, v: usize) -> (usize, usize) {
    //     if u < v {
    //         (u, v)
    //     } else {
    //         (v, u)
    //     }
    // }

    pub fn add_edge(&mut self, u_in: usize, v_in: usize, dist: f32) {
        let u = self.find(u_in);
        let v = self.find(v_in);

        // Ignore self-loops or inactive vertices (though find() should return active roots)
        if u == v || !self.active.contains(u) || !self.active.contains(v) {
            return;
        }

        // 1. Store the logical edge for the filtration output
        // Note: We use the *current* representatives u, v.
        // If u or v were results of previous contractions, this effectively
        // adds the edge to the "current" active complex.
        // PRESERVE HISTORY: If this edge was added previously (perhaps with a smaller time),
        // we keep the original time.
        if self.distances.get(u, v) < f32::INFINITY {
            return;
        }
        //add edge
        self.distances.set(u, v, dist);

        // 2. Update adjacency for future contractions
        self.neighbors[u].insert(v);
        self.neighbors[v].insert(u);
    }

    /// Contracts u and v. Automatically selects the winner to minimize complexity
    /// (Union-By-Weight on the Active Closed Star).
    pub fn contract(&mut self, u_in: usize, v_in: usize, dist: f32) -> Result<usize> {
        let u = self.find(u_in);
        let v = self.find(v_in);

        if u == v {
            anyhow!("Attempting to contract error");
        } // Already contracted

        // 1. Calculate Active Closed Star sizes (active degree)
        // We assume the BitSet crate can resize 'work' if needed, or it's pre-sized.

        // Compute u's active neighbors into self.work
        intersection_into(
            self.neighbors[u].as_slice(),
            self.active.as_slice(),
            self.work.as_mut_slice(),
        );
        // Remove v from u's set (don't count the edge being contracted)
        self.work.remove(v);
        let count_u = self.work.count();

        // Compute v's active neighbors into self.work_aux
        intersection_into(
            self.neighbors[v].as_slice(),
            self.active.as_slice(),
            self.work_aux.as_mut_slice(),
        );
        // Remove u from v's set
        self.work_aux.remove(u);
        let count_v = self.work_aux.count();

        // 2. Select Winner (Paper Logic: Merge smaller star into larger star)
        // If |ActSt(u)| <= |ActSt(v)|, u becomes inactive (loser).
        let (winner, loser, neighbors_to_move) = if count_u <= count_v {
            (v, u, &self.work)
        } else {
            (u, v, &self.work_aux)
        };

        // 3. Coning: Add edges from Winner to Loser's neighbors
        for neighbor in neighbors_to_move.iter() {
            // Check if edge already exists to preserve filtration order
            if !self.neighbors[winner].contains(neighbor) {
                self.distances.set(winner, neighbor, dist);

                self.neighbors[winner].insert(neighbor);
                self.neighbors[neighbor].insert(winner);
            }
        }

        // 4. Update state
        self.parent[loser] = winner;
        self.active.remove(loser);

        //5. Return the winner node
        Ok(winner)
    }
}
