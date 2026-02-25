use std::collections::HashMap;

pub struct CompleteLinkage {
    parent: Vec<usize>,
    size: Vec<usize>,
    /// Maps cluster_root -> (neighbor_root -> edge_count)
    edges: Vec<HashMap<usize, usize>>,
}

impl CompleteLinkage {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            edges: vec![HashMap::new(); n],
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            return i;
        }
        let p = self.parent[i];
        self.parent[i] = self.find(p);
        self.parent[i]
    }

    pub fn register_edge(&mut self, u: usize, v: usize) -> Option<(usize, usize)> {
        let root_u = self.find(u);
        let root_v = self.find(v);

        if root_u == root_v {
            return None;
        }

        // 1. Increment U -> V and capture the new count
        // The block ensures the mutable borrow of `self` ends immediately after.
        let count = {
            let entry = self.edges[root_u].entry(root_v).or_insert(0);
            *entry += 1;
            *entry
        };

        // 2. Increment V -> U
        *self.edges[root_v].entry(root_u).or_insert(0) += 1;

        // 3. Check condition
        if count == self.size[root_u] * self.size[root_v] {
            Some((root_u, root_v))
        } else {
            None
        }
    }

    pub fn merge(&mut self, root_a: usize, root_b: usize, winner: usize) {
        let root_a = self.find(root_a);
        let root_b = self.find(root_b);
        if root_a == root_b {
            return;
        }

        let loser = if winner == root_a { root_b } else { root_a };

        // 1. Update DSU and Size
        self.parent[loser] = winner;
        self.size[winner] += self.size[loser];

        // 2. Transfer edges
        // We collect to a Vec to stop borrowing `self.edges[loser]`,
        // allowing us to mutate `self.edges[winner]` in the loop.
        let loser_edges: Vec<(usize, usize)> = self.edges[loser].drain().collect();

        for (neighbor, count) in loser_edges {
            if neighbor == winner {
                continue;
            }

            // Add counts to winner
            *self.edges[winner].entry(neighbor).or_insert(0) += count;

            // Redirect neighbor to point to winner
            if let Some(n_map) = self.edges.get_mut(neighbor) {
                n_map.remove(&loser);
                *n_map.entry(winner).or_insert(0) += count;
            }
        }

        // 3. Cleanup
        self.edges[winner].remove(&loser);
    }
}
