use std::collections::{HashMap, HashSet, VecDeque};

/// Conservative Complete Linkage.
///
/// The struct itself is identical to standard `CompleteLinkage`: a union-find
/// with per-cluster edge counts. The conservative behavior lives entirely in
/// the caller's main loop, which uses two different code paths:
///
/// - **Fast path** (no ties): `register_edge` → `merge`. Identical to standard
///   complete linkage. Zero extra overhead.
///
/// - **Tie path**: `register_edge` buffers saturated pairs externally.
///   At the end of the tie batch, [`flush_ties`] resolves them with clique checks.
///
/// The struct provides both `merge` (standard, no tracking) and `merge_tracking`
/// (returns new saturations from edge aggregation) so the fast path pays nothing.
pub struct ConservativeCompleteLinkage {
    parent: Vec<usize>,
    size: Vec<usize>,
    /// Maps cluster_root -> (neighbor_root -> edge_count)
    edges: Vec<HashMap<usize, usize>>,
}

impl ConservativeCompleteLinkage {
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

    /// Register an inter-cluster edge. Returns the two roots if the pair
    /// became saturated (complete linkage condition met). O(1).
    ///
    /// Identical to standard `CompleteLinkage::register_edge`.
    pub fn register_edge(&mut self, u: usize, v: usize) -> Option<(usize, usize)> {
        let root_u = self.find(u);
        let root_v = self.find(v);
        if root_u == root_v {
            return None;
        }

        let count = {
            let entry = self.edges[root_u].entry(root_v).or_insert(0);
            *entry += 1;
            *entry
        };

        *self.edges[root_v].entry(root_u).or_insert(0) += 1;

        if count == self.size[root_u] * self.size[root_v] {
            Some((root_u, root_v))
        } else {
            None
        }
    }

    /// Merge two clusters. No saturation tracking.
    /// Used on the fast path (no ties). Identical to standard complete linkage.
    pub fn merge(&mut self, root_a: usize, root_b: usize, winner: usize) {
        let root_a = self.find(root_a);
        let root_b = self.find(root_b);
        if root_a == root_b {
            return;
        }

        let loser = if winner == root_a { root_b } else { root_a };

        self.parent[loser] = winner;
        self.size[winner] += self.size[loser];

        let loser_edges: Vec<(usize, usize)> = self.edges[loser].drain().collect();

        for (neighbor, count) in loser_edges {
            if neighbor == winner {
                continue;
            }
            *self.edges[winner].entry(neighbor).or_insert(0) += count;
            if let Some(n_map) = self.edges.get_mut(neighbor) {
                n_map.remove(&loser);
                *n_map.entry(winner).or_insert(0) += count;
            }
        }

        self.edges[winner].remove(&loser);
    }

    /// Merge two clusters, pushing any newly saturated (winner, neighbor)
    /// pairs into `new_saturations`. Used on the tie path only.
    pub fn merge_tracking(
        &mut self,
        root_a: usize,
        root_b: usize,
        winner: usize,
        new_saturations: &mut Vec<(usize, usize)>,
    ) {
        let root_a = self.find(root_a);
        let root_b = self.find(root_b);
        if root_a == root_b {
            return;
        }

        let loser = if winner == root_a { root_b } else { root_a };

        self.parent[loser] = winner;
        self.size[winner] += self.size[loser];

        let loser_edges: Vec<(usize, usize)> = self.edges[loser].drain().collect();

        for (neighbor, count) in loser_edges {
            if neighbor == winner {
                continue;
            }

            let new_count = {
                let entry = self.edges[winner].entry(neighbor).or_insert(0);
                *entry += count;
                *entry
            };

            if let Some(n_map) = self.edges.get_mut(neighbor) {
                n_map.remove(&loser);
                *n_map.entry(winner).or_insert(0) += count;
            }

            if new_count == self.size[winner] * self.size[neighbor] {
                new_saturations.push((winner, neighbor));
            }
        }

        self.edges[winner].remove(&loser);
    }

    pub fn cluster_size(&mut self, i: usize) -> usize {
        let root = self.find(i);
        self.size[root]
    }
}

/// Resolve a batch of saturated pairs using conservative clique checks.
///
/// Takes ownership of `deferred`, merges clique components, and repeats
/// until no more clique components exist. Returns the merge history.
///
/// `choose_winner` is called with `(root_a, root_b)` and must return the winner.
pub fn flush_ties<F>(
    cl: &mut ConservativeCompleteLinkage,
    deferred: &mut Vec<(usize, usize)>,
    mut choose_winner: F,
) -> Vec<(usize, usize)>
where
    F: FnMut(usize, usize) -> usize,
{
    let mut merges = Vec::new();
    let mut new_saturations = Vec::new();

    loop {
        // Resolve roots, deduplicate, validate
        let mut seen = HashSet::new();
        let mut valid = Vec::new();

        for &(a, b) in deferred.iter() {
            let ra = cl.find(a);
            let rb = cl.find(b);
            if ra == rb {
                continue;
            }
            let pair = if ra < rb { (ra, rb) } else { (rb, ra) };
            if !seen.insert(pair) {
                continue;
            }
            let count = cl.edges[pair.0].get(&pair.1).copied().unwrap_or(0);
            if count == cl.size[pair.0] * cl.size[pair.1] {
                valid.push(pair);
            }
        }

        deferred.clear();

        if valid.is_empty() {
            break;
        }

        // Single pair: trivially a 2-clique
        if valid.len() == 1 {
            let (a, b) = valid[0];
            let winner = choose_winner(a, b);
            cl.merge_tracking(a, b, winner, &mut new_saturations);
            merges.push((a, b));
            if !new_saturations.is_empty() {
                deferred.append(&mut new_saturations);
                continue;
            }
            break;
        }

        // Build adjacency from valid saturated pairs
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &(a, b) in &valid {
            adj.entry(a).or_default().insert(b);
            adj.entry(b).or_default().insert(a);
        }

        // BFS connected components
        let mut visited = HashSet::new();
        let mut merged_any = false;
        let vertices: Vec<usize> = adj.keys().copied().collect();

        for start in vertices {
            if !visited.insert(start) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                component.push(node);
                if let Some(neighbors) = adj.get(&node) {
                    for &nbr in neighbors {
                        if visited.insert(nbr) {
                            queue.push_back(nbr);
                        }
                    }
                }
            }

            if component.len() < 2 {
                continue;
            }

            // Clique check
            let m = component.len();
            let expected = m * (m - 1) / 2;
            let mut actual = 0;
            for i in 0..m {
                if let Some(nbrs) = adj.get(&component[i]) {
                    for j in (i + 1)..m {
                        if nbrs.contains(&component[j]) {
                            actual += 1;
                        }
                    }
                }
            }

            if actual == expected {
                // Merge clique component
                let mut rep = component[0];
                for &other in &component[1..] {
                    let winner = choose_winner(rep, other);
                    cl.merge_tracking(rep, other, winner, &mut new_saturations);
                    merges.push((rep, other));
                    rep = winner;
                }
                merged_any = true;
            } else {
                // Retain non-clique saturated pairs
                for i in 0..m {
                    if let Some(nbrs) = adj.get(&component[i]) {
                        for j in (i + 1)..m {
                            if nbrs.contains(&component[j]) {
                                deferred.push((component[i], component[j]));
                            }
                        }
                    }
                }
            }
        }

        // Add merge-induced saturations
        deferred.append(&mut new_saturations);

        if !merged_any && deferred.is_empty() {
            break;
        }
        if !merged_any {
            // deferred has only non-clique leftovers, no progress possible
            break;
        }
    }

    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Helper: run conservative complete linkage on sorted edges using
    // the same two-path structure the real main loop would use.
    // Returns merge history as Vec<(winner, loser, distance)>.
    // ---------------------------------------------------------------
    fn run_conservative(
        n: usize,
        sorted_edges: &[(usize, usize, u64)],
    ) -> Vec<(usize, usize, u64)> {
        let mut cl = ConservativeCompleteLinkage::new(n);
        let mut history = Vec::new();
        let mut deferred: Vec<(usize, usize)> = Vec::new();

        let len = sorted_edges.len();
        for i in 0..len {
            let (u, v, d) = sorted_edges[i];
            let next_is_same = i + 1 < len && sorted_edges[i + 1].2 == d;
            let in_tie_batch = next_is_same || !deferred.is_empty();

            if !in_tie_batch {
                // ── FAST PATH: identical to standard complete linkage ──
                if let Some((ra, rb)) = cl.register_edge(u, v) {
                    let winner = ra.min(rb);
                    cl.merge(ra, rb, winner);
                    history.push((winner, ra.max(rb), d));
                }
            } else {
                // ── TIE PATH ──
                if let Some(pair) = cl.register_edge(u, v) {
                    deferred.push(pair);
                }
                if !next_is_same {
                    let merges = flush_ties(&mut cl, &mut deferred, |a, b| a.min(b));
                    for (a, b) in merges {
                        let ra = cl.find(a);
                        let rb = cl.find(b);
                        // they're already merged, just record the event
                        history.push((ra.min(rb), ra.max(rb), d));
                    }
                }
            }
        }

        history
    }

    // Standard (greedy) complete linkage for comparison.
    fn run_standard(n: usize, sorted_edges: &[(usize, usize, u64)]) -> Vec<(usize, usize, u64)> {
        use crate::linkage::complete::CompleteLinkage;
        let mut cl = CompleteLinkage::new(n);
        let mut history = Vec::new();

        for &(u, v, d) in sorted_edges {
            if let Some((ra, rb)) = cl.register_edge(u, v) {
                let winner = ra.min(rb);
                cl.merge(ra, rb, winner);
                history.push((winner, ra.max(rb), d));
            }
        }

        history
    }

    // ---------------------------------------------------------------
    // Test 1: No ties — identical to standard
    // ---------------------------------------------------------------
    #[test]
    fn no_ties_matches_standard() {
        let edges = vec![(0, 1, 1), (1, 2, 2), (0, 2, 3)];
        assert_eq!(run_conservative(3, &edges), run_standard(3, &edges));
    }

    // ---------------------------------------------------------------
    // Test 2: No ties — 4 points
    // ---------------------------------------------------------------
    #[test]
    fn no_ties_four_points() {
        let edges = vec![
            (0, 1, 1),
            (2, 3, 2),
            (0, 2, 3),
            (1, 2, 4),
            (0, 3, 5),
            (1, 3, 6),
        ];
        assert_eq!(run_conservative(4, &edges), run_standard(4, &edges));
    }

    // ---------------------------------------------------------------
    // Test 3: Paper line example — postpones to d=2
    // ---------------------------------------------------------------
    #[test]
    fn paper_line_example_postpones_merge() {
        let edges = vec![(0, 1, 1), (1, 2, 1), (0, 2, 2)];
        let history = run_conservative(3, &edges);
        assert!(history.iter().all(|&(_, _, d)| d == 2));
        assert_eq!(history.len(), 2);
    }

    // ---------------------------------------------------------------
    // Test 4: Diameter bound at scale 1
    // ---------------------------------------------------------------
    #[test]
    fn diameter_bound_at_scale_1() {
        let mut cl = ConservativeCompleteLinkage::new(3);
        cl.register_edge(0, 1);
        cl.register_edge(1, 2);

        // Both saturated, but the component {0,1,2} is not a clique
        let mut deferred = vec![(0, 1), (1, 2)];
        let merges = flush_ties(&mut cl, &mut deferred, |a, b| a.min(b));
        assert!(merges.is_empty());
        assert_eq!(cl.cluster_size(0), 1);
        assert_eq!(cl.cluster_size(1), 1);
        assert_eq!(cl.cluster_size(2), 1);
    }

    // ---------------------------------------------------------------
    // Test 5: Order independence
    // ---------------------------------------------------------------
    #[test]
    fn order_independence_within_ties() {
        let order_a = vec![
            (0, 1, 1),
            (2, 3, 1),
            (0, 2, 1),
            (1, 3, 1),
            (0, 3, 2),
            (1, 2, 2),
        ];
        let order_b = vec![
            (1, 3, 1),
            (0, 2, 1),
            (2, 3, 1),
            (0, 1, 1),
            (1, 2, 2),
            (0, 3, 2),
        ];

        let h_a = run_conservative(4, &order_a);
        let h_b = run_conservative(4, &order_b);

        let mut dists_a: Vec<u64> = h_a.iter().map(|h| h.2).collect();
        let mut dists_b: Vec<u64> = h_b.iter().map(|h| h.2).collect();
        dists_a.sort();
        dists_b.sort();
        assert_eq!(dists_a, dists_b);
    }

    // ---------------------------------------------------------------
    // Test 6: Standard is order-dependent, conservative is not
    // ---------------------------------------------------------------
    #[test]
    fn standard_is_order_dependent_conservative_is_not() {
        let order_ab = vec![(0, 1, 1), (1, 2, 1), (0, 2, 2)];
        let order_bc = vec![(1, 2, 1), (0, 1, 1), (0, 2, 2)];

        assert_eq!(run_standard(3, &order_ab)[0].2, 1);
        assert_eq!(run_standard(3, &order_bc)[0].2, 1);

        let d1: Vec<u64> = run_conservative(3, &order_ab).iter().map(|h| h.2).collect();
        let d2: Vec<u64> = run_conservative(3, &order_bc).iter().map(|h| h.2).collect();
        assert_eq!(d1, vec![2, 2]);
        assert_eq!(d2, vec![2, 2]);
    }

    // ---------------------------------------------------------------
    // Test 7: Complete graph — all merge in one batch
    // ---------------------------------------------------------------
    #[test]
    fn cascading_merges() {
        let edges = vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
        ];
        let history = run_conservative(4, &edges);
        assert_eq!(history.len(), 3);
        assert!(history.iter().all(|&(_, _, d)| d == 1));
    }

    // ---------------------------------------------------------------
    // Test 8: Partial clique
    // ---------------------------------------------------------------
    #[test]
    fn partial_clique_merge() {
        let edges = vec![
            (0, 1, 1),
            (2, 3, 1),
            (3, 4, 1),
            (2, 4, 2),
            (0, 2, 3),
            (0, 3, 3),
            (0, 4, 3),
            (1, 2, 3),
            (1, 3, 3),
            (1, 4, 3),
        ];
        let history = run_conservative(5, &edges);
        let dists: Vec<u64> = history.iter().map(|h| h.2).collect();

        assert_eq!(dists.iter().filter(|&&d| d == 1).count(), 1);
        assert_eq!(dists.iter().filter(|&&d| d == 2).count(), 2);
        assert_eq!(dists.iter().filter(|&&d| d == 3).count(), 1);
    }

    // ---------------------------------------------------------------
    // Test 9 & 10: Edge cases
    // ---------------------------------------------------------------
    #[test]
    fn single_point() {
        assert!(run_conservative(1, &[]).is_empty());
    }

    #[test]
    fn two_points() {
        let h = run_conservative(2, &[(0, 1, 5)]);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].2, 5);
    }

    // ---------------------------------------------------------------
    // Test 11: Non-clique component blocks all merges
    // ---------------------------------------------------------------
    #[test]
    fn non_clique_component_blocks_all_merges() {
        let edges = vec![
            (0, 1, 1),
            (0, 2, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
            (0, 3, 2),
        ];
        let history = run_conservative(4, &edges);
        let dists: Vec<u64> = history.iter().map(|h| h.2).collect();

        assert_eq!(dists.iter().filter(|&&d| d == 1).count(), 0);
        assert_eq!(dists.iter().filter(|&&d| d == 2).count(), 3);
    }

    // ---------------------------------------------------------------
    // Test 12: Two independent pairs then cross-connect
    // ---------------------------------------------------------------
    #[test]
    fn repeat_loop_cascading() {
        let edges = vec![
            (0, 1, 1),
            (2, 3, 1),
            (0, 2, 2),
            (0, 3, 2),
            (1, 2, 2),
            (1, 3, 2),
        ];
        let history = run_conservative(4, &edges);
        let dists: Vec<u64> = history.iter().map(|h| h.2).collect();

        assert_eq!(dists.iter().filter(|&&d| d == 1).count(), 2);
        assert_eq!(dists.iter().filter(|&&d| d == 2).count(), 1);
    }

    // ---------------------------------------------------------------
    // Test 13: Merge-induced saturation across batches
    // ---------------------------------------------------------------
    #[test]
    fn merge_induced_saturation() {
        let edges = vec![(0, 2, 1), (1, 2, 2), (0, 1, 2)];
        let history = run_conservative(3, &edges);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].2, 1);
        assert_eq!(history[1].2, 2);
    }

    // ---------------------------------------------------------------
    // Test 14: Larger no-ties — 6 points, all distinct
    // ---------------------------------------------------------------
    #[test]
    fn no_ties_six_points() {
        let edges = vec![
            (0, 1, 1),
            (0, 2, 2),
            (1, 2, 3),
            (0, 3, 4),
            (1, 3, 5),
            (2, 3, 6),
            (0, 4, 7),
            (1, 4, 8),
            (2, 4, 9),
            (3, 4, 10),
            (0, 5, 11),
            (1, 5, 12),
            (2, 5, 13),
            (3, 5, 14),
            (4, 5, 15),
        ];
        assert_eq!(run_conservative(6, &edges), run_standard(6, &edges));
    }

    // ---------------------------------------------------------------
    // Test 15: Fast path is actually taken (no ties = no deferred)
    //   We verify by checking the result matches standard exactly,
    //   including merge order (which would differ if tie path was used).
    // ---------------------------------------------------------------
    #[test]
    fn fast_path_exercised() {
        let edges = vec![(0, 1, 1), (0, 2, 2), (1, 2, 3)];
        // Every edge is a unique distance. Fast path should handle all.
        assert_eq!(run_conservative(3, &edges), run_standard(3, &edges));
    }

    // ---------------------------------------------------------------
    // Test 16: Mixed — some batches have ties, some don't
    // ---------------------------------------------------------------
    #[test]
    fn mixed_ties_and_no_ties() {
        // d=1: tie batch (0,1) and (2,3) — two independent cliques, both merge
        // d=2: no tie — edge (0,2), not saturated (need size 2*2=4 edges)
        // d=3: no tie — edge (1,3), now edges[{0,1}][{2,3}] = 2, need 4
        // d=4: no tie — edge (0,3), edges = 3, need 4
        // d=5: no tie — edge (1,2), edges = 4 = 2*2, merge!
        let edges = vec![
            (0, 1, 1),
            (2, 3, 1),
            (0, 2, 2),
            (1, 3, 3),
            (0, 3, 4),
            (1, 2, 5),
        ];

        let history = run_conservative(4, &edges);
        let dists: Vec<u64> = history.iter().map(|h| h.2).collect();

        // d=1: 2 merges (tie path)
        assert_eq!(dists.iter().filter(|&&d| d == 1).count(), 2);
        // d=5: 1 merge (fast path)
        assert_eq!(dists.iter().filter(|&&d| d == 5).count(), 1);
        assert_eq!(history.len(), 3);
    }

    // ---------------------------------------------------------------
    // Test 17: Verify fast path produces correct winner/loser order
    // ---------------------------------------------------------------
    #[test]
    fn fast_path_winner_loser() {
        let edges = vec![(3, 1, 5)];
        let history = run_conservative(4, &edges);
        assert_eq!(history.len(), 1);
        // winner = min(root_3=3, root_1=1) = 1
        assert_eq!(history[0].0, 1); // winner
        assert_eq!(history[0].1, 3); // loser
        assert_eq!(history[0].2, 5); // distance
    }
}
