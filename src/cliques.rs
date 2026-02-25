use crate::bitvec::{BitSet, BitSetRef};

pub fn vietoris_rips_filtration_count(
    adj: &Vec<BitSet>,
    vertex_list: Vec<usize>,
    max_dim: Option<usize>,
) -> usize {
    let mut count = vertex_list.len();

    // Default to a high number if not specified, or just vertex_list len if explicitly 0?
    // Usually if max_dim is None we go as deep as possible, but let's default to max size = vertices.
    let max_dim = max_dim.unwrap_or(vertex_list.len());

    // If max_dim is 0 or 1, we only count vertices (Size 1)
    if max_dim <= 1 {
        return count;
    }

    // Workspace needs to be size of max recursion depth.
    // If max_dim (size) is 3 (Triangles), we recurse depth 0 (Edge) and depth 1 (Triangle).
    // So we need max_dim - 1 buffers.
    let buffer_count = if max_dim > 1 { max_dim - 1 } else { 0 };

    let mut workspace: Vec<Vec<u64>> = (0..buffer_count)
        .map(|_| Vec::with_capacity(vertex_list.len()))
        .collect();

    for (u, neighbours) in adj.iter().enumerate() {
        let n_ref = BitSetRef {
            bits: &neighbours.bits,
        };
        count_cofaces_recursive(adj, max_dim, u, n_ref, &mut count, &mut workspace, 0);
    }
    count
}

fn count_cofaces_recursive<'a>(
    adj: &Vec<BitSet>,
    max_dim: usize, // Interpretation: Max Simplex Size (e.g., 3 for triangles)
    last_vertex: usize,
    n: BitSetRef<'a>,
    count: &mut usize,
    workspace: &mut [Vec<u64>],
    current_dim: usize, // 0 = expanding vertex to edge (Size 2)
) {
    if workspace.is_empty() {
        return;
    }

    let (m_buffer_slice, next_workspace) = workspace.split_at_mut(1);
    let m_buffer = &mut m_buffer_slice[0];

    for v in n.iter() {
        if v <= last_vertex {
            continue;
        }

        // We found a neighbor, so we formed a simplex of Size = (current_dim + 2)
        // e.g., at dim 0, we found an edge (Size 2).
        *count += 1;

        // Recursion Guard:
        // We are currently at Size = current_dim + 2.
        // We can only recurse if we are looking for simplices of Size = current_dim + 3.
        // So we recurse if (current_dim + 2) < max_dim.
        if current_dim + 2 < max_dim {
            if let Some(v_neighbours) = adj.get(v as usize) {
                crate::bitvec::intersection_into(n.bits, &v_neighbours.bits, m_buffer);
                let m_ref = BitSetRef { bits: m_buffer };

                if !m_ref.is_empty() {
                    count_cofaces_recursive(
                        adj,
                        max_dim,
                        v,
                        m_ref,
                        count,
                        next_workspace,
                        current_dim + 1,
                    );
                }
            }
        }
    }
}

pub fn build_rips_graph(edges: &[(usize, usize, f32)], n: usize) -> Vec<BitSet> {
    let mut adj = vec![BitSet::new(); n];

    for &(u, v, _) in edges {
        if u == v {
            continue;
        }
        adj[u].insert(v);
        adj[v].insert(u);
    }
    adj
}
