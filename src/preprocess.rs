use caravela::l2sq;
use numpy::ndarray::ArrayView2;

pub fn edgelist(points: ArrayView2<f32>) -> (Vec<(usize, usize, f32)>, f32) {
    let n = points.nrows();
    let mut row_max = vec![0.0f32; n];

    let cap = n * (n - 1) / 2;
    let mut edges_with_dist: Vec<(usize, usize, f32)> = Vec::with_capacity(cap);

    // Single compute pass
    for i in 0..n {
        let point_i = points.row(i);
        let point_i_slice: &[f32] = point_i
            .as_slice()
            .expect("Row data should be contiguous and sliceable");
        for j in (i + 1)..n {
            let point_j = points.row(j);
            let point_j_slice: &[f32] = point_j
                .as_slice()
                .expect("Row data should be contiguous and sliceable");
            let d = l2sq(point_i_slice, point_j_slice).sqrt();
            edges_with_dist.push((i, j, d));

            if d > row_max[i] {
                row_max[i] = d;
            }
            if d > row_max[j] {
                row_max[j] = d;
            }
        }
    }

    let threshold = row_max.iter().cloned().fold(f32::INFINITY, f32::min);

    // Filter
    edges_with_dist.retain(|e| e.2 <= threshold);

    // // Build matrix
    // let mut matrix = DistanceMatrix::new(n);
    // for &(i, j, d) in &edges_with_dist {
    //     matrix.set(i as usize, j as usize, d);
    // }

    // Sort
    edges_with_dist.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));

    (edges_with_dist, threshold)
}
