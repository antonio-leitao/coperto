use numpy::ndarray::ArrayView2;

#[inline(always)]
fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    if len == 0 {
        return 0.0;
    }
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut sum4 = 0.0;
    let mut sum5 = 0.0;
    let mut sum6 = 0.0;
    let mut sum7 = 0.0;
    let mut i = 0;
    let upper = len - (len % 8); // Unroll 8 times
    while i < upper {
        let d0 = a[i] - b[i];
        sum0 += d0 * d0;
        let d1 = a[i + 1] - b[i + 1];
        sum1 += d1 * d1;
        let d2 = a[i + 2] - b[i + 2];
        sum2 += d2 * d2;
        let d3 = a[i + 3] - b[i + 3];
        sum3 += d3 * d3;
        let d4 = a[i + 4] - b[i + 4];
        sum4 += d4 * d4;
        let d5 = a[i + 5] - b[i + 5];
        sum5 += d5 * d5;
        let d6 = a[i + 6] - b[i + 6];
        sum6 += d6 * d6;
        let d7 = a[i + 7] - b[i + 7];
        sum7 += d7 * d7;
        i += 8;
    }
    let mut total_sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    while i < len {
        let d = a[i] - b[i];
        total_sum += d * d;
        i += 1;
    }
    total_sum
}

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
