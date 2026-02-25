use numpy::PyReadonlyArray2;
mod bitvec;
mod cliques;
mod cone;
mod distance_matrix;
mod linkage;
mod preprocess;
use pyo3::prelude::*;

#[pyfunction(name = "persistent_homology")]
#[pyo3(signature = (points, max_dim=1,greedy=true, use_128bit=false))]
pub fn persistent_homology(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    max_dim: usize,
    greedy: bool,
    use_128bit: bool,
) -> PyResult<Vec<(i32, f32, f32)>> {
    // Reject use_128bit on Windows at runtime
    #[cfg(target_os = "windows")]
    if use_128bit {
        return Err(PyRuntimeError::new_err(
            "use_128bit=True is not supported on Windows (MSVC lacks __int128 support). \
             Use use_128bit=False or run on Linux/macOS.",
        ));
    }

    let points_f32 = points.as_array().mapv(|x| x as f32);

    let result = py.detach(move || {
        // Create the view inside the detached closure
        let points_view = points_f32.view();
        let (sorted_edges, radius) = preprocess::edgelist(points_view);
        let n_points = points_view.nrows();
        let mut tower = cone::Tower::new(n_points);

        //IF GREEDY
        if greedy {
            let mut system = linkage::complete::CompleteLinkage::new(n_points);
            let mut tower = cone::Tower::new(n_points);
            for (u, v, d) in sorted_edges {
                tower.add_edge(u, v, d); // Optional, depending on tower logic
                                         // 1. Register edge & Check condition
                if let Some((root_a, root_b)) = system.register_edge(u, v) {
                    // 2. Handle Contraction and decide winner
                    let winner = tower
                        .contract(root_a, root_b, d)
                        .expect("merging pre-merged nodes");
                    // 3. Update Tracker State
                    system.merge(root_a, root_b, winner);
                }
            }
        } else {
            let mut system = linkage::conservative::ConservativeCompleteLinkage::new(n_points);
            let mut deferred: Vec<(usize, usize)> = Vec::new();

            let len = sorted_edges.len();
            for i in 0..len {
                let (u, v, d) = sorted_edges[i];
                tower.add_edge(u, v, d);

                let next_is_same = i + 1 < len && sorted_edges[i + 1].2 == d;
                let in_tie_batch = next_is_same || !deferred.is_empty();

                if !in_tie_batch {
                    // FAST PATH — identical to standard complete linkage
                    if let Some((root_a, root_b)) = system.register_edge(u, v) {
                        let winner = tower
                            .contract(root_a, root_b, d)
                            .expect("merging pre-merged nodes");
                        system.merge(root_a, root_b, winner);
                    }
                } else {
                    // TIE PATH
                    if let Some(pair) = system.register_edge(u, v) {
                        deferred.push(pair);
                    }
                    if !next_is_same {
                        let merges = linkage::conservative::flush_ties(
                            &mut system,
                            &mut deferred,
                            |a, b| tower.contract(a, b, d).expect("merging pre-merged nodes"),
                        );
                    }
                }
            }
        }

        let barcodes = if use_128bit {
            #[cfg(not(target_os = "windows"))]
            {
                ripser::ripser128(
                    &tower.distances.data,
                    tower.distances.n,
                    max_dim as i32,
                    Some(radius),
                )
            }
            #[cfg(target_os = "windows")]
            {
                // This branch is unreachable because we return an error above,
                // but we need it for the code to compile on Windows.
                unreachable!()
            }
        } else {
            ripser::ripser(
                &tower.distances.data,
                tower.distances.n,
                max_dim as i32,
                Some(radius),
            )
        };
        // 3. Map the Barcode struct to a Tuple
        barcodes
            .into_iter()
            .map(|b| (b.dim, b.birth, b.death))
            .collect()
    });

    Ok(result)
}

#[pyfunction(name = "tower_size")]
#[pyo3(signature = (points, max_size=3, greedy=true))]
pub fn tower_size(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    max_size: usize, //default to triangles 3
    greedy: bool,
) -> PyResult<usize> {
    let points_f32 = points.as_array().mapv(|x| x as f32);

    let result = py.detach(move || {
        let points_view = points_f32.view();
        let (sorted_edges, _radius) = preprocess::edgelist(points_view);
        let n_points = points_view.nrows();
        let mut tower = cone::Tower::new(n_points);

        //IF GREEDY
        if greedy {
            let mut system = linkage::complete::CompleteLinkage::new(n_points);
            for (u, v, d) in sorted_edges {
                tower.add_edge(u, v, d); // Optional, depending on tower logic
                                         // 1. Register edge & Check condition
                if let Some((root_a, root_b)) = system.register_edge(u, v) {
                    // 2. Handle Contraction and decide winner
                    let winner = tower
                        .contract(root_a, root_b, d)
                        .expect("merging pre-merged nodes");
                    // 3. Update Tracker State
                    system.merge(root_a, root_b, winner);
                }
            }
        } else {
            let mut system = linkage::conservative::ConservativeCompleteLinkage::new(n_points);
            let mut deferred: Vec<(usize, usize)> = Vec::new();

            let len = sorted_edges.len();
            for i in 0..len {
                let (u, v, d) = sorted_edges[i];
                tower.add_edge(u, v, d);

                let next_is_same = i + 1 < len && sorted_edges[i + 1].2 == d;
                let in_tie_batch = next_is_same || !deferred.is_empty();

                if !in_tie_batch {
                    // FAST PATH — identical to standard complete linkage
                    if let Some((root_a, root_b)) = system.register_edge(u, v) {
                        let winner = tower
                            .contract(root_a, root_b, d)
                            .expect("merging pre-merged nodes");
                        system.merge(root_a, root_b, winner);
                    }
                } else {
                    // TIE PATH
                    if let Some(pair) = system.register_edge(u, v) {
                        deferred.push(pair);
                    }
                    if !next_is_same {
                        let merges = linkage::conservative::flush_ties(
                            &mut system,
                            &mut deferred,
                            |a, b| tower.contract(a, b, d).expect("merging pre-merged nodes"),
                        );
                    }
                }
            }
        }

        // IF NOT GREEDY
        let vertices: Vec<usize> = (0..n_points).collect();
        cliques::vietoris_rips_filtration_count(&tower.neighbors, vertices, Some(max_size))
    });

    Ok(result)
}
#[pyfunction(name = "filtration_size")]
#[pyo3(signature = (points, max_size=3))]
pub fn filtration_size(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    max_size: usize, //default to triagnes (3)
) -> PyResult<usize> {
    // let points_view = points.as_array();
    let points_f32 = points.as_array().mapv(|x| x as f32);

    let result = py.detach(move || {
        let points_view = points_f32.view();
        let (sorted_edges, _radius) = preprocess::edgelist(points_view);
        let n_points = points_view.nrows();
        let adj = cliques::build_rips_graph(&sorted_edges, n_points);
        let vertices: Vec<usize> = (0..n_points).collect();
        cliques::vietoris_rips_filtration_count(&adj, vertices, Some(max_size))
    });

    Ok(result)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(persistent_homology, m)?)?;
    m.add_function(wrap_pyfunction!(filtration_size, m)?)?;
    m.add_function(wrap_pyfunction!(tower_size, m)?)?;
    Ok(())
}
