pub fn compute_load_imbalance_diff(
    partition_weights: [i32; 3],
    n_parts: usize,
    pijbm: Vec<f32>,
    ub_factors: Vec<f32>,
) -> f32 {
    let mut max = -1.0;

    // ncon is 1
    for j in 0..n_parts {
        let cur = partition_weights[j] as f32 * pijbm[j] - ub_factors[0];
        if cur > max {
            max = cur;
        }
    }

    max
}
