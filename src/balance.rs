use crate::config::Config;

pub fn balance_two_way(
    config: &Config,
    partition_weights: [i32; 3],
    balance_multipliers: Vec<f32>,
) {
    if crate::mc_util::compute_load_imbalance_diff(
        partition_weights,
        2,
        balance_multipliers,
        config.ub_factors,
    ) <= 0.0
    {
        return;
    }

    panic!("Not implemented");
}
