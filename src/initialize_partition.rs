use crate::config::{Config, InitialPartitioningType, DEBUG_INITIALIZE_PARTITION};
use crate::graph::WeightedGraph;
use crate::random::RangeRng;
use crate::separator_refinement::GraphPyramidLevel;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_INITIALIZE_PARTITION {
            println!($($x,)*);
        }
    };
}

pub fn initialize_separator<RNG>(
    config: &Config,
    graph_pyramid: Vec<crate::coarsen::CoarseGraphResult>,
    n_i_parts: usize,
    graph_is_compressed: bool,
    rng: &mut RNG,
) -> Vec<GraphPyramidLevel>
where
    RNG: RangeRng,
{
    debug!("CALLED initialize_separator");

    let n_t_partition_weights = [0.5, 0.5];

    let coarsest_level = graph_pyramid.last().unwrap();

    // this is required for the cut-based part of the refinement
    let balance_multipliers = setup_two_way_balance_multipliers(
        config,
        //coarsest_level.graph,
        coarsest_level.total_vertex_weights,
        n_t_partition_weights,
    );

    let (_min_cut, _boundary_info, where_id_ed) = match config.initial_partitioning_type {
        InitialPartitioningType::EDGE => {
            let (_min_cut, where_id_ed, _boundary_info) =
                if coarsest_level.graph.graph.adjacency_lists.len() == 0 {
                    random_bisection(
                        config,
                        &coarsest_level.graph,
                        n_t_partition_weights,
                        n_i_parts,
                        coarsest_level.total_vertex_weights,
                        balance_multipliers,
                        rng,
                    )
                } else {
                    grow_bisection(
                        config,
                        &coarsest_level.graph,
                        n_t_partition_weights,
                        n_i_parts,
                        coarsest_level.total_vertex_weights,
                        balance_multipliers,
                        rng,
                    )
                };

            let (_min_cut, mut where_id_ed, boundary_info) =
                crate::refinement::compute_two_way_partitioning_params(
                    config,
                    &coarsest_level.graph,
                    where_id_ed,
                );

            let (min_cut, boundary_info, _where) = crate::separator::construct_separator(
                config,
                &coarsest_level.graph,
                boundary_info,
                &where_id_ed,
                graph_is_compressed,
                rng,
            );
            where_id_ed._where = _where;

            (min_cut, boundary_info, where_id_ed)
        }
        InitialPartitioningType::NODE => panic!("Not implemented"),
        _ => panic!("WTF"),
    };

    let result = graph_pyramid[..graph_pyramid.len() - 1]
        .iter()
        .map(|pyramid_level| GraphPyramidLevel {
            graph: pyramid_level.graph.clone(),
            coarsening_map: pyramid_level.coarsening_map.clone(),
            total_vertex_weights: pyramid_level.total_vertex_weights,
            coarser_graph_where: vec![],
        })
        .chain(std::iter::once(GraphPyramidLevel {
            graph: coarsest_level.graph.clone(), // Eek, this should be unnecessary
            coarsening_map: coarsest_level.coarsening_map.clone(), // Eek
            coarser_graph_where: where_id_ed._where,
            total_vertex_weights: coarsest_level.total_vertex_weights,
        }))
        .collect();

    debug!("EXITED initialize_separator");

    result
}

fn random_bisection<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    n_t_partition_weights: [f32; 2],
    n_i_parts: usize,
    total_vertex_weights: i32,
    balance_multipliers: Vec<f32>,
    rng: &mut RNG,
) -> (
    i32,
    crate::refinement::WhereIdEd,
    crate::refinement::BoundaryInfo,
)
where
    RNG: RangeRng,
{
    // Some duplication here with grow_bisection

    debug!("CALLED random_bisection");
    debug!("n_i_parts: {}", n_i_parts);

    // ncon is 1
    let mut boundary_info = crate::refinement::BoundaryInfo {
        partition_weights: [0, 0, 0],
        boundary_ind: Vec::new(),
        boundary_ptr: vec![None; graph.graph.n_vertices()],
    };

    let mut where_id_ed = crate::refinement::WhereIdEd {
        _where: vec![0; graph.graph.n_vertices()],
        id: vec![0; graph.graph.n_vertices()],
        ed: vec![0; graph.graph.n_vertices()],
    };

    let mut best_cut = 0;
    let mut best_where = vec![0; graph.graph.n_vertices()];

    let zero_max_partition_weight =
        (config.ub_factors[0] * total_vertex_weights as f32 * n_t_partition_weights[1]) as i32;

    for i_n_bfs in 0..n_i_parts {
        let mut _where = vec![1; graph.graph.n_vertices()];

        if i_n_bfs > 0 {
            let perm = crate::random::permutation(
                graph.graph.n_vertices(),
                graph.graph.n_vertices() / 2,
                crate::random::Mode::Identity,
                rng,
            );

            boundary_info.partition_weights[1] = total_vertex_weights;
            boundary_info.partition_weights[0] = 0;

            for i in perm {
                if boundary_info.partition_weights[0] + graph.vertex_weights[i]
                    < zero_max_partition_weight
                {
                    _where[i] = 0;
                    boundary_info.partition_weights[0] += graph.vertex_weights[i];
                    boundary_info.partition_weights[1] -= graph.vertex_weights[i];
                    if boundary_info.partition_weights[0] > zero_max_partition_weight {
                        break;
                    }
                }
            }
        }

        // do some partition refinement
        let partitioning_params = crate::refinement::compute_two_way_partitioning_params(
            config,
            graph,
            crate::refinement::WhereIdEd {
                _where,
                id: vec![0; graph.graph.n_vertices()],
                ed: vec![0; graph.graph.n_vertices()],
            },
        );
        let min_cut = partitioning_params.0;
        where_id_ed = partitioning_params.1;
        boundary_info = partitioning_params.2;

        let balance_results = crate::balance::balance_two_way(
            config,
            graph,
            boundary_info,
            where_id_ed,
            min_cut,
            n_t_partition_weights,
            total_vertex_weights,
            &balance_multipliers,
            rng,
        );
        let min_cut = balance_results.0;
        boundary_info = balance_results.1;
        where_id_ed = balance_results.2;

        let new_boundary = crate::fm::two_way_refine(
            config,
            graph,
            total_vertex_weights,
            n_t_partition_weights,
            min_cut,
            boundary_info,
            where_id_ed,
            rng,
        );
        let min_cut = new_boundary.0;
        boundary_info = new_boundary.1;
        where_id_ed = new_boundary.2;

        if i_n_bfs == 0 || best_cut > min_cut {
            best_cut = min_cut;
            best_where = where_id_ed._where.clone();
            if best_cut == 0 {
                debug!("BREAKING grow_bisection");
                break;
            }
        }
    }

    let min_cut = best_cut;
    where_id_ed._where = best_where;

    debug!("EXITED grow_bisection");

    (min_cut, where_id_ed, boundary_info)
}

fn setup_two_way_balance_multipliers(
    config: &Config,
    // graph: WeightedGraph,
    total_vertex_weights: i32,
    n_t_partition_weights: [f32; 2],
) -> Vec<f32> {
    if config.n_balancing_constraints != 1 {
        panic!("Not implemented");
    }

    (0..2)
        .map(|i| 1.0 / (total_vertex_weights as f32 * n_t_partition_weights[i] as f32))
        .collect()
}

fn grow_bisection<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    n_t_partition_weights: [f32; 2],
    n_i_parts: usize,
    total_vertex_weights: i32,
    balance_multipliers: Vec<f32>,
    rng: &mut RNG,
) -> (
    i32,
    crate::refinement::WhereIdEd,
    crate::refinement::BoundaryInfo,
)
where
    RNG: RangeRng,
{
    debug!("CALLED grow_bisection");
    debug!("n_i_parts: {}", n_i_parts);

    // ncon is 1
    let mut boundary_info = crate::refinement::BoundaryInfo {
        partition_weights: [0, 0, 0],
        boundary_ind: Vec::new(),
        boundary_ptr: vec![None; graph.graph.n_vertices()],
    };

    let mut where_id_ed = crate::refinement::WhereIdEd {
        _where: vec![0; graph.graph.n_vertices()],
        id: vec![0; graph.graph.n_vertices()],
        ed: vec![0; graph.graph.n_vertices()],
    };

    let mut best_cut = 0;
    let mut best_where = vec![0; graph.graph.n_vertices()];
    let mut queue = vec![0; graph.graph.n_vertices()];
    // let mut touched = vec![false; graph.graph.n_vertices()];

    let one_max_partition_weight =
        (config.ub_factors[0] * total_vertex_weights as f32 * n_t_partition_weights[1]) as i32;
    let one_min_partition_weight = ((1.0 / config.ub_factors[0])
        * total_vertex_weights as f32
        * n_t_partition_weights[1]) as i32;

    for i_n_bfs in 0..n_i_parts {
        where_id_ed._where = vec![1; graph.graph.n_vertices()];
        let mut touched = vec![false; graph.graph.n_vertices()];

        boundary_info.partition_weights[0] = 0;
        boundary_info.partition_weights[1] = total_vertex_weights;

        queue[0] = rng.gen_range(0..graph.graph.n_vertices());
        touched[queue[0]] = true;

        let mut first = 0;
        let mut last = 1;
        let mut n_left = graph.graph.n_vertices() - 1;
        let mut drain = false;

        // start the BFS from queue to get a partition
        loop {
            if first == last {
                // empty.  disconnected graph!
                if n_left == 0 || drain {
                    break;
                }

                let k = rng.gen_range(0..n_left);
                let i = (0..graph.graph.n_vertices())
                    .filter(|i| !touched[*i])
                    .nth(k)
                    .unwrap();

                queue[0] = i;
                touched[i] = true;
                first = 0;
                last = 1;
                n_left -= 1;
            }

            let i = queue[first];
            first += 1;

            if boundary_info.partition_weights[0] > 0
                && boundary_info.partition_weights[1] - graph.vertex_weights[i]
                    < one_min_partition_weight
            {
                drain = true;
                continue;
            }

            where_id_ed._where[i] = 0;
            boundary_info.partition_weights[0] += graph.vertex_weights[i];
            boundary_info.partition_weights[1] -= graph.vertex_weights[i];

            if boundary_info.partition_weights[1] <= one_max_partition_weight {
                break;
            }

            drain = false;
            for &k in graph.graph.neighbors(i) {
                if !touched[k] {
                    queue[last] = k;
                    last += 1;
                    touched[k] = true;
                    n_left -= 1;
                }
            }
        }

        // check to see if we hit any bad limiting cases
        if boundary_info.partition_weights[1] == 0 {
            where_id_ed._where[rng.gen_range(0..graph.graph.n_vertices())] = 1;
        }
        if boundary_info.partition_weights[0] == 0 {
            where_id_ed._where[rng.gen_range(0..graph.graph.n_vertices())] = 0;
        }

        // do some partition refinement
        let partitioning_params =
            crate::refinement::compute_two_way_partitioning_params(config, graph, where_id_ed);
        let min_cut = partitioning_params.0;
        where_id_ed = partitioning_params.1;
        boundary_info = partitioning_params.2;

        let balance_results = crate::balance::balance_two_way(
            config,
            graph,
            boundary_info,
            where_id_ed,
            min_cut,
            n_t_partition_weights,
            total_vertex_weights,
            &balance_multipliers,
            rng,
        );
        let min_cut = balance_results.0;
        boundary_info = balance_results.1;
        where_id_ed = balance_results.2;

        let new_boundary = crate::fm::two_way_refine(
            config,
            graph,
            total_vertex_weights,
            n_t_partition_weights,
            min_cut,
            boundary_info,
            where_id_ed,
            rng,
        );
        let min_cut = new_boundary.0;
        boundary_info = new_boundary.1;
        where_id_ed = new_boundary.2;

        if i_n_bfs == 0 || best_cut > min_cut {
            best_cut = min_cut;
            best_where = where_id_ed._where.clone();
            if best_cut == 0 {
                debug!("BREAKING grow_bisection");
                break;
            }
        }
    }

    let min_cut = best_cut;
    where_id_ed._where = best_where;

    debug!("EXITED grow_bisection");

    (min_cut, where_id_ed, boundary_info)
}
