use crate::config::Config;
use crate::graph::WeightedGraph;
use crate::priority_queue::PriorityQueue;
use crate::random::RangeRng;
use crate::refinement::{BoundaryInfo, WhereIdEd};

const DEBUG_FM: bool = false;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_FM {
            println!($($x,)*);
        }
    };
}

pub fn two_way_refine<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    total_vertex_weights: i32,
    n_t_partition_weights: [f32; 2],
    min_cut: i32,
    boundary_info: BoundaryInfo,
    where_id_ed: WhereIdEd,
    rng: &mut RNG,
) -> (i32, BoundaryInfo, WhereIdEd)
where
    RNG: RangeRng,
{
    // ncon is 1
    two_way_cut_refine(
        config,
        graph,
        total_vertex_weights,
        n_t_partition_weights,
        min_cut,
        boundary_info,
        where_id_ed,
        rng,
    )
}

fn two_way_cut_refine<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    total_vertex_weights: i32,
    n_t_partition_weights: [f32; 2],
    mut min_cut: i32,
    mut boundary_info: BoundaryInfo,
    mut where_id_ed: WhereIdEd,
    rng: &mut RNG,
) -> (i32, BoundaryInfo, WhereIdEd)
where
    RNG: RangeRng,
{
    debug!("CALLED two_way_cut_refine");
    debug!("{:?}", graph);
    // debug!("tvwgt: {}", total_vertex_weights);
    // debug!("n_t_partition_weights: {:?}", n_t_partition_weights);
    // debug!("min_cut: {}", min_cut);
    // debug!("boundary_info: {:?}", boundary_info);
    // debug!("where_id_ed: {:?}", where_id_ed);
    let total_partition_weights = [
        (total_vertex_weights as f32 * n_t_partition_weights[0]) as i32,
        total_vertex_weights - (total_vertex_weights as f32 * n_t_partition_weights[0]) as i32,
    ];

    let limit = ((0.01 * graph.graph.n_vertices() as f32) as i32).clamp(15, 100);
    let average_vertex_weight = std::cmp::min(
        (boundary_info.partition_weights[0] + boundary_info.partition_weights[1]) / 20,
        2 * (boundary_info.partition_weights[0] + boundary_info.partition_weights[1])
            / graph.graph.n_vertices() as i32,
    );

    let mut queues = [
        PriorityQueue::create(graph.graph.n_vertices()),
        PriorityQueue::create(graph.graph.n_vertices()),
    ];

    let orig_diff = (total_partition_weights[0] - boundary_info.partition_weights[0]).abs();

    let mut moved = vec![None; graph.graph.n_vertices()];

    for _pass in 0..config.n_iterations {
        // do a number of passes
        queues.iter_mut().for_each(|q| q.reset());

        let mut min_cut_order = -1;
        let mut new_cut = min_cut;
        let init_cut = min_cut;
        let mut min_diff = (total_partition_weights[0] - boundary_info.partition_weights[0]).abs();

        let perm = crate::random::permutation(
            boundary_info.boundary_ind.len(),
            boundary_info.boundary_ind.len(),
            crate::random::Mode::Identity,
            rng,
        );

        for ii in 0..boundary_info.boundary_ind.len() {
            let i = perm[ii];
            queues[where_id_ed._where[boundary_info.boundary_ind[i]]].insert(
                boundary_info.boundary_ind[i],
                (where_id_ed.ed[boundary_info.boundary_ind[i]]
                    - where_id_ed.id[boundary_info.boundary_ind[i]]) as f32,
            );
        }

        debug!("perm: {:?}", perm);
        debug!("queues: {:?}", queues);

        let mut swaps = Vec::new();
        for n_swaps in 0..graph.graph.n_vertices() {
            let (from, to) = if total_partition_weights[0] - boundary_info.partition_weights[0]
                < total_partition_weights[1] - boundary_info.partition_weights[1]
            {
                (0, 1)
            } else {
                (1, 0)
            };

            debug!("swap queues: {:?}", queues);

            let maybe_high_gain = queues[from].pop();
            if maybe_high_gain.is_none() {
                break;
            }

            let high_gain = match maybe_high_gain {
                Some(high_gain) => high_gain,
                None => panic!(),
            };

            new_cut -= where_id_ed.ed[high_gain] - where_id_ed.id[high_gain];

            boundary_info.partition_weights[to] += graph.vertex_weights[high_gain];
            boundary_info.partition_weights[from] -= graph.vertex_weights[high_gain];

            let new_diff = (total_partition_weights[0] - boundary_info.partition_weights[0]).abs();
            if (new_cut < min_cut && new_diff <= orig_diff + average_vertex_weight)
                || (new_cut == min_cut && new_diff < min_diff)
            {
                min_cut = new_cut;
                min_diff = new_diff;
                min_cut_order = n_swaps as i32;
            } else if n_swaps as i32 - min_cut_order > limit {
                // we hit the limit, undo last move

                // Never read:
                // new_cut += where_id_ed.ed[high_gain] - where_id_ed.id[high_gain];

                boundary_info.partition_weights[to] -= graph.vertex_weights[high_gain];
                boundary_info.partition_weights[from] += graph.vertex_weights[high_gain];
                break;
            }

            debug!("Picked high_gain: {}", high_gain);

            where_id_ed._where[high_gain] = to;
            moved[high_gain] = Some(n_swaps);
            swaps.push(high_gain);

            // update the id/ed values of the affected nodes
            std::mem::swap(
                &mut where_id_ed.id[high_gain],
                &mut where_id_ed.ed[high_gain],
            );
            if where_id_ed.ed[high_gain] == 0 && graph.graph.degree(high_gain) > 0 {
                boundary_info.delete(high_gain);
            }

            for (&k, &adjacency_weight) in graph.weighted_neighbors(high_gain) {
                let k_weight = if to == where_id_ed._where[k] {
                    adjacency_weight
                } else {
                    -adjacency_weight
                };

                where_id_ed.id[k] += k_weight;
                where_id_ed.ed[k] -= k_weight;

                // update its boundary information and queue position
                debug!(
                    "bndptr: {}, ed: {}, moved: {}",
                    boundary_info.boundary_ptr[k].is_some() as i32,
                    where_id_ed.ed[k],
                    moved[k].is_none() as i32
                );
                if boundary_info.boundary_ptr[k].is_some() {
                    // if k was a boundary vertex
                    if where_id_ed.ed[k] == 0 {
                        // not a boundary vertex any more
                        boundary_info.delete(k);

                        if moved[k].is_none() {
                            // remove it if in the queues
                            queues[where_id_ed._where[k]].delete(k);
                            debug!("After delete {} {}: {:?}", where_id_ed._where[k], k, queues);
                        }
                    } else if moved[k].is_none() {
                        // if it has not been moved, update its position in the queue
                        queues[where_id_ed._where[k]]
                            .update(k, (where_id_ed.ed[k] - where_id_ed.id[k]) as f32);
                        debug!("After update: {:?}", queues);
                    }
                } else if where_id_ed.ed[k] > 0 {
                    // it will now become a boundary vertex
                    boundary_info.insert(k);

                    if moved[k].is_none() {
                        queues[where_id_ed._where[k]]
                            .insert(k, (where_id_ed.ed[k] - where_id_ed.id[k]) as f32);
                        debug!("After insert: {:?}", queues);
                    }
                }
            }
        }

        // roll back computations
        for swap in swaps.iter() {
            moved[*swap] = None;
        }

        for i_swap in (((min_cut_order + 1) as usize)..swaps.len()).rev() {
            let high_gain = swaps[i_swap];
            debug!("Unrolling high_gain {}", high_gain);

            where_id_ed._where[high_gain] = if where_id_ed._where[high_gain] == 0 {
                1
            } else {
                0
            };

            let to = where_id_ed._where[high_gain];
            let from = if to == 0 { 1 } else { 0 };

            std::mem::swap(
                &mut where_id_ed.id[high_gain],
                &mut where_id_ed.ed[high_gain],
            );

            if where_id_ed.ed[high_gain] == 0
                && boundary_info.boundary_ptr[high_gain].is_some()
                && graph.graph.degree(high_gain) > 0
            {
                boundary_info.delete(high_gain);
            } else if where_id_ed.ed[high_gain] > 0
                && boundary_info.boundary_ptr[high_gain].is_none()
            {
                boundary_info.insert(high_gain);
            }

            boundary_info.partition_weights[to] += graph.vertex_weights[high_gain];
            boundary_info.partition_weights[from] -= graph.vertex_weights[high_gain];

            for (&k, &adjacency_weight) in graph.weighted_neighbors(high_gain) {
                let k_weight = if to == where_id_ed._where[k] {
                    adjacency_weight
                } else {
                    -adjacency_weight
                };

                where_id_ed.id[k] += k_weight;
                where_id_ed.ed[k] -= k_weight;

                if boundary_info.boundary_ptr[k].is_some() && where_id_ed.ed[k] == 0 {
                    boundary_info.delete(k);
                }
                if boundary_info.boundary_ptr[k].is_none() && where_id_ed.ed[k] > 0 {
                    boundary_info.insert(k);
                }
            }
        }

        if min_cut_order <= 0 || min_cut == init_cut {
            break;
        }
    }

    debug!("EXITED two_way_cut_refine");

    (min_cut, boundary_info, where_id_ed)
}
