use crate::config::{Config, Index, Real};
use crate::graph::WeightedGraph;
use crate::priority_queue::PriorityQueue;
use crate::random;
use crate::refinement::{BoundaryInfo, WhereIdEd};

pub fn balance_two_way<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    boundary_info: BoundaryInfo,
    where_id_ed: WhereIdEd,
    min_cut: Index,
    n_t_partition_weights: [Real; 2],
    total_vertex_weights: Index,
    balance_multipliers: &Vec<f32>,
    rng: &mut RNG,
) -> (Index, BoundaryInfo, WhereIdEd)
where
    RNG: random::RangeRng,
{
    if crate::mc_util::compute_load_imbalance_diff(
        boundary_info.partition_weights,
        2,
        balance_multipliers,
        &config.ub_factors,
    ) <= 0.0
    {
        return (min_cut, boundary_info, where_id_ed);
    }

    // ncon better be 1

    // return right away if the balance is OK
    if (n_t_partition_weights[0] * total_vertex_weights as Real
        - boundary_info.partition_weights[0] as Real)
        .abs()
        < (3 * total_vertex_weights as usize / graph.graph.n_vertices()) as Real
    {
        return (min_cut, boundary_info, where_id_ed);
    }

    if boundary_info.len() > 0 {
        return boundary_two_way_balance(
            graph,
            boundary_info,
            where_id_ed,
            min_cut,
            n_t_partition_weights,
            total_vertex_weights,
            rng,
        );
    } else {
        panic!("Not implemented");
    }
}

fn boundary_two_way_balance<RNG>(
    graph: &WeightedGraph,
    mut boundary_info: BoundaryInfo,
    mut where_id_ed: WhereIdEd,
    mut min_cut: Index,
    n_t_partition_weights: [Real; 2],
    total_vertex_weights: Index,
    rng: &mut RNG,
) -> (Index, BoundaryInfo, WhereIdEd)
where
    RNG: random::RangeRng,
{
    // determine from which domain you will be moving data
    let total_partition_weights = [
        (total_vertex_weights as Real * n_t_partition_weights[0]) as Index,
        total_vertex_weights - (total_vertex_weights as Real * n_t_partition_weights[0]) as Index,
    ];

    let min_diff = (total_partition_weights[0] - boundary_info.partition_weights[0]).abs();
    let from = if boundary_info.partition_weights[0] < total_partition_weights[0] {
        1
    } else {
        0
    };
    let to = (from + 1) % 2;

    let mut queue = PriorityQueue::create(graph.graph.n_vertices());

    let mut moved = vec![None; graph.graph.n_vertices()];

    // insert the boundary nodes of the proper partition whose size is OK in the priority queue
    let perm = random::permutation(
        boundary_info.len(),
        boundary_info.len() / 5,
        random::Mode::Identity,
        rng,
    );

    for i in perm {
        if where_id_ed._where[boundary_info.boundary_ind[i]] == from
            && graph.vertex_weights[boundary_info.boundary_ind[i]] <= min_diff
        {
            queue.insert(
                boundary_info.boundary_ind[i],
                (where_id_ed.ed[boundary_info.boundary_ind[i]]
                    - where_id_ed.id[boundary_info.boundary_ind[i]]) as Real,
            );
        }
    }

    for n_swaps in 0..graph.graph.n_vertices() {
        if queue.is_empty() {
            break;
        }

        let high_gain = queue.pop().unwrap();

        if boundary_info.partition_weights[to] + graph.vertex_weights[high_gain]
            > total_partition_weights[to]
        {
            break;
        }

        min_cut -= where_id_ed.ed[high_gain] - where_id_ed.id[high_gain];
        boundary_info.partition_weights[to] += graph.vertex_weights[high_gain];
        boundary_info.partition_weights[from] -= graph.vertex_weights[high_gain];

        where_id_ed._where[high_gain] = to;
        moved[high_gain] = Some(n_swaps);

        // update the id/ed values of the affected nodes
        std::mem::swap(
            &mut where_id_ed.id[high_gain],
            &mut where_id_ed.ed[high_gain],
        );

        if where_id_ed.ed[high_gain] == 0 && graph.graph.degree(high_gain) > 0 {
            boundary_info.delete(high_gain);
        }

        for (&k, &k_weight) in graph.weighted_neighbors(high_gain) {
            let k_weight_signed = if to == where_id_ed._where[k] {
                k_weight
            } else {
                -k_weight
            };

            where_id_ed.id[k] += k_weight_signed;
            where_id_ed.ed[k] -= k_weight_signed;

            // update its boundary information and queue position
            if boundary_info.boundary_ptr[k].is_none() {
                // if k was a boundary vertex
                if where_id_ed.ed[k] == 0 {
                    // not a boundary vertex any more
                    boundary_info.delete(k);
                    if moved[k].is_none()
                        && where_id_ed._where[k] == from
                        && graph.vertex_weights[k] <= min_diff
                    {
                        // remove it if in the queues
                        queue.delete(k);
                    }
                } else {
                    // if it has not been moved, update its position in the queue
                    if moved[k].is_none()
                        && where_id_ed._where[k] == from
                        && graph.vertex_weights[k] <= min_diff
                    {
                        queue.update(k, (where_id_ed.ed[k] - where_id_ed.id[k]) as Real);
                    }
                }
            } else {
                if where_id_ed.ed[k] > 0 {
                    // it will now become a boundary vertex
                    boundary_info.insert(k);
                    if moved[k].is_none()
                        && where_id_ed._where[k] == from
                        && graph.vertex_weights[k] <= min_diff
                    {
                        queue.insert(k, (where_id_ed.ed[k] - where_id_ed.id[k]) as Real);
                    }
                }
            }
        }
    }

    (min_cut, boundary_info, where_id_ed)
}
